"""Etapa 7: leave-one-arm-out -- a pergunta que o trabalho de fato faz.

Os numeros da etapa 5 sao medidos em **conteudo inedito mas implementacao
vista**: os 7 arms estao todos no treino. A tese e sobre generalizar entre
implementacoes, e isso exige perguntar a um plugin que a rede nunca ouviu. Sem
esta etapa, os 44% nao sustentam a frase "funciona num plugin novo".

Cada execucao treina em N-1 arms e responde com o MESMO modelo:

- `transferencia`: consulta do arm retirado contra o catalogo dos vistos. E a
  tarefa real.
- `vistos`: consulta dos arms vistos contra o mesmo catalogo. E o controle
  interno, mas **nao** e a comparacao certa -- ele mistura 6 arms de dificuldade
  muito diferente. A comparacao certa e contra o proprio arm na etapa 5, onde a
  pergunta e o catalogo sao identicos e a unica diferenca e ter estado no treino.

O recorte preserva o cruzamento da grade, entao o alvo exato da troca de codigos
continua existindo dentro dele -- a extensao da fase 2 sobrevive ao leave-one-out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_TECHNIQUE = "contrastive_aux"
DEFAULT_ROOT = Path("datasets/disent")
DEFAULT_OUTPUT = Path("results/disent/etapa5/loo")
#: Estratos do roster, para agregar o resultado pelo eixo que o desenho previu.
STRATA: Dict[str, str] = {
    "pedalboard-tanh": "S1", "lsp-tanh": "S1",
    "lsp-hardclip": "S2", "lsp-arctan": "S2", "lsp-sine": "S2",
    "byod-mxr": "S3", "byod-bigmuff": "S3",
}


def _evaluate_held_out(run_dir: Path, root: Path, held_out: str, seen: Sequence[str],
                       batch: int = 64) -> List[Dict[str, object]]:
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import DisentModel, EncoderConfig, HeadConfig
    from gefx.disent.retrieval import retrieve_by_arm
    from gefx.disent.train import embed, split_frames

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = DisentModel(EncoderConfig.from_dict(manifest["config"]["encoder"]),
                        HeadConfig(**manifest["heads"]))
    model.load_weights(run_dir / "weights")
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")

    frames = split_frames(root)
    catalog = frames["catalog"][frames["catalog"]["arm"].isin(list(seen))]
    catalog = catalog.reset_index(drop=True)
    z_catalog = embed(model, FeatureStore(root, catalog, "Spec"), standardizer, batch)

    rows: List[Dict[str, object]] = []
    for label, arms in (("transferencia", [held_out]), ("vistos", list(seen))):
        queries = frames["query"][frames["query"]["arm"].isin(arms)].reset_index(drop=True)
        z_query = embed(model, FeatureStore(root, queries, "Spec"), standardizer, batch)
        # `same_arm=True` na transferencia nao afrouxa nada: o arm retirado nao
        # esta no catalogo por construcao, entao nao ha o que excluir. Em
        # `vistos` a exclusao vale e e a da etapa 5.
        result = retrieve_by_arm(queries, catalog, z_query, z_catalog,
                                 same_arm=(label == "transferencia"), metric="cosine")
        overall = result.metrics["overall"]
        rows.append({
            "arm_retirado": held_out, "estrato": STRATA.get(held_out, "?"),
            "condicao": label,
            "drive_exact": float(overall["drive_level"]["exact"]),
            "within_one": float(overall["drive_level"]["within_one"]),
            "mae_db": float(overall["mae_db"]), "n": int(overall["n"]),
        })
    return rows


def leave_one_arm_out(
    root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    technique: str = DEFAULT_TECHNIQUE,
    arms: Optional[Sequence[str]] = None,
    steps: int = 4000,
    seed: int = 20260908,
    verbose: bool = True,
) -> pd.DataFrame:
    """Uma execucao por arm retirado. Reaproveita o que ja estiver em disco."""
    from gefx.disent.sidecar import arm_dirs
    from gefx.disent.train import TrainConfig, train

    root, output_dir = Path(root), Path(output_dir)
    todos = list(arms) if arms else [path.name for path in arm_dirs(root)]
    if len(todos) < 3:
        raise ValueError(f"leave-one-out precisa de ao menos 3 arms, ha {len(todos)}")

    rows: List[Dict[str, object]] = []
    for held_out in todos:
        seen = [arm for arm in todos if arm != held_out]
        run_dir = output_dir / held_out
        if verbose:
            print(f"[{held_out} fora]", flush=True)
        if not (run_dir / "run.json").exists():
            train(TrainConfig(technique=technique, arms=tuple(seen), steps=steps,
                              seed=seed, eval_every=0, output_dir=run_dir), verbose=False)
        rows.extend(_evaluate_held_out(run_dir, root, held_out, seen))
        if verbose:
            for row in rows[-2:]:
                print(f"  {row['condicao']:14s} {row['drive_exact']*100:5.1f}%  "
                      f"{row['mae_db']:5.2f} dB", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "resumo.csv", index=False)
    return pd.DataFrame(rows)


def transfer_cost(
    loo: pd.DataFrame, etapa5_metrics: Path = Path("results/disent/etapa5")
        / DEFAULT_TECHNIQUE / "metrics.json",
) -> pd.DataFrame:
    """Custo de nunca ter visto a implementacao, arm por arm.

    Compara cada arm retirado com **ele mesmo** na etapa 5, onde a pergunta e o
    catalogo sao os mesmos (uma consulta contra os outros 6 arms) e a unica
    diferenca e ter estado no treino. Comparar com a coluna `vistos` seria
    errado: ela mistura 6 arms de dificuldade muito diferente e o `byod-bigmuff`
    sozinho a puxa vários pontos.
    """
    reference = json.loads(Path(etapa5_metrics).read_text(encoding="utf-8"))["per_query_arm"]
    rows: List[Dict[str, object]] = []
    for _, row in loo[loo["condicao"] == "transferencia"].iterrows():
        arm = str(row["arm_retirado"])
        if arm not in reference:
            continue
        visto = float(reference[arm]["drive_level"]["exact"])
        rows.append({
            # O estrato sai de `STRATA`, e nao da coluna: assim a funcao le
            # tambem um resumo gravado por uma versao que nao tinha a coluna.
            "arm": arm, "estrato": STRATA.get(arm, str(row.get("estrato", "?"))),
            "visto": visto, "inedito": float(row["drive_exact"]),
            "custo_pontos": (float(row["drive_exact"]) - visto) * 100,
            "visto_mae_db": float(reference[arm]["mae_db"]),
            "inedito_mae_db": float(row["mae_db"]),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["estrato", "arm"]).reset_index(drop=True)
