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
                       catalog_arms: Optional[Sequence[str]] = None, batch: int = 64,
                       extra: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
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
    # O catalogo pode ser maior que o treino: na curva de diversidade ele fica
    # **fixo** nos 6 arms enquanto o treino cresce de 1 a 6, senao a comparacao
    # entre pontos da curva mistura "treinou com mais" com "tem mais candidato".
    # Catalogo nao e treino -- e dado de consulta.
    pool = list(catalog_arms) if catalog_arms is not None else list(seen)
    catalog = frames["catalog"][frames["catalog"]["arm"].isin(pool)]
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
                                 same_arm=(held_out not in pool and label == "transferencia"),
                                 metric="cosine")
        overall = result.metrics["overall"]
        rows.append({
            "arm_retirado": held_out, "estrato": STRATA.get(held_out, "?"),
            "condicao": label, **(extra or {}),
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
    seeds: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Uma execucao por (arm retirado, semente). Reaproveita o que ja esta em disco.

    Com uma semente so, o custo de transferencia de um arm e um numero sem barra:
    800 consultas por arm contra 5.600 do agregado, e a dispersao entre sementes
    do agregado (0,8 ponto) nao limita a de um arm sozinho. Sementes adicionais
    sao o unico jeito de saber se `-8,8` no estrato S3 e um resultado ou uma
    execucao. A primeira semente da lista grava em `<arm>/` -- e a que ja esta em
    disco -- e as demais em `<arm>_s<semente>/`.
    """
    from gefx.disent.sidecar import arm_dirs
    from gefx.disent.train import TrainConfig, train

    root, output_dir = Path(root), Path(output_dir)
    todos = list(arms) if arms else [path.name for path in arm_dirs(root)]
    if len(todos) < 3:
        raise ValueError(f"leave-one-out precisa de ao menos 3 arms, ha {len(todos)}")
    todas_sementes = list(seeds) if seeds else [seed]

    rows: List[Dict[str, object]] = []
    for held_out in todos:
        seen = [arm for arm in todos if arm != held_out]
        for indice, semente in enumerate(todas_sementes):
            run_dir = output_dir / (held_out if indice == 0 else f"{held_out}_s{semente}")
            if verbose:
                print(f"[{held_out} fora, semente {semente}]", flush=True)
            if not (run_dir / "run.json").exists():
                train(TrainConfig(technique=technique, arms=tuple(seen), steps=steps,
                                  seed=semente, eval_every=0, output_dir=run_dir),
                      verbose=False)
            rows.extend(_evaluate_held_out(run_dir, root, held_out, seen,
                                           extra={"seed": semente}))
            if verbose:
                for row in rows[-2:]:
                    print(f"  {row['condicao']:14s} {row['drive_exact']*100:5.1f}%  "
                          f"{row['mae_db']:5.2f} dB", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "resumo.csv", index=False)
    return pd.DataFrame(rows)


# --- B2 e B3: quantas implementacoes o treino precisa ver? ---------------------
#: O arm retirado da curva. E o `byod-mxr` porque ele e o caso interessante: no
#: leave-one-out ele custou -8,8 pontos treinando com 6 arms. Se diversidade
#: comprasse transferencia, e nele que a compra apareceria. O `byod-bigmuff` nao
#: serve -- ele cai ao acaso e nao sobraria dinamica para medir nada.
DIVERSITY_HELD_OUT = "byod-mxr"

#: Ordem de acumulo do treino. Declarada, e nao sorteada, porque a ordem define o
#: que cada ponto significa: os dois primeiros sao o mesmo estrato (S1), os tres
#: seguintes acrescentam o S2 e o ultimo acrescenta o S3. A curva mede entao duas
#: coisas de uma vez -- quantidade e variedade -- e a coluna `estratos` e o que
#: separa as duas na leitura.
DIVERSITY_ORDER: Sequence[str] = (
    "pedalboard-tanh", "lsp-tanh", "lsp-hardclip", "lsp-arctan", "lsp-sine",
    "byod-bigmuff",
)


def arm_diversity_curve(
    root: Path = DEFAULT_ROOT,
    output_dir: Path = Path("results/disent/etapa5/diversidade"),
    held_out: str = DIVERSITY_HELD_OUT,
    order: Sequence[str] = DIVERSITY_ORDER,
    technique: str = DEFAULT_TECHNIQUE,
    steps: int = 4000,
    seed: int = 20260908,
    reuse: Optional[Path] = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """B2 e B3 na mesma curva: treinar com 1, 2, ... N-1 implementacoes.

    B2 (um arm so) e B3 (N-1 arms) do plano original sao os dois extremos disto,
    e medi-los isolados responderia menos: a pergunta que o leave-one-out abriu e
    se **diversidade de implementacao** compra transferencia, e isso e uma curva,
    nao dois pontos.

    O catalogo fica fixo nos N-1 arms em todos os pontos -- so a pertinencia ao
    treino varia. O ultimo ponto e, por construcao, a execucao do leave-one-out
    para o mesmo arm; ela e reaproveitada de `reuse` em vez de retreinada, o que
    faz a curva terminar exatamente no numero ja publicado da etapa 7.
    """
    from gefx.disent.train import TrainConfig, train

    root, output_dir = Path(root), Path(output_dir)
    order = [arm for arm in order if arm != held_out]
    pool = list(order)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for k in range(1, len(order) + 1):
        treinados = order[:k]
        run_dir = output_dir / f"k{k}"
        reuso = Path(reuse) / held_out if reuse else None
        if k == len(order) and reuso is not None and (reuso / "run.json").exists():
            run_dir = reuso
        if verbose:
            print(f"[k={k}] {', '.join(treinados)}", flush=True)
        if not (run_dir / "run.json").exists():
            # `evaluate_at_end=False`: com uma implementacao so no recorte a
            # tarefa entre implementacoes nao existe por dentro da execucao, e
            # nos outros pontos a avaliacao interna seria sobre um catalogo que
            # muda de tamanho -- incomparavel entre pontos. Quem pontua e o arm
            # retirado, contra o catalogo fixo, logo abaixo.
            train(TrainConfig(technique=technique, arms=tuple(treinados), steps=steps,
                              seed=seed, eval_every=0, evaluate_at_end=False,
                              output_dir=run_dir), verbose=False)
        novas = _evaluate_held_out(
            run_dir, root, held_out, treinados, catalog_arms=pool,
            extra={"k": k, "arms_treinados": "|".join(treinados),
                   "estratos": len({STRATA.get(arm, "?") for arm in treinados}),
                   "run_dir": str(run_dir)},
        )
        rows.extend(novas)
        if verbose:
            for row in novas:
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
    transferencia = loo[loo["condicao"] == "transferencia"]
    rows: List[Dict[str, object]] = []
    for arm, grupo in transferencia.groupby("arm_retirado", sort=False):
        arm = str(arm)
        if arm not in reference:
            continue
        visto = float(reference[arm]["drive_level"]["exact"])
        inedito = grupo["drive_exact"].astype(float)
        rows.append({
            # O estrato sai de `STRATA`, e nao da coluna: assim a funcao le
            # tambem um resumo gravado por uma versao que nao tinha a coluna.
            "arm": arm, "estrato": STRATA.get(arm, str(grupo.iloc[0].get("estrato", "?"))),
            "visto": visto, "inedito": float(inedito.mean()),
            "custo_pontos": (float(inedito.mean()) - visto) * 100,
            # Media entre sementes, e a amplitude ao lado: com uma semente so a
            # amplitude e zero, e e assim que se ve que ela e zero por falta de
            # medida e nao por concordancia.
            "sementes": int(len(grupo)),
            "amplitude_pontos": float(inedito.max() - inedito.min()) * 100,
            "visto_mae_db": float(reference[arm]["mae_db"]),
            "inedito_mae_db": float(grupo["mae_db"].astype(float).mean()),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["estrato", "arm"]).reset_index(drop=True)
