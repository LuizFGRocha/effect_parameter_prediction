"""Leave-one-arm-out e curva de diversidade: um plugin que a rede nunca ouviu.

Cada execucao treina em N-1 arms e responde com o mesmo modelo:

- `transferencia`: consulta do arm retirado contra o catalogo dos vistos;
- `vistos`: consulta dos arms vistos contra o mesmo catalogo (controle interno).

A comparacao certa para o custo de transferencia e o proprio arm no encoder
treinado com todos (`transfer_cost`), nao a linha `vistos`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from gefx.data.dataset import FeatureStore
from gefx.data.grid import DATASET_ROOT, EVAL_SPLITS, arm_dirs, read_dataset, split_file
from gefx.evaluation.retrieval import center_by_arm, embed, metrics_file, retrieve_by_arm
from gefx.training.encoder import RESULTS_ROOT, EncoderTrainConfig, run_name

DEFAULT_OUTPUT = RESULTS_ROOT / "loo"


def summary_file(split: str) -> str:
    """O resumo de cada particao; as execucoes sao as mesmas, so a busca muda."""
    return split_file("resumo", split, "csv")


def cost_file(split: str) -> str:
    return split_file("custo_de_transferencia", split, "csv")


def strata(root: Path) -> Dict[str, str]:
    """Estrato de cada arm, como o render gravou no sidecar."""
    return {str(arm): str(stratum) for arm, stratum
            in read_dataset(Path(root)).groupby("arm")["stratum"].first().items()}


def _evaluate_held_out(run_dir: Path, root: Path, held_out: str, seen: Sequence[str],
                       catalog_arms: Optional[Sequence[str]] = None, batch: int = 64,
                       extra: Optional[Dict[str, object]] = None,
                       split: str = "validacao") -> List[Dict[str, object]]:
    from gefx.training.inference import open_run

    # Todos os arms: o retirado tambem e consultado.
    model, _, standardizer, frames = open_run(run_dir, root, all_arms=True)
    query_key, catalog_key = EVAL_SPLITS[split]
    # Na curva de diversidade o catalogo fica fixo enquanto o treino cresce.
    pool = list(catalog_arms) if catalog_arms is not None else list(seen)
    catalog = frames[catalog_key][frames[catalog_key]["arm"].isin(pool)]
    catalog = catalog.reset_index(drop=True)
    z_catalog = embed(model, FeatureStore(root, catalog), standardizer, batch)

    rows: List[Dict[str, object]] = []
    for label, arms in (("transferencia", [held_out]), ("vistos", list(seen))):
        queries = frames[query_key][frames[query_key]["arm"].isin(arms)].reset_index(drop=True)
        z_query = embed(model, FeatureStore(root, queries), standardizer, batch)
        # `centrado`: cada arm menos a propria media (`center_by_arm`), inclusive o
        # retirado, cuja media sai das consultas dele, sem rotulo.
        for centered in (False, True):
            zq, zc = ((center_by_arm(queries, z_query), center_by_arm(catalog, z_catalog))
                      if centered else (z_query, z_catalog))
            # Na transferencia o arm retirado ja esta fora do catalogo.
            result = retrieve_by_arm(queries, catalog, zq, zc,
                                     catalog_arms=("todos" if label == "transferencia"
                                                   else "outros"))
            overall = result.metrics["overall"]
            rows.append({
                "arm_retirado": held_out,
                "estrato": str(frames[query_key].loc[frames[query_key]["arm"] == held_out,
                                                     "stratum"].iloc[0]),
                "condicao": label, "centrado": centered, **(extra or {}),
                "drive_exact": float(overall["drive_level"]["exact"]),
                "within_one": float(overall["drive_level"]["within_one"]),
                "mae_db": float(overall["mae_db"]), "n": int(overall["n"]),
            })
    return rows


def _print_rows(rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        print(f"  {row['condicao']:14s} {'centrado' if row['centrado'] else '':8s} "
              f"{row['drive_exact']*100:5.1f}%  "
              f"{row['mae_db']:5.2f} dB", flush=True)


def leave_one_arm_out(
    root: Path = DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    arms: Optional[Sequence[str]] = None,
    steps: Optional[int] = None,
    seeds: Sequence[int] = (EncoderTrainConfig.seed,),
    verbose: bool = True,
    split: str = "validacao",
) -> pd.DataFrame:
    """Uma execucao por (arm retirado, semente); reaproveita o que ja esta em disco.

    A primeira semente grava em `<arm>/`, as demais em `<arm>_s<semente>/`.
    """
    from gefx.training.encoder import train

    root, output_dir = Path(root), Path(output_dir)
    todos = list(arms) if arms else [path.name for path in arm_dirs(root)]
    if len(todos) < 3:
        raise ValueError(f"leave-one-out precisa de ao menos 3 arms, ha {len(todos)}")

    rows: List[Dict[str, object]] = []
    for held_out in todos:
        seen = [arm for arm in todos if arm != held_out]
        for indice, semente in enumerate(seeds):
            run_dir = output_dir / (held_out if indice == 0 else f"{held_out}_s{semente}")
            if verbose:
                print(f"[{held_out} fora, semente {semente}]", flush=True)
            if not (run_dir / "run.json").exists():
                # Quem pontua e `_evaluate_held_out`, logo abaixo.
                train(EncoderTrainConfig(dataset_root=root, arms=tuple(seen),
                                         steps=steps or EncoderTrainConfig.steps, seed=semente,
                                         evaluate_at_end=False, output_dir=run_dir),
                      verbose=False)
            novas = _evaluate_held_out(run_dir, root, held_out, seen,
                                       extra={"seed": semente}, split=split)
            rows.extend(novas)
            if verbose:
                _print_rows(novas)
            pd.DataFrame(rows).to_csv(output_dir / summary_file(split), index=False)
    return pd.DataFrame(rows)


# --- B2 e B3: quantas implementacoes o treino precisa ver? ---------------------
def _loo_run(loo_dir: Path, held_out: str, seed: int) -> Optional[Path]:
    """A execucao do leave-one-out com o mesmo arm retirado e a mesma semente.

    A primeira semente do leave-one-out grava em `<arm>/`, as demais em
    `<arm>_s<semente>/`; a semente vem do `run.json`, para que uma curva de outra
    semente nao termine na execucao da primeira.
    """
    for run_dir in (loo_dir / held_out, loo_dir / f"{held_out}_s{seed}"):
        manifesto = run_dir / "run.json"
        if manifesto.exists():
            config = json.loads(manifesto.read_text(encoding="utf-8")).get("config", {})
            if config.get("seed") == seed:
                return run_dir
    return None


def arm_diversity_curve(
    held_out: str,
    root: Path = DATASET_ROOT,
    output_dir: Path = RESULTS_ROOT / "diversidade",
    order: Optional[Sequence[str]] = None,
    steps: Optional[int] = None,
    seed: int = EncoderTrainConfig.seed,
    reuse: Optional[Path] = DEFAULT_OUTPUT,
    verbose: bool = True,
    split: str = "validacao",
) -> pd.DataFrame:
    """B2 e B3 na mesma curva: treinar com 1, 2, ... N-1 implementacoes.

    O catalogo fica fixo nos N-1 arms; so a pertinencia ao treino varia. O ultimo
    ponto e a execucao do leave-one-out, reaproveitada de `reuse`. Sem `order`, o
    treino acumula por estrato (S1 antes de S2...), e dentro dele por nome.
    """
    from gefx.training.encoder import train

    root, output_dir = Path(root), Path(output_dir)
    estratos = strata(root)
    if order is None:
        order = sorted(estratos, key=lambda arm: (estratos[arm], arm))
    order = [arm for arm in order if arm != held_out]
    pool = list(order)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for k in range(1, len(order) + 1):
        treinados = order[:k]
        run_dir = output_dir / f"k{k}"
        reuso = _loo_run(Path(reuse), held_out, seed) if reuse else None
        if k == len(order) and reuso is not None:
            run_dir = reuso
        if verbose:
            print(f"[k={k}] {', '.join(treinados)}", flush=True)
        if not (run_dir / "run.json").exists():
            # Quem pontua e o arm retirado, contra o catalogo fixo, logo abaixo.
            train(EncoderTrainConfig(dataset_root=root, arms=tuple(treinados),
                                     steps=steps or EncoderTrainConfig.steps, seed=seed,
                                     evaluate_at_end=False, output_dir=run_dir), verbose=False)
        novas = _evaluate_held_out(
            run_dir, root, held_out, treinados, catalog_arms=pool,
            extra={"k": k, "arms_treinados": "|".join(treinados),
                   "estratos": len({estratos[arm] for arm in treinados}),
                   "run_dir": str(run_dir)},
            split=split,
        )
        rows.extend(novas)
        if verbose:
            _print_rows(novas)
        pd.DataFrame(rows).to_csv(output_dir / summary_file(split), index=False)
    return pd.DataFrame(rows)


def transfer_cost(
    loo: pd.DataFrame,
    encoder_dir: Path = RESULTS_ROOT,
    split: str = "validacao",
) -> pd.DataFrame:
    """Custo de nunca ter visto a implementacao: cada arm retirado contra ele mesmo
    no encoder treinado com todos, semente a semente (`<encoder_dir>/supcon`,
    `supcon_s<semente>`). As sementes do encoder diferem alguns pontos por arm, e
    a media do leave-one-out contra uma semente so poria essa diferenca no custo."""
    reference: Dict[int, Dict[str, object]] = {}

    def seen(seed: int) -> Dict[str, object]:
        if seed not in reference:
            path = Path(encoder_dir) / run_name("supcon", seed) / metrics_file(split)
            reference[seed] = json.loads(path.read_text(encoding="utf-8"))["per_query_arm"]
        return reference[seed]

    transferencia = loo[(loo["condicao"] == "transferencia") & ~loo["centrado"].astype(bool)]
    rows: List[Dict[str, object]] = []
    for arm, grupo in transferencia.groupby("arm_retirado", sort=False):
        arm = str(arm)
        seeds = [int(seed) for seed in grupo["seed"]]
        if any(arm not in seen(seed) for seed in seeds):
            continue
        visto = np.array([seen(seed)[arm]["drive_level"]["exact"] for seed in seeds], float)
        inedito = grupo["drive_exact"].to_numpy(float)
        rows.append({
            "arm": arm, "estrato": str(grupo.iloc[0]["estrato"]),
            "visto": float(visto.mean()), "inedito": float(inedito.mean()),
            "custo_pontos": float((inedito - visto).mean()) * 100,
            # Com uma semente so, amplitude zero e falta de medida.
            "sementes": int(len(grupo)),
            "amplitude_pontos": float(np.ptp(inedito - visto)) * 100,
            "visto_mae_db": float(np.mean([seen(seed)[arm]["mae_db"] for seed in seeds])),
            "inedito_mae_db": float(grupo["mae_db"].astype(float).mean()),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["estrato", "arm"]).reset_index(drop=True)
