"""Etapa 7: leave-one-arm-out, a pergunta sobre um plugin que a rede nunca ouviu.

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

import pandas as pd

from gefx.disent.train import RESULTS_ROOT

DEFAULT_ROOT = Path("datasets/disent_v2")
DEFAULT_OUTPUT = RESULTS_ROOT / "loo"


def strata(root: Path) -> Dict[str, str]:
    """Estrato de cada arm, como o render gravou no sidecar."""
    from gefx.disent.sidecar import read_dataset

    return {str(arm): str(stratum) for arm, stratum
            in read_dataset(Path(root)).groupby("arm")["stratum"].first().items()}


def _evaluate_held_out(run_dir: Path, root: Path, held_out: str, seen: Sequence[str],
                       catalog_arms: Optional[Sequence[str]] = None, batch: int = 64,
                       extra: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    from gefx.disent.probes import load_run
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.retrieval import center_by_arm, retrieve_by_arm
    from gefx.disent.sidecar import split_frames
    from gefx.disent.train import embed

    model, _ = load_run(run_dir)
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")

    frames = split_frames(root)
    # Na curva de diversidade o catalogo fica fixo enquanto o treino cresce.
    pool = list(catalog_arms) if catalog_arms is not None else list(seen)
    catalog = frames["catalog"][frames["catalog"]["arm"].isin(pool)]
    catalog = catalog.reset_index(drop=True)
    z_catalog = embed(model, FeatureStore(root, catalog, "Spec"), standardizer, batch)

    rows: List[Dict[str, object]] = []
    for label, arms in (("transferencia", [held_out]), ("vistos", list(seen))):
        queries = frames["query"][frames["query"]["arm"].isin(arms)].reset_index(drop=True)
        z_query = embed(model, FeatureStore(root, queries, "Spec"), standardizer, batch)
        # `centrado`: cada arm menos a propria media (`center_by_arm`), inclusive o
        # retirado, cuja media sai das consultas dele, sem rotulo.
        for centered in (False, True):
            zq, zc = ((center_by_arm(queries, z_query), center_by_arm(catalog, z_catalog))
                      if centered else (z_query, z_catalog))
            # Na transferencia o arm retirado ja esta fora do catalogo.
            result = retrieve_by_arm(queries, catalog, zq, zc,
                                     same_arm=(held_out not in pool and label == "transferencia"))
            overall = result.metrics["overall"]
            rows.append({
                "arm_retirado": held_out,
                "estrato": str(frames["query"].loc[frames["query"]["arm"] == held_out,
                                                   "stratum"].iloc[0]),
                "condicao": label, "centrado": centered, **(extra or {}),
                "drive_exact": float(overall["drive_level"]["exact"]),
                "within_one": float(overall["drive_level"]["within_one"]),
                "mae_db": float(overall["mae_db"]), "n": int(overall["n"]),
            })
    return rows


def leave_one_arm_out(
    root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    arms: Optional[Sequence[str]] = None,
    steps: Optional[int] = None,
    seed: int = 20260908,
    seeds: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Uma execucao por (arm retirado, semente); reaproveita o que ja esta em disco.

    A primeira semente grava em `<arm>/`, as demais em `<arm>_s<semente>/`.
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
                train(TrainConfig(dataset_root=root, arms=tuple(seen),
                                  steps=steps or TrainConfig.steps, seed=semente,
                                  output_dir=run_dir),
                      verbose=False)
            rows.extend(_evaluate_held_out(run_dir, root, held_out, seen,
                                           extra={"seed": semente}))
            if verbose:
                for row in rows[-4:]:
                    print(f"  {row['condicao']:14s} {'centrado' if row['centrado'] else '':8s} "
                          f"{row['drive_exact']*100:5.1f}%  "
                          f"{row['mae_db']:5.2f} dB", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "resumo.csv", index=False)
    return pd.DataFrame(rows)


# --- B2 e B3: quantas implementacoes o treino precisa ver? ---------------------
def arm_diversity_curve(
    held_out: str,
    root: Path = DEFAULT_ROOT,
    output_dir: Path = RESULTS_ROOT / "diversidade",
    order: Optional[Sequence[str]] = None,
    steps: Optional[int] = None,
    seed: int = 20260908,
    reuse: Optional[Path] = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """B2 e B3 na mesma curva: treinar com 1, 2, ... N-1 implementacoes.

    O catalogo fica fixo nos N-1 arms; so a pertinencia ao treino varia. O ultimo
    ponto e a execucao do leave-one-out, reaproveitada de `reuse`. Sem `order`, o
    treino acumula por estrato (S1 antes de S2...), e dentro dele por nome.
    """
    from gefx.disent.train import TrainConfig, train

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
        reuso = Path(reuse) / held_out if reuse else None
        if k == len(order) and reuso is not None and (reuso / "run.json").exists():
            run_dir = reuso
        if verbose:
            print(f"[k={k}] {', '.join(treinados)}", flush=True)
        if not (run_dir / "run.json").exists():
            # Quem pontua e o arm retirado, contra o catalogo fixo, logo abaixo.
            train(TrainConfig(dataset_root=root, arms=tuple(treinados),
                              steps=steps or TrainConfig.steps, seed=seed,
                              evaluate_at_end=False, output_dir=run_dir), verbose=False)
        novas = _evaluate_held_out(
            run_dir, root, held_out, treinados, catalog_arms=pool,
            extra={"k": k, "arms_treinados": "|".join(treinados),
                   "estratos": len({estratos[arm] for arm in treinados}),
                   "run_dir": str(run_dir)},
        )
        rows.extend(novas)
        if verbose:
            for row in novas:
                print(f"  {row['condicao']:14s} {'centrado' if row['centrado'] else '':8s} "
                      f"{row['drive_exact']*100:5.1f}%  "
                      f"{row['mae_db']:5.2f} dB", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "resumo.csv", index=False)
    return pd.DataFrame(rows)


def centered_rows(table: pd.DataFrame) -> pd.Series:
    """Linhas da busca com `center_by_arm`; resumos antigos nao tem a coluna."""
    if "centrado" not in table.columns:
        return pd.Series(False, index=table.index)
    return table["centrado"].astype(str).str.lower() == "true"


def transfer_cost(
    loo: pd.DataFrame,
    seen_metrics: Path = RESULTS_ROOT / "supcon" / "metrics.json",
) -> pd.DataFrame:
    """Custo de nunca ter visto a implementacao: cada arm retirado contra ele mesmo
    no encoder treinado com todos."""
    reference = json.loads(Path(seen_metrics).read_text(encoding="utf-8"))["per_query_arm"]
    transferencia = loo[(loo["condicao"] == "transferencia") & ~centered_rows(loo)]
    rows: List[Dict[str, object]] = []
    for arm, grupo in transferencia.groupby("arm_retirado", sort=False):
        arm = str(arm)
        if arm not in reference:
            continue
        visto = float(reference[arm]["drive_level"]["exact"])
        inedito = grupo["drive_exact"].astype(float)
        rows.append({
            "arm": arm, "estrato": str(grupo.iloc[0]["estrato"]),
            "visto": visto, "inedito": float(inedito.mean()),
            "custo_pontos": (float(inedito.mean()) - visto) * 100,
            # Com uma semente so, amplitude zero e falta de medida.
            "sementes": int(len(grupo)),
            "amplitude_pontos": float(inedito.max() - inedito.min()) * 100,
            "visto_mae_db": float(reference[arm]["mae_db"]),
            "inedito_mae_db": float(grupo["mae_db"].astype(float).mean()),
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["estrato", "arm"]).reset_index(drop=True)
