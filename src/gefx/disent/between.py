"""Configuracoes entre os niveis: consultas nos pontos medios, catalogo na grade.

O treino e a busca so viram os 8 niveis. Aqui as consultas sao os 7 pontos medios
(`calibrate.between_levels`), renderizadas so na particao de consulta, e o
catalogo e o de sempre. A busca sempre devolve um nivel da grade, entao nao ha
acerto exato; mede-se:

- `neighbor`: o nivel recuperado e um dos dois vizinhos do ponto medio (acaso 2/8);
- `err_nn_db`: erro em dB do vizinho mais proximo (o minimo possivel e meio degrau);
- `err_knn_db`: erro em dB da media dos `k` vizinhos mais proximos. Mede a ordem
  de `z_e`: num espaco ordenado, os vizinhos de um ponto medio se dividem entre
  os dois niveis ao lado dele, e o erro cai abaixo de meio degrau.

O B1 (regressor do POC I) entra pela saida continua dele, sem busca.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from gefx.disent.sidecar import read_dataset, split_frames

DEFAULT_K = 10


def between_predictions(
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    z_query: np.ndarray,
    z_catalog: np.ndarray,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Uma linha por consulta, buscando so em outras implementacoes."""
    catalog_db = catalog["drive_db_equivalente"].to_numpy()
    catalog_level = catalog["drive_level"].to_numpy()
    z_query = z_query / np.linalg.norm(z_query, axis=1, keepdims=True)
    z_catalog = z_catalog / np.linalg.norm(z_catalog, axis=1, keepdims=True)
    parts: List[pd.DataFrame] = []
    for arm in sorted(queries["arm"].unique()):
        q = np.flatnonzero((queries["arm"] == arm).to_numpy())
        c = np.flatnonzero((catalog["arm"] != arm).to_numpy())
        similarity = z_query[q] @ z_catalog[c].T
        top = c[np.argsort(-similarity, axis=1)[:, :k]]  # o primeiro e o mais proximo
        parts.append(pd.DataFrame({
            "file_name": queries["file_name"].to_numpy()[q],
            "query_arm": arm,
            "source_audio_id": queries["source_audio_id"].to_numpy()[q],
            "between": queries["drive_level"].to_numpy()[q],
            "true_drive_db": queries["drive_db_equivalente"].to_numpy()[q],
            "pred_drive_level": catalog_level[top[:, 0]],
            "pred_drive_db": catalog_db[top[:, 0]],
            "knn_drive_db": catalog_db[top].mean(axis=1),
        }))
    return pd.concat(parts, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> Dict[str, float]:
    level, between = predictions["pred_drive_level"], predictions["between"]
    return {
        "neighbor": float(((level == between) | (level == between + 1)).mean()),
        "err_nn_db": float((predictions["pred_drive_db"] - predictions["true_drive_db"]).abs().mean()),
        "err_knn_db": float((predictions["knn_drive_db"] - predictions["true_drive_db"]).abs().mean()),
    }


def between_study(
    results_dir: Path,
    grid_root: Path,
    between_root: Path,
    runs: Sequence[str] = ("supcon",),
    k: int = DEFAULT_K,
    with_b1: bool = True,
) -> pd.DataFrame:
    """Uma linha por execucao (e o B1); as predicoes vao para `<results_dir>/entre_niveis/`."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.probes import load_run
    from gefx.disent.retrieval import baseline_b1
    from gefx.disent.train import embed

    data = read_dataset(Path(between_root))
    queries = data[data["split"] == "query"].reset_index(drop=True)
    catalog = split_frames(Path(grid_root))["catalog"]
    out_dir = Path(results_dir) / "entre_niveis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for name in runs:
        run_dir = Path(results_dir) / name
        model, manifest = load_run(run_dir)
        standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
        z_query = embed(model, FeatureStore(Path(between_root), queries), standardizer)
        z_catalog = embed(model, FeatureStore(Path(grid_root), catalog), standardizer)
        predictions = between_predictions(queries, catalog, z_query, z_catalog, k)
        predictions.to_csv(out_dir / f"{name}.csv", index=False)
        rows.append({"run": name, "technique": manifest["config"]["technique"],
                     **summarize(predictions)})

    if with_b1:
        b1 = baseline_b1(Path(between_root)).predictions
        ladder = np.array(sorted(catalog.groupby("drive_level")["drive_db_equivalente"].first()))
        continuous = b1["pred_drive_db"].to_numpy()
        level = np.abs(continuous[:, None] - ladder[None, :]).argmin(axis=1)
        # Como na busca: o nivel da grade mais proximo, e a saida continua no
        # lugar da media dos k vizinhos.
        b1 = pd.DataFrame({
            "file_name": b1["file_name"], "query_arm": b1["query_arm"],
            "source_audio_id": queries["source_audio_id"].to_numpy(),
            "between": b1["true_drive_level"], "true_drive_db": b1["true_drive_db"],
            "pred_drive_level": level, "pred_drive_db": ladder[level],
            "knn_drive_db": continuous,
        })
        b1.to_csv(out_dir / "b1.csv", index=False)
        rows.append({"run": "B1", "technique": "baseline", **summarize(b1)})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "resumo.csv", index=False)
    return table
