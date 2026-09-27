"""A tarefa do POC II: recuperacao em catalogo, e os baselines sem aprendizado.

Consulta e catalogo vem de splits de conteudo disjuntos, e por padrao o catalogo
exclui o arm da consulta: a resposta tem de atravessar implementacoes. A
diagonal (mesmo arm) fica como controle. A busca e por cosseno, a mesma para o
B0 e para o codigo aprendido.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.disent.sidecar import EFFECT_FOLDER, read_dataset, split_frames

LEVEL_AXES: Tuple[str, ...] = ("drive_level",)


def nearest(queries: np.ndarray, catalog: np.ndarray,
            chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Indice e distancia de cosseno do item de catalogo mais proximo de cada consulta."""
    queries = np.asarray(queries, dtype=np.float32)
    catalog = np.asarray(catalog, dtype=np.float32)
    if queries.shape[1] != catalog.shape[1]:
        raise ValueError(
            f"dimensoes incompativeis: {queries.shape[1]} e {catalog.shape[1]}"
        )
    catalog = catalog / (np.linalg.norm(catalog, axis=1, keepdims=True) + 1e-12)
    picks = np.empty(len(queries), dtype=np.int64)
    dists = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        block = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-12)
        similarity = block @ catalog.T
        picks[start : start + chunk] = similarity.argmax(axis=1)
        dists[start : start + chunk] = 1.0 - similarity.max(axis=1)
    return picks, dists


# --- a tarefa -----------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalResult:
    """Predicoes linha a linha mais o resumo agregado."""

    predictions: pd.DataFrame
    metrics: Dict[str, object]


def retrieve(
    query_frame: pd.DataFrame,
    catalog_frame: pd.DataFrame,
    query_vectors: np.ndarray,
    catalog_vectors: np.ndarray,
) -> pd.DataFrame:
    """Uma linha por consulta, com o que foi recuperado e o que era certo."""
    # Por gravacao, nao so por trecho: dois trechos da mesma execucao tambem vazam.
    keys = [key for key in ("content_id", "source_audio_id")
            if key in query_frame.columns and key in catalog_frame.columns]
    if any(set(query_frame[key]) & set(catalog_frame[key]) for key in keys):
        raise ValueError(
            "consulta e catalogo compartilham conteudo: o acerto poderia vir de "
            "reconhecer a execucao, e nao o ajuste"
        )
    picks, dists = nearest(query_vectors, catalog_vectors)
    hit = catalog_frame.iloc[picks]
    out = pd.DataFrame(
        {
            "file_name": query_frame["file_name"].to_numpy(),
            "query_arm": query_frame["arm"].to_numpy(),
            "query_content": query_frame["content_id"].to_numpy(),
            "distance": dists,
            "retrieved_file": hit["file_name"].to_numpy(),
            "retrieved_arm": hit["arm"].to_numpy(),
        }
    )
    for axis in LEVEL_AXES:
        out[f"true_{axis}"] = query_frame[axis].to_numpy()
        out[f"pred_{axis}"] = hit[axis].to_numpy()
    out["true_drive_db"] = query_frame["drive_db_equivalente"].to_numpy()
    out["pred_drive_db"] = hit["drive_db_equivalente"].to_numpy()
    return out


def score(predictions: pd.DataFrame, alphabet: Mapping[str, int]) -> Dict[str, object]:
    """Acerto exato, acerto a um nivel e erro medio por eixo, com o acaso do eixo."""
    out: Dict[str, object] = {"n": int(len(predictions))}
    for axis, size in alphabet.items():
        true = predictions[f"true_{axis}"].to_numpy()
        pred = predictions[f"pred_{axis}"].to_numpy()
        out[axis] = {
            "exact": float(np.mean(true == pred)),
            "chance": 1.0 / size,
            "mae_levels": float(np.mean(np.abs(true - pred))),
            "within_one": float(np.mean(np.abs(true - pred) <= 1)),
        }
    out["mae_db"] = float(
        np.mean(np.abs(predictions["true_drive_db"] - predictions["pred_drive_db"]))
    )
    out["cross_implementation"] = bool(
        not (predictions["query_arm"] == predictions["retrieved_arm"]).any()
    )
    return out


def alphabet(frame: pd.DataFrame) -> Dict[str, int]:
    """Quantos niveis cada eixo tem neste recorte; o acaso sai daqui, nao da grade cheia."""
    return {axis: int(frame[axis].nunique()) for axis in LEVEL_AXES}


def _context(frame: pd.DataFrame) -> Dict[str, object]:
    """O que as figuras precisam saber do dataset sem reler o roster: a escada de
    dB da referencia e o estrato de cada arm."""
    return {
        "drive_db_ladder": sorted(
            float(v) for v in frame.groupby("drive_level")["drive_db_equivalente"].first()),
        "strata": {str(arm): str(stratum) for arm, stratum
                   in frame.groupby("arm")["stratum"].first().items()},
    }


def retrieve_by_arm(
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    query_vectors: np.ndarray,
    catalog_vectors: np.ndarray,
    same_arm: bool = False,
) -> RetrievalResult:
    """A tarefa do POC II sobre uma representacao qualquer das duas particoes.

    E o protocolo unico que avalia o B0 e o codigo aprendido. `same_arm=False` (o
    padrao) exclui o arm da consulta do catalogo.
    """
    if len(queries) != len(query_vectors) or len(catalog) != len(catalog_vectors):
        raise ValueError(
            f"tabelas e vetores desalinhados: {len(queries)}/{len(query_vectors)} "
            f"consultas, {len(catalog)}/{len(catalog_vectors)} catalogo"
        )
    parts: List[pd.DataFrame] = []
    for arm in sorted(queries["arm"].unique()):
        q_where = np.flatnonzero((queries["arm"] == arm).to_numpy())
        keep = (
            np.ones(len(catalog), dtype=bool)
            if same_arm
            else (catalog["arm"] != arm).to_numpy()
        )
        c_where = np.flatnonzero(keep)
        parts.append(
            retrieve(
                queries.iloc[q_where].reset_index(drop=True),
                catalog.iloc[c_where].reset_index(drop=True),
                query_vectors[q_where],
                catalog_vectors[c_where],
            )
        )
    predictions = pd.concat(parts, ignore_index=True)

    sizes = alphabet(queries)
    por_arm = int(len(catalog) / catalog["arm"].nunique())
    metrics: Dict[str, object] = {
        "overall": score(predictions, sizes),
        "per_query_arm": {
            str(arm): score(part, sizes)
            for arm, part in predictions.groupby("query_arm", sort=True)
        },
        "same_arm": bool(same_arm),
        "alphabet": sizes,
        # Quantos itens cada consulta de fato alcanca.
        "catalog_size": len(catalog) if same_arm else len(catalog) - por_arm,
        **_context(queries),
    }
    return RetrievalResult(predictions=predictions, metrics=metrics)


# --- baseline B0 --------------------------------------------------------------
def standardized_rows(store, standardizer, batch: int = 256) -> np.ndarray:
    """Cada linha do `store` padronizada e achatada: `(n, frequencia * tempo)`."""
    out = np.empty((len(store), int(np.prod(store.feature_shape))), dtype=np.float32)
    for block, features in store.stream(np.arange(len(store)), chunk=batch):
        out[block] = standardizer.transform(features).reshape(len(block), -1)
    return out


def baseline_b0(
    root: Path,
    arms: Optional[Sequence[str]] = None,
    feature: str = "Spec",
) -> RetrievalResult:
    """B0: vizinho mais proximo na propria entrada do encoder, sem aprender nada.

    Mesma feature e mesma padronizacao por pixel (ajustada no treino) que o encoder
    recebe: a diferenca entre os dois e so o aprendizado.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer

    frames = split_frames(Path(root), arms)
    stores = {name: FeatureStore(Path(root), frame, feature)
              for name, frame in frames.items()}
    standardizer = PixelStandardizer.fit(stores["train"])
    return retrieve_by_arm(
        frames["query"], frames["catalog"],
        standardized_rows(stores["query"], standardizer),
        standardized_rows(stores["catalog"], standardizer),
    )


# --- baseline B1 --------------------------------------------------------------
POC1_MODEL_DIR = Path("results/second_main_run/Spec/distortion")


def _nearest_level(values: np.ndarray, ladder: np.ndarray) -> np.ndarray:
    """Nivel da escada mais proximo de cada valor continuo."""
    return np.abs(np.asarray(values)[:, None] - np.asarray(ladder)[None, :]).argmin(axis=1)


def baseline_b1(
    root: Path,
    model_dir: Path = POC1_MODEL_DIR,
    split: str = "query",
    arms: Optional[Sequence[str]] = None,
    feature_name: str = "Spec",
    batch: int = 256,
) -> RetrievalResult:
    """B1: o regressor do POC I, sem retreino, aplicado aos arms do POC II.

    A saida ja esta em `drive_db`, a unidade de `drive_db_equivalente`, entao o erro
    sai em dB sem conversao. Em `pedalboard-tanh`, a mesma implementacao do
    treino, so muda a faixa de niveis.
    """
    from gefx.data.features import extract_feature, stack_features
    from gefx.effects.catalog import EFFECT_PARAMETER_RANGES
    from gefx.training.inference import load_trained_chain

    spec = EFFECT_PARAMETER_RANGES["distortion"][0]
    lo, hi = float(spec["min"]), float(spec["max"])

    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    queries = data[data["split"] == split].reset_index(drop=True)
    if queries.empty:
        raise ValueError(f"nenhuma linha no split {split!r}")

    chain = load_trained_chain(model_dir)
    predicted = np.empty(len(queries), dtype=float)
    for start in range(0, len(queries), batch):
        part = queries.iloc[start : start + batch]
        stacked = stack_features(
            [
                extract_feature(Path(root) / arm / EFFECT_FOLDER / name, feature_name)
                for arm, name in zip(part["arm"], part["file_name"])
            ],
            feature_name,
        )
        predicted[start : start + batch] = chain.predict(stacked)[:, 0]

    drive_db = lo + predicted * (hi - lo)
    ladder = np.array(
        sorted(queries.groupby("drive_level")["drive_db_equivalente"].first())
    )
    out = pd.DataFrame(
        {
            "file_name": queries["file_name"].to_numpy(),
            "query_arm": queries["arm"].to_numpy(),
            "query_content": queries["content_id"].to_numpy(),
            "retrieved_arm": "(regressor POC I)",
            "retrieved_file": "",
            "distance": np.nan,
            "true_drive_level": queries["drive_level"].to_numpy(),
            "pred_drive_level": _nearest_level(drive_db, ladder),
            "true_drive_db": queries["drive_db_equivalente"].to_numpy(),
            "pred_drive_db": drive_db,
        }
    )
    sizes = alphabet(queries)
    metrics: Dict[str, object] = {
        "overall": score(out, sizes),
        "per_query_arm": {
            str(arm): score(part, sizes)
            for arm, part in out.groupby("query_arm", sort=True)
        },
        "model_dir": str(model_dir),
        "feature": feature_name,
        "alphabet": sizes,
        **_context(queries),
    }
    return RetrievalResult(predictions=out, metrics=metrics)
