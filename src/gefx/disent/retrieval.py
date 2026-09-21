"""Recuperacao em catalogo sem aprendizado: o baseline B0 e o protocolo da tarefa.

Consulta e catalogo vem de splits de conteudo disjuntos, e por padrao o catalogo
exclui o arm da consulta: a resposta tem de atravessar implementacoes. A
diagonal (mesmo arm) fica como controle.

O descritor e o do oraculo (L1 sobre log-mel multirresolucao) com o tempo
agrupado em `MEL_TIME_POOL` faixas, para caber em 5.600 x 5.600 comparacoes.
`reduction_fidelity` mede o preco dessa reducao.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.audio import load_audio_file
from gefx.disent.oracle import FFT_SIZES, N_MELS, log_mel_stack
from gefx.disent.sidecar import EFFECT_FOLDER, read_sidecar

# Faixas de tempo do descritor, escolhidas por `fidelity_sweep` (Spearman 0,97
# contra a distancia integral).
MEL_TIME_POOL = 16

# Orcamento de memoria de um bloco da matriz (consultas x catalogo x dimensoes).
NEAREST_BLOCK_BYTES = 512 * 1024 * 1024

DESCRIPTOR_FILENAME = "retrieval_descriptor.npz"

LEVEL_AXES: Tuple[str, ...] = ("drive_level", "tone_level")


# --- descritor ----------------------------------------------------------------
def pool_time(frame: np.ndarray, bands: int = MEL_TIME_POOL) -> np.ndarray:
    """Media do log-mel dentro de cada faixa de tempo. `(n_mels, bands)`."""
    if bands < 1:
        raise ValueError("bands deve ser >= 1")
    if frame.shape[1] < bands:
        raise ValueError(
            f"{frame.shape[1]} quadros nao dao para {bands} faixas de tempo"
        )
    return np.stack(
        [chunk.mean(axis=1) for chunk in np.array_split(frame, bands, axis=1)], axis=1
    )


def descriptor(
    audio: np.ndarray,
    sr: int,
    fft_sizes: Sequence[int] = FFT_SIZES,
    n_mels: int = N_MELS,
    bands: int = MEL_TIME_POOL,
) -> np.ndarray:
    """Vetor achatado `(len(fft_sizes) * n_mels * bands,)`.

    Toda resolucao contribui com o mesmo numero de celulas, entao a media simples
    reproduz a ponderacao do oraculo.
    """
    stack = log_mel_stack(audio, sr, fft_sizes=fft_sizes, n_mels=n_mels)
    return np.concatenate([pool_time(part, bands).ravel() for part in stack]).astype(
        np.float32
    )


def _descriptor_of_folder(
    folder: Path, names: Sequence[str], bands: int
) -> np.ndarray:
    out = np.empty((len(names), len(FFT_SIZES) * N_MELS * bands), dtype=np.float32)
    for row, name in enumerate(names):
        audio, sr = load_audio_file(Path(folder) / name)
        out[row] = descriptor(audio, sr, bands=bands)
    return out


def ensure_descriptors(
    root: Path,
    arm: str,
    bands: int = MEL_TIME_POOL,
    rebuild: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """`(descritores, nomes)` daquele arm, calculados e cacheados na primeira vez.

    O cache guarda `bands`: mudar o agrupamento invalida o cache.
    """
    root = Path(root)
    path = root / arm / DESCRIPTOR_FILENAME
    frame = read_sidecar(root / arm)
    names = [str(name) for name in frame["file_name"]]

    if path.exists() and not rebuild:
        stored = np.load(path, allow_pickle=False)
        if int(stored["bands"]) == bands and list(stored["file_names"]) == names:
            return stored["descriptor"], names

    matrix = _descriptor_of_folder(root / arm / EFFECT_FOLDER, names, bands)
    np.savez(path, descriptor=matrix, file_names=np.array(names), bands=np.int64(bands))
    return matrix, names


def load_descriptors(
    root: Path, frame: pd.DataFrame, bands: int = MEL_TIME_POOL, rebuild: bool = False
) -> np.ndarray:
    """Descritores na ordem das linhas de `frame`, casados por nome de arquivo."""
    frame = frame.reset_index(drop=True)
    out = np.empty((len(frame), len(FFT_SIZES) * N_MELS * bands), dtype=np.float32)
    for arm in frame["arm"].unique():
        where = np.flatnonzero((frame["arm"] == arm).to_numpy())
        matrix, names = ensure_descriptors(root, str(arm), bands=bands, rebuild=rebuild)
        position = {name: index for index, name in enumerate(names)}
        wanted = [str(name) for name in frame["file_name"].to_numpy()[where]]
        missing = [name for name in wanted if name not in position]
        if missing:
            raise ValueError(
                f"{arm}: {len(missing)} linhas sem descritor (ex.: {missing[:3]})"
            )
        out[where] = matrix[np.array([position[name] for name in wanted], dtype=np.int64)]
    return out


# --- vizinho mais proximo -----------------------------------------------------
def block_size(n_catalog: int, dims: int, budget: int = NEAREST_BLOCK_BYTES) -> int:
    """Quantas consultas cabem num bloco de diferencas dentro do orcamento."""
    per_query = max(1, n_catalog * dims * 4)
    return max(1, budget // per_query)


#: `l1` para o descritor cru, `cosine` para o codigo aprendido.
METRICS: Tuple[str, ...] = ("l1", "cosine")


def nearest(
    queries: np.ndarray,
    catalog: np.ndarray,
    chunk: Optional[int] = None,
    metric: str = "l1",
) -> Tuple[np.ndarray, np.ndarray]:
    """Indice e distancia do item de catalogo mais proximo de cada consulta.

    Em L1 a matriz de diferencas nao cabe inteira; `chunk=None` dimensiona o bloco
    por `NEAREST_BLOCK_BYTES`.
    """
    if metric not in METRICS:
        raise ValueError(f"metrica desconhecida: {metric!r}. Ha {list(METRICS)}")
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    catalog = np.ascontiguousarray(catalog, dtype=np.float32)
    if queries.shape[1] != catalog.shape[1]:
        raise ValueError(
            f"dimensoes incompativeis: {queries.shape[1]} e {catalog.shape[1]}"
        )

    if metric == "cosine":
        # Normaliza aqui tambem: nem todo codigo chega com norma unitaria.
        q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        c = catalog / (np.linalg.norm(catalog, axis=1, keepdims=True) + 1e-12)
        similarity = q @ c.T
        picks = similarity.argmax(axis=1)
        return picks, (1.0 - similarity.max(axis=1)).astype(np.float32)

    if chunk is None:
        chunk = block_size(len(catalog), queries.shape[1])
    picks = np.empty(len(queries), dtype=np.int64)
    dists = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        d = np.abs(block[:, None, :] - catalog[None, :, :]).mean(axis=2)
        picks[start : start + chunk] = d.argmin(axis=1)
        dists[start : start + chunk] = d.min(axis=1)
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
    query_descriptors: np.ndarray,
    catalog_descriptors: np.ndarray,
    metric: str = "l1",
) -> pd.DataFrame:
    """Uma linha por consulta, com o que foi recuperado e o que era certo."""
    if set(query_frame["content_id"]) & set(catalog_frame["content_id"]):
        raise ValueError(
            "consulta e catalogo compartilham conteudo: o acerto poderia vir de "
            "reconhecer a execucao, e nao o ajuste"
        )
    picks, dists = nearest(query_descriptors, catalog_descriptors, metric=metric)
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
    both = np.ones(len(predictions), dtype=bool)
    for axis in alphabet:
        both &= predictions[f"true_{axis}"].to_numpy() == predictions[f"pred_{axis}"].to_numpy()
    out["config_exact"] = float(np.mean(both))
    out["config_chance"] = float(np.prod([1.0 / size for size in alphabet.values()]))
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


def retrieve_by_arm(
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    query_vectors: np.ndarray,
    catalog_vectors: np.ndarray,
    same_arm: bool = False,
    metric: str = "l1",
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
                metric=metric,
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
        "metric": metric,
        "alphabet": sizes,
        # Quantos itens cada consulta de fato alcanca: o denominador de `hubness`.
        "catalog_size": len(catalog) if same_arm else len(catalog) - por_arm,
    }
    return RetrievalResult(predictions=predictions, metrics=metrics)


# --- fidelidade da reducao ----------------------------------------------------
def reduction_fidelity(
    root: Path,
    frame: pd.DataFrame,
    n_pairs: int = 400,
    bands: int = MEL_TIME_POOL,
    seed: int = 0,
) -> Dict[str, float]:
    """Correlacao entre a distancia do descritor agrupado e a integral, em `n_pairs` pares.

    A de postos e a que importa: a busca so depende da ordem.
    """
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    left = rng.integers(0, len(frame), size=n_pairs)
    right = rng.integers(0, len(frame), size=n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]

    reduced = load_descriptors(root, frame, bands=bands)
    approx = np.abs(reduced[left] - reduced[right]).mean(axis=1)

    from gefx.disent.oracle import spectral_distance

    cache: Dict[int, tuple] = {}

    def stack_of(row: int):
        if row not in cache:
            arm = str(frame["arm"].iloc[row])
            name = str(frame["file_name"].iloc[row])
            audio, sr = load_audio_file(Path(root) / arm / EFFECT_FOLDER / name)
            cache[row] = log_mel_stack(audio, sr)
        return cache[row]

    full = np.array(
        [spectral_distance(stack_of(a), stack_of(b)) for a, b in zip(left, right)]
    )
    return {
        "n_pairs": int(len(full)),
        "pearson": float(pearsonr(full, approx)[0]),
        "spearman": float(spearmanr(full, approx)[0]),
        "mean_ratio": float(np.mean(approx / full)),
    }


# --- baseline B0 --------------------------------------------------------------
def baseline_b0(
    root: Path,
    query_split: str = "query",
    catalog_split: str = "catalog",
    arms: Optional[Sequence[str]] = None,
    bands: int = MEL_TIME_POOL,
    same_arm: bool = False,
) -> RetrievalResult:
    """B0: vizinho mais proximo em distancia espectral crua, sem aprender nada.

    `same_arm=True` e o controle sem travessia de implementacao.
    """
    from gefx.disent.sidecar import read_dataset

    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    queries = data[data["split"] == query_split].reset_index(drop=True)
    catalog = data[data["split"] == catalog_split].reset_index(drop=True)
    if queries.empty or catalog.empty:
        raise ValueError(f"recorte vazio: {len(queries)} consultas, {len(catalog)} catalogo")

    q_desc = load_descriptors(root, queries, bands=bands)
    c_desc = load_descriptors(root, catalog, bands=bands)

    return retrieve_by_arm(queries, catalog, q_desc, c_desc, same_arm=same_arm)


def cross_arm_table(
    predictions: pd.DataFrame, axis: str = "drive_level"
) -> pd.DataFrame:
    """Acerto exato por (arm da consulta, arm recuperado)."""
    hit = predictions[f"true_{axis}"] == predictions[f"pred_{axis}"]
    return (
        predictions.assign(hit=hit)
        .pivot_table(index="query_arm", columns="retrieved_arm", values="hit",
                     aggfunc=["mean", "size"])
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
    sai em dB sem conversao. Mistura duas mudancas de dominio (implementacao e o
    estagio de tone novo); em `pedalboard-tanh` so a segunda age.
    """
    from gefx.data.features import extract_feature, stack_features
    from gefx.effects.catalog import EFFECT_PARAMETER_RANGES
    from gefx.disent.sidecar import read_dataset
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
            "true_tone_level": queries["tone_level"].to_numpy(),
            "pred_tone_level": -1,  # o POC I nao tinha estagio de tone
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
        "note": "tone nao e predito pelo regressor do POC I",
    }
    return RetrievalResult(predictions=out, metrics=metrics)


# --- diagnostico: sorvedouros -------------------------------------------------
def hubness(predictions: pd.DataFrame, catalog_size: int) -> Dict[str, object]:
    """Quantas consultas cada item de catalogo atrai (Radovanovic et al. 2010).

    Assimetria alta indica sorvedouros: itens que respondem por quase tudo.
    """
    counts = predictions["retrieved_file"].value_counts()
    full = np.zeros(catalog_size, dtype=float)
    full[: len(counts)] = counts.to_numpy()
    esperado = len(predictions) / catalog_size
    return {
        "expected_per_item": float(esperado),
        "max_occurrence": int(counts.max()),
        "skewness": float(
            np.mean((full - full.mean()) ** 3) / (full.std() ** 3 + 1e-12)
        ),
        "share_of_top_1pct": float(
            counts.head(max(1, catalog_size // 100)).sum() / len(predictions)
        ),
        "by_retrieved_arm": (
            predictions["retrieved_arm"].value_counts(normalize=True).round(4).to_dict()
        ),
    }


def pairwise_b0(
    root: Path,
    query_split: str = "query",
    catalog_split: str = "catalog",
    arms: Optional[Sequence[str]] = None,
    bands: int = MEL_TIME_POOL,
) -> pd.DataFrame:
    """Acerto de drive por par (arm da consulta, arm do catalogo), com catalogo de um arm so.

    Separa a transferencia entre o par da competicao entre arms num catalogo comum.
    """
    from gefx.disent.sidecar import read_dataset

    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    queries = data[data["split"] == query_split].reset_index(drop=True)
    catalog = data[data["split"] == catalog_split].reset_index(drop=True)
    q_desc = load_descriptors(root, queries, bands=bands)
    c_desc = load_descriptors(root, catalog, bands=bands)

    rows: List[Dict[str, object]] = []
    for q_arm in sorted(queries["arm"].unique()):
        q_where = np.flatnonzero((queries["arm"] == q_arm).to_numpy())
        q_part = queries.iloc[q_where].reset_index(drop=True)
        for c_arm in sorted(catalog["arm"].unique()):
            c_where = np.flatnonzero((catalog["arm"] == c_arm).to_numpy())
            out = retrieve(
                q_part, catalog.iloc[c_where].reset_index(drop=True),
                q_desc[q_where], c_desc[c_where],
            )
            metrics = score(out, alphabet(q_part))
            rows.append(
                {
                    "query_arm": q_arm,
                    "catalog_arm": c_arm,
                    "drive_exact": metrics["drive_level"]["exact"],
                    "drive_mae_levels": metrics["drive_level"]["mae_levels"],
                    "drive_mae_db": metrics["mae_db"],
                    "tone_exact": metrics["tone_level"]["exact"],
                    "n": metrics["n"],
                }
            )
    return pd.DataFrame(rows)


def fidelity_sweep(
    root: Path,
    frame: pd.DataFrame,
    bands_list: Sequence[int] = (1, 4, 8, 16, 32, 64),
    n_items: int = 120,
    n_pairs: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """`reduction_fidelity` para varios agrupamentos, com a pilha integral calculada uma vez."""
    from scipy.stats import pearsonr, spearmanr

    from gefx.disent.oracle import spectral_distance

    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    sample = rng.choice(len(frame), size=min(n_items, len(frame)), replace=False)

    stacks = {}
    for row in sample:
        arm = str(frame["arm"].iloc[row])
        name = str(frame["file_name"].iloc[row])
        audio, sr = load_audio_file(Path(root) / arm / EFFECT_FOLDER / name)
        stacks[row] = log_mel_stack(audio, sr)

    left = rng.choice(sample, n_pairs)
    right = rng.choice(sample, n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]
    full = np.array(
        [spectral_distance(stacks[a], stacks[b]) for a, b in zip(left, right)]
    )

    rows: List[Dict[str, object]] = []
    for bands in bands_list:
        pooled = {
            row: np.concatenate([pool_time(part, bands).ravel() for part in stack])
            for row, stack in stacks.items()
        }
        approx = np.array(
            [np.abs(pooled[a] - pooled[b]).mean() for a, b in zip(left, right)]
        )
        rows.append(
            {
                "bands": bands,
                "dims": len(FFT_SIZES) * N_MELS * bands,
                "pearson": float(pearsonr(full, approx)[0]),
                "spearman": float(spearmanr(full, approx)[0]),
                "mean_ratio": float(np.mean(approx / full)),
                "n_pairs": int(len(full)),
            }
        )
    return pd.DataFrame(rows)


def paired_content_b0(
    root: Path,
    split: str = "query",
    arms: Optional[Sequence[str]] = None,
    bands: int = MEL_TIME_POOL,
    same_arm: bool = False,
) -> RetrievalResult:
    """B0 com candidatos do mesmo conteudo em outra implementacao: um teto, nao a tarefa.

    A diferenca para o `baseline_b0` e o que o conteudo custa.
    """
    from gefx.disent.sidecar import read_dataset

    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    frame = data[data["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"nenhuma linha no split {split!r}")
    descriptors = load_descriptors(root, frame, bands=bands)

    parts: List[pd.DataFrame] = []
    for content in sorted(frame["content_id"].unique()):
        do_conteudo = np.flatnonzero((frame["content_id"] == content).to_numpy())
        bloco = frame.iloc[do_conteudo].reset_index(drop=True)
        for arm in sorted(bloco["arm"].unique()):
            q_where = np.flatnonzero((bloco["arm"] == arm).to_numpy())
            keep = (
                np.ones(len(bloco), dtype=bool)
                if same_arm
                else (bloco["arm"] != arm).to_numpy()
            )
            c_where = np.flatnonzero(keep)
            q_frame = bloco.iloc[q_where].reset_index(drop=True)
            c_frame = bloco.iloc[c_where].reset_index(drop=True)
            # A mao porque `retrieve` recusa conteudo compartilhado, que aqui e o ponto.
            picks, dists = nearest(
                descriptors[do_conteudo][q_where], descriptors[do_conteudo][c_where]
            )
            hit = c_frame.iloc[picks]
            saida = pd.DataFrame(
                {
                    "file_name": q_frame["file_name"].to_numpy(),
                    "query_arm": q_frame["arm"].to_numpy(),
                    "query_content": q_frame["content_id"].to_numpy(),
                    "distance": dists,
                    "retrieved_file": hit["file_name"].to_numpy(),
                    "retrieved_arm": hit["arm"].to_numpy(),
                    "true_drive_db": q_frame["drive_db_equivalente"].to_numpy(),
                    "pred_drive_db": hit["drive_db_equivalente"].to_numpy(),
                }
            )
            for axis in LEVEL_AXES:
                saida[f"true_{axis}"] = q_frame[axis].to_numpy()
                saida[f"pred_{axis}"] = hit[axis].to_numpy()
            parts.append(saida)

    predictions = pd.concat(parts, ignore_index=True)
    sizes = alphabet(frame)
    metrics: Dict[str, object] = {
        "overall": score(predictions, sizes),
        "per_query_arm": {
            str(arm): score(part, sizes)
            for arm, part in predictions.groupby("query_arm", sort=True)
        },
        "same_arm": bool(same_arm),
        "alphabet": sizes,
        "catalog_size": int(len(frame) / frame["content_id"].nunique()),
        "note": "teto: candidatos do mesmo conteudo, indisponivel no uso real",
    }
    return RetrievalResult(predictions=predictions, metrics=metrics)
