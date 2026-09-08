"""Recuperacao em catalogo: a tarefa do POC II sem aprendizado nenhum.

A pergunta operacional do trabalho e de **recuperacao**, nao de regressao: dado
um audio de implementacao desconhecida, achar no catalogo o ajuste equivalente.
Este modulo executa essa tarefa diretamente sobre distancia espectral, sem
treinar nada, e e por isso o baseline **B0** -- o piso contra o qual qualquer
desemaranhamento aprendido precisa se justificar.

O recorte que faz a pergunta ser sobre implementacao, e nao sobre gravacao:

- consulta e catalogo vem de **splits de conteudo disjuntos**, entao acertar
  reconhecendo o que foi tocado esta fora de questao;
- por padrao o catalogo **exclui o arm da consulta**, entao acertar reconhecendo
  o timbre da implementacao tambem esta. A diagonal (consulta e catalogo no mesmo
  arm) fica disponivel como controle: e o teto do que a distancia crua alcanca
  quando nao ha troca de implementacao.

**O descritor.** A distancia e a mesma familia do oraculo (`disent/oracle.py`):
L1 sobre log-mel multi-resolucao. A diferenca e que aqui ela precisa rodar
5.600 x 5.600 vezes, e a pilha inteira tem 82.880 numeros por item -- 2,6e12
operacoes, inviavel. Entao o eixo do tempo e agrupado em `MEL_TIME_POOL` faixas
antes da comparacao, o que reduz o item a 4.096 numeros mantendo a distincao
ataque/corpo. `reduction_fidelity` mede o preco dessa reducao contra a distancia
integral, num subconjunto, e o resultado entra no relatorio: o baseline nao pode
se apoiar numa aproximacao cujo erro nao foi medido.

Como cada resolucao contribui com o mesmo numero de celulas depois do
agrupamento, a media simples sobre o vetor achatado reproduz a media-sobre-
resolucoes-de-media-sobre-celulas do oraculo. Nao ha peso implicito.
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

# Faixas de tempo mantidas no descritor. 1 seria o espectro medio, que apaga a
# diferenca entre ataque e sustentacao -- e distorcao age diferente nos dois.
#
# 16 e escolha medida, nao arbitraria: `reduction_fidelity` sobre 500 pares do
# split de consulta da rho de Spearman 0,556 / 0,701 / 0,880 / 0,973 / 0,993 para
# 1 / 4 / 8 / 16 / 32 faixas contra a distancia integral do oraculo. Em 16 a
# ordem induzida ja e praticamente a mesma (rho 0,97) a 1/4 do custo de 32, e e a
# ordem -- nao o valor -- que decide uma recuperacao.
MEL_TIME_POOL = 16

# Teto de memoria de um bloco de comparacao. A matriz de diferencas e
# (consultas x catalogo x dimensoes) em float32: com 4.096 dimensoes e 4.800
# itens de catalogo, cada linha de consulta ja custa 79 MB, entao o tamanho do
# bloco tem de sair de um orcamento e nao de uma constante.
NEAREST_BLOCK_BYTES = 512 * 1024 * 1024

DESCRIPTOR_FILENAME = "retrieval_descriptor.npz"

# Eixos da grade sobre os quais o acerto e reportado, com o tamanho do respectivo
# alfabeto -- e dele que sai o acaso.
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

    Toda resolucao contribui com a mesma quantidade de celulas, o que mantem a
    ponderacao do oraculo sem precisar de pesos explicitos.
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
    """`(descritores, nomes)` daquele arm, calculando e cacheando na primeira vez.

    O cache mora ao lado do audio, como o do POC I, e guarda `bands` junto: mudar
    o agrupamento invalida o cache em vez de misturar descritores incompativeis.
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
    """Descritores na ordem das linhas de `frame`, casados por nome de arquivo.

    Mesma armadilha de `disent/features.py`: a ordem do cache e a alfabetica dos
    wavs, que nao e a ordem do recorte pedido. Trocar as duas nao levanta erro --
    so responde a pergunta errada.
    """
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


#: Metricas de busca. `l1` e a do descritor cru (a mesma do oraculo, so que
#: reduzida); `cosine` e a do codigo aprendido, onde `z_e` vive na esfera e o
#: contrastivo otimiza produto interno. Buscar em L1 um espaco treinado em
#: cosseno mediria outra coisa que nao o que a rede aprendeu.
METRICS: Tuple[str, ...] = ("l1", "cosine")


def nearest(
    queries: np.ndarray,
    catalog: np.ndarray,
    chunk: Optional[int] = None,
    metric: str = "l1",
) -> Tuple[np.ndarray, np.ndarray]:
    """Indice e distancia do item de catalogo mais proximo de cada consulta.

    Em blocos porque a matriz de diferencas nao cabe inteira: 5.600 x 4.800 x
    4.096 em float32 seriam 440 GB. `chunk=None` dimensiona o bloco pelo
    orcamento de memoria, que e o que mantem isto valido quando o descritor muda
    de tamanho.
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
        # Sem bloco: o produto de matrizes e 5.600 x 5.600, cabe folgado, e a
        # normalizacao aqui torna a funcao segura mesmo se o codigo chegar sem
        # norma unitaria.
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
    """Acerto e erro por eixo, sempre ao lado do acaso do proprio eixo.

    Acerto exato sozinho nao diz nada: 25% e otimo em 8 niveis e pessimo em 2. E
    o erro medio importa mais que o acerto, porque errar por um nivel e um
    resultado diferente de errar por cinco.
    """
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
    """Quantos niveis cada eixo tem *neste recorte* -- o acaso sai daqui.

    Lido do dado e nao de `arms.py` de proposito: um recorte que so contenha
    parte da grade tem outro acaso, e reportar o acaso da grade cheia
    esconderia isso.
    """
    return {axis: int(frame[axis].nunique()) for axis in LEVEL_AXES}


def retrieve_by_arm(
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    query_vectors: np.ndarray,
    catalog_vectors: np.ndarray,
    same_arm: bool = False,
    metric: str = "l1",
) -> RetrievalResult:
    """A tarefa do POC II, dada uma representacao qualquer das duas particoes.

    Separada de `baseline_b0` porque e exatamente o mesmo protocolo que avalia o
    codigo aprendido: mesma exclusao do proprio arm, mesmo alfabeto, mesmo
    denominador de sorvedouro. Se as duas avaliacoes divergissem em qualquer
    detalhe, a comparacao entre B0 e a rede deixaria de medir a rede.

    `same_arm=False` (o padrao) e a tarefa: o catalogo nao contem o arm da
    consulta, entao a resposta tem de atravessar implementacoes.
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
        # Quantos itens cada consulta de fato pode alcancar. Sem `same_arm` o
        # proprio arm sai do catalogo, entao nao e `len(catalog)` -- e e este o
        # denominador da ocupacao esperada no diagnostico de sorvedouro.
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
    """Quanto o descritor agrupado no tempo diverge da distancia integral.

    Compara as duas distancias sobre `n_pairs` pares sorteados. O que importa
    para recuperacao e a **ordem**, nao o valor: por isso o relatorio traz a
    correlacao de postos, e nao so a de Pearson.
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

    `same_arm=False` (o padrao) e a tarefa do POC II: o catalogo nao contem o arm
    da consulta, entao a resposta tem de atravessar implementacoes.
    `same_arm=True` e o controle -- o mesmo procedimento sem essa travessia.
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
    """Acerto exato por (arm da consulta, arm recuperado).

    Diz de onde vieram os acertos: se um arm concentra as respostas, o baseline
    esta explorando proximidade de timbre e nao equivalencia de ajuste.
    """
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

    E o baseline mais importante para a tese do trabalho, porque responde
    diretamente "isto ja nao estava resolvido?". A saida dele e continua e ja
    esta na unidade da referencia (`drive_db` do catalogo, [5, 40]), a mesma de
    `drive_db_equivalente` -- entao o erro sai em dB sem nenhuma conversao, e e
    comparavel ao 1,28 dB que o POC I relata dentro da propria implementacao.

    Duas mudancas de dominio incidem sobre ele de uma vez, e o resultado nao as
    separa: a implementacao muda (e essa e a pergunta) e o estagio de tone e novo
    (o POC I nao tinha). Por isso o recorte em `pedalboard-tanh` importa: ali so
    a segunda mudanca age, e ele mede quanto do erro e dela.

    As features saem na hora, sem cache: sao 5.600 itens lidos uma vez, contra
    ~5 GB de `Spec.npz` que so a etapa de treino justifica.
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
            # O regressor do POC I nao prediz tone: nao havia estagio de tone la.
            # Fica constante para o eixo aparecer no relatorio com o valor que
            # tem -- o de acaso -- em vez de sumir.
            "pred_tone_level": -1,
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
    """Quantas consultas cada item de catalogo atrai, e o quanto isso desvia.

    Em espaco de alta dimensao, alguns itens viram vizinho de quase todo mundo --
    o efeito de *hub* (Radovanovic et al. 2010). Quando isso acontece, a resposta
    do vizinho mais proximo diz mais sobre a posicao do item no espaco do que
    sobre a consulta, e a acuracia agregada esconde o fenomeno. Assimetria alta e
    o sinal.
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
    """Acerto de drive por par (arm da consulta, arm do catalogo), um par por vez.

    Cada celula usa um catalogo de um arm so. Isso e o que separa duas coisas que
    o B0 agregado mistura: a **transferencia** entre aquele par especifico e a
    **competicao** entre arms dentro de um catalogo comum, que um sorvedouro
    domina. A diagonal e o controle sem troca de implementacao.
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
    """Fidelidade da reducao para varios agrupamentos, num sorteio de itens.

    Existe para que a escolha de `MEL_TIME_POOL` seja refazivel e nao um numero
    herdado. Calcula a pilha integral uma vez por item e reagrupa, entao varrer
    seis valores custa quase o mesmo que medir um.
    """
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
    """B0 no cenario do oraculo: candidatos do MESMO conteudo, outra implementacao.

    **Isto nao e a tarefa** -- exige que exista em disco a mesma execucao tocada
    sob todos os ajustes, que e justamente o que falta no uso real. E um teto, e
    existe para responder uma pergunta que o B0 sozinho nao responde: quando o
    vizinho mais proximo erra, e porque a distancia espectral nao distingue
    ajuste, ou porque ela esta ocupada distinguindo *gravacao*?

    A diferenca entre este numero e o do `baseline_b0` e, por construcao, o que o
    conteudo custa. Se for grande, o alvo da etapa 5 e invariancia a conteudo; se
    for pequeno, o problema esta na propria nocao de distancia e a rede tem menos
    a ganhar.
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
            # `retrieve` recusa conteudo compartilhado, que aqui e o ponto: a
            # montagem do quadro e feita a mao para dizer que a violacao e
            # deliberada e que o resultado e teto, nao desempenho.
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
