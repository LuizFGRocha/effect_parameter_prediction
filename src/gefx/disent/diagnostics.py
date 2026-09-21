"""Diagnosticos da etapa 6 sobre execucoes ja treinadas.

Sondas lineares (a perda do adversario no acaso nao prova remocao -- Elazar &
Goldberg, 2018), ablacao da representacao, subespacos da busca, reescore pelo
oraculo, resolucao da grade, DCI/MIG e o bootstrap agrupado por conteudo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair de z_e)",
    "content_id": "conteudo (deve sair de z_e)",
    "drive_level": "configuracao (deve ficar em z_e)",
    "tone_level": "configuracao (deve ficar em z_e)",
}

PROBE_BLOCKS: Tuple[str, ...] = ("z_e", "z_c")

DEFAULT_FOLDS = 3

EPSILON = 1e-8


def _factor_labels(frame: pd.DataFrame, factor: str) -> np.ndarray:
    if factor not in frame.columns:
        raise KeyError(f"fator ausente no sidecar: {factor!r}")
    return frame[factor].astype("category").cat.codes.to_numpy()


def _check_rows(codes: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.float64)
    if len(codes) != len(frame):
        raise ValueError(f"{len(codes)} codigos e {len(frame)} linhas")
    return codes


def linear_probes(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = tuple(PROBE_FACTORS),
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Acerto de uma regressao logistica por fator, com o acaso ao lado.

    Linear de proposito: pergunta se o fator esta legivel, nao se e recuperavel.

    A padronizacao e necessaria: com L2 de `C` fixo, sem ela a sonda compara a
    escala dos blocos (`z_e` tem norma 1, `z_c` nao e normalizado).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    codes = _check_rows(codes, frame)
    out: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        labels = _factor_labels(frame, factor)
        classes = int(len(set(labels)))
        chance = 1.0 / classes
        partition = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        accuracy = float(
            cross_val_score(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                codes, labels, cv=partition, n_jobs=folds,
            ).mean()
        )
        out[factor] = {
            "accuracy": accuracy,
            "chance": chance,
            "classes": classes,
            "above_chance": (accuracy - chance) / (1.0 - chance),
        }
    return out


def load_run(run_dir: Path):
    """`(modelo, manifesto)` de uma execucao gravada."""
    from gefx.disent.model import BetaVAE, DisentModel, EncoderConfig, HeadConfig

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    encoder_config = EncoderConfig.from_dict(manifest["config"]["encoder"])
    if manifest["config"]["technique"] == "beta_vae":
        model = BetaVAE(encoder_config, beta=manifest["config"].get("beta", 4.0))
    else:
        model = DisentModel(encoder_config, HeadConfig(**manifest["heads"]))
    model.load_weights(run_dir / "weights")
    return model, manifest


def embed_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    splits: Sequence[str] = ("catalog",),
    feature: str = "Spec",
):
    """`(manifesto, frames, codigos)` de uma execucao; `codigos[split] = (z_e, z_c)`.

    Arms e padronizador saem do que a execucao gravou.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.train import embed_blocks, split_frames

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
    frames = split_frames(dataset_root, manifest["config"].get("arms"))
    codes = {
        split: embed_blocks(model, FeatureStore(dataset_root, frames[split], feature),
                            standardizer)
        for split in splits
    }
    return manifest, frames, codes


def _runs(results_dir: Path, techniques: Sequence[str]) -> List[Tuple[str, Path]]:
    """`(tecnica, diretorio)` das tecnicas pedidas que tem `run.json`, na ordem pedida."""
    results_dir = Path(results_dir)
    found = [(name, results_dir / name) for name in techniques
             if (results_dir / name / "run.json").exists()]
    if not found:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")
    return found


def probe_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "catalog",
    feature: str = "Spec",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, object]:
    """Sonda os dois blocos de uma execucao ja treinada, do que ela gravou."""
    manifest, frames, codes = embed_run(run_dir, dataset_root, (split,), feature)
    frame = frames[split]
    blocos = dict(zip(PROBE_BLOCKS, codes[split]))
    return {
        "technique": manifest["config"]["technique"],
        "split": split,
        "n": int(len(frame)),
        "probes": {nome: linear_probes(codigo, frame, folds=folds, seed=seed)
                   for nome, codigo in blocos.items()},
    }


def probe_study(
    results_dir: Path = Path("results/disent/etapa5"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Optional[Sequence[str]] = None,
    split: str = "catalog",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por (tecnica, bloco, fator). Escreve nada; grava quem chama."""
    from gefx.disent.train import STUDY_ORDER

    rows: List[Dict[str, object]] = []
    for name, run_dir in _runs(results_dir, techniques or STUDY_ORDER):
        report = probe_run(run_dir, dataset_root, split=split, folds=folds, seed=seed)
        for bloco, probes in report["probes"].items():  # type: ignore[union-attr]
            for factor, numbers in probes.items():
                rows.append({"technique": name, "bloco": bloco, "factor": factor,
                             "meaning": PROBE_FACTORS.get(factor, ""), **numbers})
    return pd.DataFrame(rows)


# --- de onde vem o ganho sobre o B0 -------------------------------------------
#: Entre o B0 (descritor log-mel, 4.096-d) e o encoder (`Spec`, 44.288-d) mudam
#: representacao, escala, dimensao e arquitetura; a ablacao troca uma por vez.
ABLATION_DIMS: Tuple[int, ...] = (32,)


def _cross_retrieval(frames, queries, catalog, metric: str = "cosine"):
    """Consulta contra catalogo, sem o proprio arm: a tarefa da etapa 5."""
    from gefx.disent.retrieval import retrieve_by_arm

    return retrieve_by_arm(
        frames["query"], frames["catalog"], queries, catalog,
        same_arm=False, metric=metric,
    )


def _random_projection(rng: np.random.Generator, dim: int, size: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, size)).astype(np.float32)
    return matrix / np.sqrt(size)


def _score(frames, queries, catalog, metric: str = "cosine") -> Dict[str, float]:
    result = _cross_retrieval(frames, queries, catalog, metric)
    overall = result.metrics["overall"]
    return {
        "drive_exact": float(overall["drive_level"]["exact"]),
        "mae_db": float(overall["mae_db"]),
        "top_arm_share": float(
            result.predictions["retrieved_arm"].value_counts(normalize=True).max()
        ),
    }


def ablate_representation(
    dataset_root: Path = Path("datasets/disent"),
    standardizer_path: Optional[Path] = None,
    dims: Sequence[int] = ABLATION_DIMS,
    seed: int = 0,
) -> pd.DataFrame:
    """A escada entre o B0 e o encoder, uma troca por degrau, sem modelo nenhum."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.retrieval import load_descriptors
    from gefx.disent.train import map_store, split_frames

    dataset_root = Path(dataset_root)
    frames = split_frames(dataset_root)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    descriptors = {
        name: load_descriptors(dataset_root, frames[name]) for name in ("query", "catalog")
    }
    rows.append({"step": "B0: descritor, L1, 4096-d",
                 **_score(frames, descriptors["query"], descriptors["catalog"], "l1")})

    mean = descriptors["catalog"].mean(axis=0)
    std = descriptors["catalog"].std(axis=0) + 1e-8
    scaled = {name: (value - mean) / std for name, value in descriptors.items()}
    rows.append({"step": "+ padronizado (L1)",
                 **_score(frames, scaled["query"], scaled["catalog"], "l1")})

    for size in dims:
        matrix = _random_projection(rng, descriptors["query"].shape[1], size)
        rows.append({"step": f"+ projetado ao acaso em {size}-d (cosseno)",
                     **_score(frames, scaled["query"] @ matrix, scaled["catalog"] @ matrix)})

    # Troca de representacao: o mesmo tratamento sobre o `Spec`.
    standardizer = PixelStandardizer.load(
        standardizer_path
        or Path("results/disent/etapa5/contrastive_aux/standardizer.npz")
    )
    stores = {name: FeatureStore(dataset_root, frames[name], "Spec")
              for name in ("query", "catalog")}
    # Sorteadas antes da passada para projetar bloco a bloco: o Spec achatado
    # de um recorte tem ~1 GB.
    flat_dim = int(np.prod(stores["query"].feature_shape))
    matrices = [_random_projection(rng, flat_dim, size) for size in dims]
    projected = {
        name: map_store(
            store, standardizer,
            lambda t: tuple(t[..., 0].reshape(len(t), -1) @ m for m in matrices), 256,
        )
        for name, store in stores.items()
    }
    for position, size in enumerate(dims):
        rows.append({"step": f"Spec padronizado, projetado em {size}-d (cosseno)",
                     **_score(frames, projected["query"][position],
                              projected["catalog"][position])})

    return pd.DataFrame(rows)


# --- de que subespaco a busca vive --------------------------------------------
#: Direcoes discriminantes por fator: o posto maximo da LDA (classes - 1).
SUBSPACE_FACTORS: Dict[str, int] = {"tone_level": 4, "drive_level": 7}

#: Subespacos ao acaso da mesma dimensao, o controle de cada linha da LDA.
SUBSPACE_CONTROL_SEEDS: int = 5


def _discriminant_basis(codes: np.ndarray, labels, components: int) -> np.ndarray:
    """Base ortonormal do subespaco que melhor separa as classes do fator.

    Chamar so com o catalogo: nenhuma linha de consulta pode entrar no ajuste.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=components).fit(codes, labels)
    base, _ = np.linalg.qr(lda.scalings_[:, :components])
    return base.astype(np.float32)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)


def _retrieval_row(frames, query, catalog, label: str) -> Dict[str, object]:
    overall = _cross_retrieval(frames, query, catalog).metrics["overall"]
    return {"metrica": label, "dim": int(query.shape[1]),
            "drive_exact": float(overall["drive_level"]["exact"]),
            "tone_exact": float(overall["tone_level"]["exact"]),
            "mae_db": float(overall["mae_db"])}


def retrieval_subspaces(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    feature: str = "Spec",
    factors: Mapping[str, int] = SUBSPACE_FACTORS,
    controls: int = SUBSPACE_CONTROL_SEEDS,
) -> pd.DataFrame:
    """A mesma busca sobre recortes do codigo ja treinado, uma linha por recorte.

    Por fator: o subespaco discriminante sozinho, o codigo sem ele e `controls`
    subespacos ao acaso da mesma dimensao.
    """
    _, frames, blocks = embed_run(run_dir, dataset_root, ("query", "catalog"), feature)
    (q_e, q_c), (c_e, c_c) = blocks["query"], blocks["catalog"]

    rows = [
        _retrieval_row(frames, q_e, c_e, "z_e (a busca do trabalho)"),
        _retrieval_row(frames, q_c, c_c, "z_c sozinho"),
        _retrieval_row(frames, _unit(np.hstack([q_e, q_c])),
                       _unit(np.hstack([c_e, c_c])), "z_e + z_c"),
    ]
    identity = np.eye(c_e.shape[1], dtype=np.float32)
    for factor, components in factors.items():
        base = _discriminant_basis(c_e, frames["catalog"][factor].to_numpy(), components)
        rows.append(_retrieval_row(frames, _unit(q_e @ base), _unit(c_e @ base),
                                   f"z_e no subespaco de {factor} ({components}-d)"))
        rest = identity - base @ base.T
        rows.append(_retrieval_row(frames, _unit(q_e @ rest), _unit(c_e @ rest),
                                   f"z_e SEM o subespaco de {factor}"))
        for seed in range(controls):
            rng = np.random.default_rng(seed)
            acaso, _ = np.linalg.qr(rng.normal(size=(c_e.shape[1], components)))
            acaso = acaso.astype(np.float32)
            rows.append(_retrieval_row(
                frames, _unit(q_e @ acaso), _unit(c_e @ acaso),
                f"z_e em {components} direcoes ao acaso (semente {seed})"))
    return pd.DataFrame(rows)


# --- o oraculo sonico como verdade fundamental --------------------------------
#: O placar nominal supoe que o nivel k de um arm soa como o nivel k de outro;
#: estas funcoes reescoram o estudo pelo oraculo, sem essa suposicao.
def oracle_alignment(
    root: Path = Path("datasets/disent"),
    split: str = "query",
    bands: int = 16,
) -> Dict[str, object]:
    """Distancia pareada media entre cada nivel de drive de cada par de arms.

    Pareada por conteudo e por nivel de tom. Usa o descritor reduzido, o mesmo
    do teto de conteudo pareado.
    """
    from gefx.disent.retrieval import load_descriptors
    from gefx.disent.sidecar import read_dataset

    root = Path(root)
    frame = read_dataset(root)
    frame = frame[frame["split"] == split].reset_index(drop=True)
    descriptors = load_descriptors(root, frame, bands=bands)

    arms = sorted(frame["arm"].unique())
    drives = sorted(frame["drive_level"].unique())
    tones = sorted(frame["tone_level"].unique())
    contents = sorted(frame["content_id"].unique())

    shape = (len(arms), len(drives), len(tones), len(contents))
    cube = np.full((*shape, descriptors.shape[1]), np.nan, dtype=np.float32)
    position = {
        (arm, drive, tone, content): (a, d, t, c)
        for a, arm in enumerate(arms)
        for d, drive in enumerate(drives)
        for t, tone in enumerate(tones)
        for c, content in enumerate(contents)
    }
    for row in range(len(frame)):
        key = (frame["arm"][row], frame["drive_level"][row],
               frame["tone_level"][row], frame["content_id"][row])
        cube[position[key]] = descriptors[row]
    if np.isnan(cube[..., 0]).any():
        raise ValueError("a grade do split nao esta completa; o pareamento exige isso")

    alignment = np.empty((len(arms), len(arms), len(drives), len(drives)), dtype=np.float32)
    for a in range(len(arms)):
        for b in range(len(arms)):
            for p in range(len(drives)):
                alignment[a, b, p] = np.abs(cube[a, p][None] - cube[b]).mean(axis=(1, 2, 3))
    return {"arms": arms, "drives": drives, "alignment": alignment,
            "correspondence": alignment.argmin(axis=3)}


def label_noise(alignment: Mapping[str, object]) -> pd.DataFrame:
    """Onde o rotulo nominal discorda do oraculo, por par de implementacoes."""
    arms = list(alignment["arms"])  # type: ignore[arg-type]
    correspondence = np.asarray(alignment["correspondence"])
    nominal = np.arange(correspondence.shape[2])
    rows: List[Dict[str, object]] = []
    for a, query_arm in enumerate(arms):
        for b, catalog_arm in enumerate(arms):
            if a == b:
                continue
            picks = correspondence[a, b]
            rows.append({
                "query_arm": query_arm,
                "catalog_arm": catalog_arm,
                "niveis_movidos": int((picks != nominal).sum()),
                "desvio_medio": float((picks - nominal).mean()),
                "casamento": picks.tolist(),
            })
    return pd.DataFrame(rows)


def rescore_with_oracle(
    predictions: pd.DataFrame, alignment: Mapping[str, object]
) -> Dict[str, object]:
    """Acerto do estudo medido contra o oraculo em vez do rotulo nominal."""
    arms = {name: index for index, name in enumerate(alignment["arms"])}  # type: ignore
    correspondence = np.asarray(alignment["correspondence"])
    drives = list(alignment["drives"])  # type: ignore[arg-type]
    level = {value: index for index, value in enumerate(drives)}

    a = predictions["query_arm"].map(arms).to_numpy()
    b = predictions["retrieved_arm"].map(arms).to_numpy()
    p = predictions["true_drive_level"].map(level).to_numpy()
    alvo = np.array([drives[correspondence[i, j, k]] for i, j, k in zip(a, b, p)])
    previsto = predictions["pred_drive_level"].to_numpy()
    nominal = predictions["true_drive_level"].to_numpy()

    return {
        "nominal_exact": float(np.mean(previsto == nominal)),
        "oracle_exact": float(np.mean(previsto == alvo)),
        "alvos_que_diferem": float(np.mean(alvo != nominal)),
        "nominal_mae_levels": float(np.mean(np.abs(previsto - nominal))),
        "oracle_mae_levels": float(np.mean(np.abs(previsto - alvo))),
    }


# --- a grade e fina demais para a tarefa? -------------------------------------
LEVEL_SUBSETS: Dict[str, Sequence[int]] = {
    "8 niveis (4,15 dB por passo)": (0, 1, 2, 3, 4, 5, 6, 7),
    "4 niveis pares (8,3 dB)": (0, 2, 4, 6),
    "4 niveis impares (8,3 dB)": (1, 3, 5, 7),
    "3 niveis (12,5 dB)": (0, 3, 6),
    "2 niveis (24,9 dB)": (0, 7),
}


def grid_resolution(
    results_dir: Path = Path("results/disent/etapa5"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Sequence[str] = ("contrastive_aux", "random_encoder"),
    subsets: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Acerto sobre subconjuntos de niveis; compare `acima_do_acaso`, nao o cru."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.retrieval import retrieve_by_arm
    from gefx.disent.train import embed, split_frames

    dataset_root = Path(dataset_root)
    subsets = dict(subsets or LEVEL_SUBSETS)
    frames = split_frames(dataset_root)
    rows: List[Dict[str, object]] = []

    for technique, run_dir in _runs(results_dir, techniques):
        model, _ = load_run(run_dir)
        standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
        codes = {
            name: embed(model, FeatureStore(dataset_root, frames[name], "Spec"),
                        standardizer, 64)
            for name in ("query", "catalog")
        }
        for label, levels in subsets.items():
            queries = frames["query"][frames["query"]["drive_level"].isin(list(levels))]
            catalog = frames["catalog"][frames["catalog"]["drive_level"].isin(list(levels))]
            result = retrieve_by_arm(
                queries.reset_index(drop=True), catalog.reset_index(drop=True),
                codes["query"][queries.index.to_numpy()],
                codes["catalog"][catalog.index.to_numpy()],
                same_arm=False, metric="cosine",
            )
            overall = result.metrics["overall"]
            chance = 1.0 / len(levels)
            exact = float(overall["drive_level"]["exact"])
            rows.append({
                "technique": technique, "grade": label, "niveis": len(levels),
                "acerto": exact, "acaso": chance,
                "acima_do_acaso": (exact - chance) / (1.0 - chance),
                "mae_db": float(overall["mae_db"]),
            })
    return pd.DataFrame(rows)


# --- etapa 6: onde cada fator esta escrito ------------------------------------
STRUCTURE_FACTORS: Tuple[str, ...] = ("drive_level", "tone_level", "arm", "content_id")

#: Valor do `disentanglement_lib`; o MIG e sensivel a ele.
MIG_BINS = 20

DEFAULT_TREES = 200


def _entropy(weights: np.ndarray, base: int) -> float:
    """Entropia de pesos nao negativos, na base pedida."""
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or base <= 1:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-(p * (np.log(p) / np.log(base))).sum())


def importance_matrix(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = STRUCTURE_FACTORS,
    seed: int = 0,
    trees: int = DEFAULT_TREES,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Matriz `R[dimensao, fator]` de importancia, e a informatividade ao lado.

    Floresta, e nao sonda linear, porque aqui a pergunta e em que dimensoes o
    fator mora. A informatividade sai da metade retida.

    A importancia de Gini favorece dimensoes de ruido continuo, entao o teto
    pratico de `D` fica perto de 0,8, nao de 1,0.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    codes = _check_rows(codes, frame)
    matrix = np.zeros((codes.shape[1], len(factors)), dtype=np.float64)
    informativeness: Dict[str, Dict[str, float]] = {}
    for column, factor in enumerate(factors):
        labels = _factor_labels(frame, factor)
        x_fit, x_test, y_fit, y_test = train_test_split(
            codes, labels, test_size=0.5, random_state=seed, stratify=labels
        )
        forest = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1)
        forest.fit(x_fit, y_fit)
        matrix[:, column] = forest.feature_importances_
        informativeness[factor] = {
            "accuracy": float(forest.score(x_test, y_test)),
            "chance": 1.0 / int(len(set(labels))),
        }
    return matrix, informativeness


def dci_scores(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = STRUCTURE_FACTORS,
    blocks: Optional[Mapping[str, Sequence[int]]] = None,
    seed: int = 0,
    trees: int = DEFAULT_TREES,
) -> Dict[str, object]:
    """DCI de Eastwood & Williams (2018), mais a leitura por bloco.

    D e por dimensao, e nada na perda pede isso; a afirmacao do trabalho e por
    bloco, em `block_mass` (fracao da importancia de cada fator em cada bloco).
    """
    matrix, informativeness = importance_matrix(codes, frame, factors, seed, trees)
    factors = tuple(factors)

    # D: concentracao de cada dimensao num fator, ponderada pela massa dela.
    per_latent = np.array([1.0 - _entropy(row, len(factors)) for row in matrix])
    mass = matrix.sum(axis=1)
    disentanglement = float((per_latent * mass).sum() / mass.sum()) if mass.sum() else 0.0
    # C: concentracao de cada fator em poucas dimensoes.
    completeness = {
        factor: 1.0 - _entropy(matrix[:, column], matrix.shape[0])
        for column, factor in enumerate(factors)
    }

    out: Dict[str, object] = {
        "factors": list(factors),
        "importance": matrix,
        "disentanglement": disentanglement,
        "per_latent_disentanglement": per_latent,
        "completeness": completeness,
        "informativeness": informativeness,
    }
    if blocks:
        block_mass: Dict[str, Dict[str, float]] = {}
        block_completeness: Dict[str, float] = {}
        for column, factor in enumerate(factors):
            total = matrix[:, column].sum()
            shares = {
                name: float(matrix[list(index), column].sum() / total) if total else 0.0
                for name, index in blocks.items()
            }
            block_mass[factor] = shares
            block_completeness[factor] = 1.0 - _entropy(
                np.array(list(shares.values())), len(shares)
            )
        out["block_mass"] = block_mass
        out["block_completeness"] = block_completeness
    return out


def mutual_information_gap(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = STRUCTURE_FACTORS,
    bins: int = MIG_BINS,
) -> Dict[str, Dict[str, float]]:
    """MIG de Chen et al. (2018): a distancia entre as duas dimensoes que mais
    sabem de cada fator, normalizada pela entropia do fator. Sem classificador.
    """
    from sklearn.metrics import mutual_info_score

    codes = _check_rows(codes, frame)
    binned = np.empty(codes.shape, dtype=np.int32)
    for dim in range(codes.shape[1]):
        column = codes[:, dim]
        low, high = float(column.min()), float(column.max())
        if high <= low:  # dimensao morta: uma caixa so, informacao mutua zero
            binned[:, dim] = 0
            continue
        edges = np.linspace(low, high, bins + 1)[1:-1]
        binned[:, dim] = np.digitize(column, edges)

    out: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        labels = _factor_labels(frame, factor)
        entropy = mutual_info_score(labels, labels)
        scores = np.array([
            mutual_info_score(labels, binned[:, dim]) for dim in range(codes.shape[1])
        ])
        order = np.argsort(scores)[::-1]
        gap = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
        out[factor] = {
            "mig": gap / entropy if entropy > 0 else 0.0,
            "top_latent": int(order[0]),
            "i_top": float(scores[order[0]]),
            "i_second": float(scores[order[1]]) if len(order) > 1 else 0.0,
            "entropy": float(entropy),
        }
    return out


def structure_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "catalog",
    feature: str = "Spec",
    factors: Sequence[str] = STRUCTURE_FACTORS,
    seed: int = 0,
    trees: int = DEFAULT_TREES,
    bins: int = MIG_BINS,
) -> Dict[str, object]:
    """DCI e MIG sobre `[z_e | z_c]` de uma execucao ja treinada."""
    manifest, frames, blocos = embed_run(run_dir, dataset_root, (split,), feature)
    frame = frames[split]
    z_e, z_c = blocos[split]
    codes = np.concatenate([z_e, z_c], axis=1)
    blocks = {
        "z_e": range(z_e.shape[1]),
        "z_c": range(z_e.shape[1], codes.shape[1]),
    }

    dci = dci_scores(codes, frame, factors, blocks=blocks, seed=seed, trees=trees)
    return {
        "technique": manifest["config"]["technique"],
        "split": split,
        "n": int(len(frame)),
        "dims": {"z_e": int(z_e.shape[1]), "z_c": int(z_c.shape[1])},
        "dci": dci,
        "mig": mutual_information_gap(codes, frame, factors, bins=bins),
    }


def structure_study(
    results_dir: Path = Path("results/disent/etapa5"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Optional[Sequence[str]] = None,
    split: str = "catalog",
    factors: Sequence[str] = STRUCTURE_FACTORS,
    seed: int = 0,
    trees: int = DEFAULT_TREES,
) -> pd.DataFrame:
    """Uma linha por (tecnica, fator). `disentanglement` e por tecnica e repete."""
    from gefx.disent.train import STUDY_ORDER

    rows: List[Dict[str, object]] = []
    for name, run_dir in _runs(results_dir, techniques or STUDY_ORDER):
        report = structure_run(run_dir, dataset_root, split=split, factors=factors,
                               seed=seed, trees=trees)
        dci = report["dci"]  # type: ignore[index]
        mig = report["mig"]  # type: ignore[index]
        dims = report["dims"]  # type: ignore[index]
        # Um codigo que nao separa nada espalha a massa na proporcao das dimensoes.
        nula = dims["z_e"] / (dims["z_e"] + dims["z_c"])
        for factor in factors:
            rows.append({
                "technique": name,
                "factor": factor,
                "esperado_em": "z_e" if factor in ("drive_level", "tone_level") else "z_c",
                "massa_z_e": dci["block_mass"][factor]["z_e"],
                "massa_nula": nula,
                "dim_z_e": dims["z_e"],
                "dim_z_c": dims["z_c"],
                "separacao_por_bloco": dci["block_completeness"][factor],
                "completude": dci["completeness"][factor],
                "informatividade": dci["informativeness"][factor]["accuracy"],
                "acaso": dci["informativeness"][factor]["chance"],
                "mig": mig[factor]["mig"],
                "desemaranhamento": dci["disentanglement"],
            })
    return pd.DataFrame(rows)


# --- a barra que decide se duas tecnicas sao diferentes ------------------------
#: As consultas nao sao independentes: o conteudo domina a dificuldade, entao o
#: IC reamostra conteudos inteiros, nao linhas.
BOOTSTRAP_REPS = 4000

#: O nome do arquivo se repete entre implementacoes; so o par identifica a consulta.
QUERY_KEY = ("file_name", "query_arm")


def read_predictions(run_dir: Path, axis: str = "drive_level") -> pd.DataFrame:
    """`predictions.csv` de uma execucao, com a coluna de acerto do eixo pedido."""
    frame = pd.read_csv(Path(run_dir) / "predictions.csv")
    frame["acerto"] = (frame[f"pred_{axis}"] == frame[f"true_{axis}"]).astype(float)
    return frame.set_index(list(QUERY_KEY))


def clustered_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
    cluster: str = "query_content",
) -> Dict[str, float]:
    """Diferenca de acerto entre duas execucoes, em pontos, com IC 95% agrupado.

    Pareada linha a linha e reamostrada por `cluster`.
    """
    junto = a.join(b["acerto"].rename("acerto_b"), how="inner")
    if len(junto) != len(a) or len(junto) != len(b):
        raise ValueError(
            f"as duas execucoes nao respondem as mesmas consultas: "
            f"{len(a)} e {len(b)} linhas dao {len(junto)} pareadas"
        )
    # Uma reamostra so depende de quantas vezes cada grupo saiu.
    somas = junto.groupby(cluster)[["acerto", "acerto_b"]].agg(["sum", "size"])
    soma_a = somas[("acerto", "sum")].to_numpy()
    soma_b = somas[("acerto_b", "sum")].to_numpy()
    tamanho = somas[("acerto", "size")].to_numpy()
    n_grupos = len(tamanho)
    rng = np.random.default_rng(seed)
    observado = float(junto["acerto"].mean() - junto["acerto_b"].mean()) * 100
    amostras = np.empty(reps)
    for indice in range(reps):
        contagem = np.bincount(rng.integers(0, n_grupos, n_grupos), minlength=n_grupos)
        amostras[indice] = (contagem @ soma_a - contagem @ soma_b) / (contagem @ tamanho) * 100
    baixo, alto = (float(valor) for valor in np.percentile(amostras, [2.5, 97.5]))
    return {
        "diferenca_pontos": observado,
        "ic_baixo": baixo,
        "ic_alto": alto,
        "grupos": n_grupos,
        "n": int(len(junto)),
        "distinguivel": bool(baixo > 0 or alto < 0),
    }


def bootstrap_study(
    runs: Mapping[str, Path],
    pairs: Sequence[Tuple[str, str]],
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
    axis: str = "drive_level",
) -> pd.DataFrame:
    """Uma linha por par comparado. `runs` mapeia nome -> diretorio de execucao."""
    carregadas = {nome: read_predictions(caminho, axis=axis)
                  for nome, caminho in runs.items()}
    rows: List[Dict[str, object]] = []
    for esquerda, direita in pairs:
        numbers = clustered_bootstrap(carregadas[esquerda], carregadas[direita],
                                      reps=reps, seed=seed)
        rows.append({"a": esquerda, "b": direita, "eixo": axis, **numbers})
    return pd.DataFrame(rows)


def find_runs(
    results_dir: Path = Path("results/disent/etapa5"),
    extra: Sequence[str] = ("pesos", "largura"),
) -> Dict[str, Path]:
    """Execucoes com `predictions.csv` sob o diretorio e sob as subpastas `extra`.

    Uma pasta sem `predictions.csv` com exatamente uma execucao dentro (uma
    varredura grava `<recorte>/<tecnica>/`) conta com o nome do recorte.
    """
    results_dir = Path(results_dir)
    encontradas: Dict[str, Path] = {}
    for base in [results_dir, *(results_dir / nome for nome in extra)]:
        if not base.is_dir():
            continue
        for pasta in sorted(base.iterdir()):
            if not pasta.is_dir() or pasta.name in encontradas or pasta.name in extra:
                continue
            if (pasta / "predictions.csv").exists():
                encontradas[pasta.name] = pasta
                continue
            dentro = [filho for filho in sorted(pasta.iterdir())
                      if filho.is_dir() and (filho / "predictions.csv").exists()]
            if len(dentro) == 1:
                encontradas[pasta.name] = dentro[0]
    return encontradas
