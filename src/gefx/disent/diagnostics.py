"""Sondas lineares sobre os codigos aprendidos.

Existe porque a perda do adversario **nao e evidencia de remocao**. No estudo da
etapa 5 o classificador de implementacao ficou no acaso (ln 7 = 1,946) do
primeiro passo ao ultimo, e mesmo assim uma regressao logistica ajustada depois
sobre o mesmo `z_e` le a implementacao a 25,7% contra 14,3% de acaso. Um
adversario no acaso significa que ele parou de achar o sinal, nao que o sinal
saiu -- e o resultado conhecido de Elazar & Goldberg (2018) sobre remocao
adversaria.

E a sonda que decide entre as duas leituras, e por isso ela mora no codigo e nao
num script: o numero entra no relatorio.

Este modulo e o comeco da etapa 6. DCI e MIG entram aqui depois; a interface
(`probe_report` sobre um diretorio de execucao) ja e a que eles vao usar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Fatores sondados e o que cada resposta significa. `arm` e `content_id` sao os
#: que o desemaranhamento promete remover de `z_e`; `drive_level` e o que ele
#: promete manter -- sondar so os dois primeiros mediria metade da afirmacao.
PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair de z_e)",
    "content_id": "conteudo (deve sair de z_e)",
    "drive_level": "configuracao (deve ficar em z_e)",
}

DEFAULT_FOLDS = 3


def linear_probes(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = tuple(PROBE_FACTORS),
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Acerto de uma regressao logistica por fator, com o acaso ao lado.

    Linear de proposito: a pergunta e se o fator esta **legivel** no codigo, e
    uma sonda nao-linear responderia outra coisa (se o fator e recuperavel por
    algum modelo, o que quase sempre e verdade). Validacao cruzada porque o
    numero de amostras por classe e pequeno -- 20 conteudos de catalogo.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    codes = np.asarray(codes, dtype=np.float64)
    if len(codes) != len(frame):
        raise ValueError(f"{len(codes)} codigos e {len(frame)} linhas")

    out: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        if factor not in frame.columns:
            raise KeyError(f"fator ausente no sidecar: {factor!r}")
        labels = frame[factor].astype("category").cat.codes.to_numpy()
        classes = int(len(set(labels)))
        chance = 1.0 / classes
        partition = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        accuracy = float(
            cross_val_score(
                LogisticRegression(max_iter=2000), codes, labels,
                cv=partition, n_jobs=folds,
            ).mean()
        )
        out[factor] = {
            "accuracy": accuracy,
            "chance": chance,
            "classes": classes,
            # Quanto do caminho entre o acaso e o acerto total foi percorrido.
            # Comparar 25,7% em 7 classes com 16,5% em 20 nao diria nada sem isso.
            "above_chance": (accuracy - chance) / (1.0 - chance),
        }
    return out


def probe_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "catalog",
    feature: str = "Spec",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, object]:
    """Sonda o `z_e` de uma execucao ja treinada, a partir do que ela gravou."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import BetaVAE, DisentModel, EncoderConfig, HeadConfig
    from gefx.disent.train import embed, split_frames

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    encoder_config = EncoderConfig.from_dict(manifest["config"]["encoder"])
    # O controle beta-VAE tem outro encoder (a posterior sai com o dobro da
    # largura). Sondar os dois pelo mesmo caminho e o ponto: o que se compara e
    # o `z_e` que cada um oferece a busca, e nao a arquitetura que o produziu.
    if manifest["config"]["technique"] == "beta_vae":
        model = BetaVAE(encoder_config, beta=manifest["config"].get("beta", 4.0))
    else:
        model = DisentModel(encoder_config, HeadConfig(**manifest["heads"]))
    model.load_weights(run_dir / "weights")

    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    store = FeatureStore(dataset_root, frame, feature)
    codes = embed(model, store, PixelStandardizer.load(run_dir / "standardizer.npz"))
    return {
        "technique": manifest["config"]["technique"],
        "split": split,
        "n": int(len(frame)),
        "probes": linear_probes(codes, frame, folds=folds, seed=seed),
    }


def probe_study(
    results_dir: Path = Path("results/disent/etapa5"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Optional[Sequence[str]] = None,
    split: str = "catalog",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por (tecnica, fator). Escreve nada; quem grava e quem chama."""
    from gefx.disent.train import STUDY_ORDER

    results_dir = Path(results_dir)
    wanted = list(techniques) if techniques else list(STUDY_ORDER)
    rows: List[Dict[str, object]] = []
    for name in wanted:
        run_dir = results_dir / name
        if not (run_dir / "run.json").exists():
            continue
        report = probe_run(run_dir, dataset_root, split=split, folds=folds, seed=seed)
        for factor, numbers in report["probes"].items():  # type: ignore[union-attr]
            rows.append({"technique": name, "factor": factor,
                         "meaning": PROBE_FACTORS.get(factor, ""), **numbers})
    if not rows:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")
    return pd.DataFrame(rows)


# --- de onde vem o ganho sobre o B0 -------------------------------------------
#: O B0 e o encoder nao leem a mesma coisa: o B0 opera sobre o descritor log-mel
#: multirresolucao (4.096-d) e o encoder sobre o `Spec` (256 x 173 = 44.288-d).
#: Comparar os dois de frente mistura quatro mudancas -- representacao, escala,
#: dimensao e arquitetura -- e credita todas ao desemaranhamento. Esta ablacao
#: separa as quatro trocando **uma de cada vez**, e foi ela que mostrou que a
#: reducao de dimensao explica o fim do efeito de sorvedouro mas **nao** explica
#: o ganho de acerto.
ABLATION_DIMS: Tuple[int, ...] = (32,)


def _score(frames, queries, catalog, metric: str = "cosine") -> Dict[str, float]:
    from gefx.disent.retrieval import retrieve_by_arm

    result = retrieve_by_arm(
        frames["query"], frames["catalog"], queries, catalog,
        same_arm=False, metric=metric,
    )
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
    """A escada entre o B0 e o encoder nao treinado, uma troca por degrau.

    Nao carrega modelo nenhum: todos os degraus sao lineares ou nada. A linha do
    encoder entra depois, do `run.json` -- e a diferenca entre o ultimo degrau
    daqui e ela que mede o vies indutivo da arquitetura convolucional.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.retrieval import load_descriptors
    from gefx.disent.train import split_frames

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
        matrix = rng.normal(size=(descriptors["query"].shape[1], size)).astype(np.float32)
        matrix /= np.sqrt(size)
        rows.append({"step": f"+ projetado ao acaso em {size}-d (cosseno)",
                     **_score(frames, scaled["query"] @ matrix, scaled["catalog"] @ matrix)})

    # Troca de representacao: o mesmo tratamento sobre o `Spec`, que e o que o
    # encoder de fato le.
    standardizer = PixelStandardizer.load(
        standardizer_path
        or Path("results/disent/etapa5/contrastive_aux/standardizer.npz")
    )
    flat = {}
    for name in ("query", "catalog"):
        store = FeatureStore(dataset_root, frames[name], "Spec")
        block_out = None
        for block, features in store.stream(np.arange(len(store)), chunk=256):
            plane = standardizer.transform(features)[..., 0].reshape(len(block), -1)
            if block_out is None:
                block_out = np.empty((len(store), plane.shape[1]), dtype=np.float32)
            block_out[block] = plane
        flat[name] = block_out

    for size in dims:
        matrix = rng.normal(size=(flat["query"].shape[1], size)).astype(np.float32)
        matrix /= np.sqrt(size)
        rows.append({"step": f"Spec padronizado, projetado em {size}-d (cosseno)",
                     **_score(frames, flat["query"] @ matrix, flat["catalog"] @ matrix)})

    return pd.DataFrame(rows)


# --- o rotulo nominal e mesmo a verdade fundamental? --------------------------
#: A etapa 5 pontua "acertou" comparando `pred_drive_level` com
#: `true_drive_level`. Isso **assume** que o nivel 3 de um arm soa como o nivel 3
#: de outro -- que e o que a calibracao tentou garantir e o que o oraculo ja
#: mostrou ser falso no `byod-bigmuff` (casamento `[0,0,0,0,2,4,5,6]`). Onde a
#: suposicao falha, o placar cobra do modelo um erro que e do rotulo.
#:
#: O desenho do trabalho diz que a verdade fundamental e o **oraculo sonico**,
#: nao o rotulo. Estas funcoes reescoram o estudo por ele.
def oracle_alignment(
    root: Path = Path("datasets/disent"),
    split: str = "query",
    bands: int = 16,
) -> Dict[str, object]:
    """Distancia pareada media entre cada nivel de drive de cada par de arms.

    Pareada por conteudo **e por nivel de tom**: so o drive varia entre os dois
    lados da comparacao. E o cenario do oraculo, que exige o mesmo conteudo em
    disco -- disponivel aqui porque a grade e totalmente cruzada.

    Usa o descritor reduzido, nao a pilha integral: a reducao tem Spearman 0,973
    contra a distancia integral, medido, e e o mesmo descritor com que o teto de
    conteudo pareado foi calculado. Trocar de distancia entre o teto e este
    numero e que seria incomparavel.
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
    """Onde o rotulo nominal discorda do oraculo, por par de implementacoes.

    Uma linha por (arm consultado, arm respondido) com quantos dos niveis o
    oraculo move e para onde. E a medida direta do ruido de rotulo que o placar
    da etapa 5 esta cobrando do modelo.
    """
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
    """Acerto do estudo medido contra o oraculo em vez do rotulo nominal.

    A diferenca entre os dois numeros nao e "correcao": e a parcela do erro
    reportado que era do rotulo e nao do modelo.
    """
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
#: Subconjuntos de niveis. A grade esta totalmente cruzada em disco, entao dobrar
#: o passo do eixo e pontuar sobre niveis alternados -- sem renderizar nada, sem
#: recalibrar e sem invalidar os testes de escuta ja feitos. E a versao gratuita
#: da pergunta "e se o eixo tivesse mais resolucao perceptual".
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
    """Acerto sobre subconjuntos de niveis, com o acaso de cada grade ao lado.

    O numero que importa e o **normalizado pelo acaso**: engrossar a grade sobe o
    acerto cru por construcao, e comparar 8 niveis com 2 sem normalizar nao diz
    nada. Se a distancia entre a tecnica treinada e o encoder nao treinado for a
    mesma em todas as resolucoes, a resolucao nao e o gargalo do metodo -- e
    regerar o dataset com um eixo mais grosso compraria um numero melhor, nao uma
    conclusao diferente.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import DisentModel, EncoderConfig, HeadConfig
    from gefx.disent.retrieval import retrieve_by_arm
    from gefx.disent.train import embed, split_frames

    results_dir, dataset_root = Path(results_dir), Path(dataset_root)
    subsets = dict(subsets or LEVEL_SUBSETS)
    frames = split_frames(dataset_root)
    rows: List[Dict[str, object]] = []

    for technique in techniques:
        run_dir = results_dir / technique
        if not (run_dir / "run.json").exists():
            continue
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        model = DisentModel(EncoderConfig.from_dict(manifest["config"]["encoder"]),
                            HeadConfig(**manifest["heads"]))
        model.load_weights(run_dir / "weights")
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
    if not rows:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")
    return pd.DataFrame(rows)
