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

Este modulo e a etapa 6 inteira: as sondas, a ablacao da representacao, o
oraculo como verdade fundamental alternativa, a resolucao da grade e -- na
ultima secao -- o DCI e o MIG, que perguntam **onde** cada fator esta escrito em
vez de perguntar se ele esta legivel.
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
    "tone_level": "configuracao (deve ficar em z_e)",
}

#: Os dois blocos do codigo, sondados separadamente. Sondar so o `z_e` responde
#: metade da pergunta: um fator que devia estar nele pode estar **legivel** e
#: ainda assim estar escrito sobretudo no `z_c` -- e o `z_c` nao entra na busca,
#: entao a recuperacao nao alcanca o que esta la. E a diferenca entre "o codigo
#: nao tem a informacao" e "a informacao esta no bloco errado", que sao
#: diagnosticos com consertos opostos.
PROBE_BLOCKS: Tuple[str, ...] = ("z_e", "z_c")

DEFAULT_FOLDS = 3

#: Piso da norma, para nao dividir por zero num codigo degenerado.
EPSILON = 1e-8


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

    **A padronizacao nao e cosmetica, e sem ela a sonda mede a escala do bloco.**
    A `LogisticRegression` tem penalidade L2 com `C=1` fixo, entao um bloco de
    norma grande recebe muito menos regularizacao efetiva que um de norma
    pequena. Os dois blocos deste trabalho vivem em escalas incomparaveis por
    construcao: o `z_e` e L2-normalizado (norma 1) e o `z_c` nao e -- norma media
    10,4 na tecnica `contrastive_aux` e **233,7** na `full`. Sem padronizar, a
    comparacao entre blocos media sobretudo isso, e o `lbfgs` nem convergia em
    2.000 iteracoes. O escalonador entra **dentro** da validacao cruzada, ajustado
    so na dobra de treino.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

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
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                codes, labels, cv=partition, n_jobs=folds,
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


def load_run(run_dir: Path):
    """`(modelo, manifesto)` de uma execucao gravada.

    O `beta_vae` tem outro encoder (a posterior sai com o dobro da largura) e
    precisa ser reconstruido pela classe dele. Carregar os dois pelo mesmo lugar e
    o que garante que todo diagnostico veja o mesmo `z_e` que a busca vE.
    """
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


def probe_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "catalog",
    feature: str = "Spec",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, object]:
    """Sonda os dois blocos de uma execucao ja treinada, do que ela gravou."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.train import embed_blocks, split_frames

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    store = FeatureStore(dataset_root, frame, feature)
    z_e, z_c = embed_blocks(model, store, PixelStandardizer.load(run_dir / "standardizer.npz"))
    blocos = dict(zip(PROBE_BLOCKS, (z_e, z_c)))
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

    results_dir = Path(results_dir)
    wanted = list(techniques) if techniques else list(STUDY_ORDER)
    rows: List[Dict[str, object]] = []
    for name in wanted:
        run_dir = results_dir / name
        if not (run_dir / "run.json").exists():
            continue
        report = probe_run(run_dir, dataset_root, split=split, folds=folds, seed=seed)
        for bloco, probes in report["probes"].items():  # type: ignore[union-attr]
            for factor, numbers in probes.items():
                rows.append({"technique": name, "bloco": bloco, "factor": factor,
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


# --- de que subespaco a busca vive --------------------------------------------
#: Quantas direcoes discriminantes cada fator recebe. E o posto maximo de uma
#: LDA: numero de classes menos uma. Nao e escolha de hiperparametro.
SUBSPACE_FACTORS: Dict[str, int] = {"tone_level": 4, "drive_level": 7}

#: Quantos subespacos ao acaso, da MESMA dimensao, acompanham cada fator. Sem
#: este controle a linha da LDA nao significa nada: "4 direcoes recuperam o tom"
#: pode ser apenas "o tom sobrevive a qualquer projecao em 4 dimensoes" -- e para
#: o drive e exatamente esse o caso.
SUBSPACE_CONTROL_SEEDS: int = 5


def _discriminant_basis(codes: np.ndarray, labels, components: int) -> np.ndarray:
    """Base ortonormal do subespaco que melhor separa as classes do fator.

    Ajustada **no catalogo**, cujos rotulos sao conhecidos por construcao -- e
    portanto uma operacao legitima de tempo de inferencia, e nao um vazamento:
    nenhuma linha do recorte de consulta entra no ajuste.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=components).fit(codes, labels)
    base, _ = np.linalg.qr(lda.scalings_[:, :components])
    return base.astype(np.float32)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)


def _retrieval_row(frames, query, catalog, label: str) -> Dict[str, object]:
    from gefx.disent.retrieval import retrieve_by_arm

    overall = retrieve_by_arm(
        frames["query"], frames["catalog"], query, catalog,
        same_arm=False, metric="cosine",
    ).metrics["overall"]
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
    """A mesma busca, sobre recortes do mesmo codigo ja treinado.

    Responde uma pergunta que a sonda linear nao responde: a sonda diz que o
    fator esta **legivel** no bloco, e a busca pode ainda assim nao alcanca-lo --
    porque o cosseno soma todas as direcoes de uma vez e o eixo em disputa pode
    abafar o outro. Cada linha e uma metrica diferente sobre os mesmos vetores;
    nada aqui retreina coisa alguma.

    Os dois recortes que carregam o resultado sao o par por fator: o subespaco
    discriminante **sozinho**, e o codigo **sem** ele. Se o eixo estivesse
    abafado, o primeiro subiria; se o eixo estivesse concentrado, o segundo
    desabaria.

    Cada fator vem acompanhado de `controls` subespacos **ao acaso** da mesma
    dimensao, e eles nao sao decoracao: sem essa linha, "k direcoes recuperam o
    eixo" pode ser so "o eixo sobrevive a qualquer projecao em k dimensoes". No
    drive e exatamente isso que acontece -- um subespaco aleatorio de 7 direcoes
    recupera tanto quanto o discriminante.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.train import embed_blocks, split_frames

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
    frames = split_frames(dataset_root, manifest["config"].get("arms"))
    blocks = {
        name: embed_blocks(model, FeatureStore(dataset_root, frames[name], feature),
                           standardizer)
        for name in ("query", "catalog")
    }
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


# --- etapa 6: onde cada fator esta escrito ------------------------------------
#: Fatores das metricas de estrutura. Sao os quatro que a grade cruza -- os dois
#: que `z_e` deve carregar (`drive_level`, `tone_level`) e os dois que ele deve
#: largar (`arm`, `content_id`). Medir so os primeiros mediria meia afirmacao.
STRUCTURE_FACTORS: Tuple[str, ...] = ("drive_level", "tone_level", "arm", "content_id")

#: Caixas por dimensao latente na discretizacao do MIG. E o valor do
#: `disentanglement_lib`; o MIG e sensivel a ele, entao ele fica declarado e nao
#: escolhido por execucao.
MIG_BINS = 20

DEFAULT_TREES = 200


def _entropy(weights: np.ndarray, base: int) -> float:
    """Entropia de uma distribuicao ja normalizada, na base pedida."""
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

    Floresta aleatoria e nao regressao logistica de proposito, e a escolha muda o
    que se mede: as sondas de `linear_probes` perguntam se o fator esta
    **legivel** linearmente; aqui a pergunta e **em que dimensoes** ele mora, e
    para isso e preciso um modelo que atribua importancia por dimensao. O
    `disentanglement_lib` usa arvores impulsionadas pela mesma razao; a floresta
    da a mesma leitura e cabe no orcamento com 20 classes de conteudo.

    A informatividade sai de uma metade retida: importancia se mede no ajuste,
    acerto nao.

    **Vies conhecido:** a importancia de Gini prefere dimensoes continuas de
    cardinalidade alta. Num codigo plantado, com um fator por dimensao e uma
    dimensao de puro ruido, a de ruido recebe mais credito que as dimensoes dos
    outros fatores (esta medido em `tests/disent/test_diagnostics.py`). Por isso
    o teto pratico de `D` fica na casa de 0,8, e nao em 1,0 -- ler a diferenca
    como emaranhamento seria ler o estimador.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    codes = np.asarray(codes, dtype=np.float64)
    if len(codes) != len(frame):
        raise ValueError(f"{len(codes)} codigos e {len(frame)} linhas")

    matrix = np.zeros((codes.shape[1], len(factors)), dtype=np.float64)
    informativeness: Dict[str, Dict[str, float]] = {}
    for column, factor in enumerate(factors):
        if factor not in frame.columns:
            raise KeyError(f"fator ausente no sidecar: {factor!r}")
        labels = frame[factor].astype("category").cat.codes.to_numpy()
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

    **A parte D nao e a afirmacao deste trabalho.** D pergunta se cada dimensao
    isolada carrega um fator so, e nada na perda pede isso: o contrastivo empurra
    a configuracao para `z_e` inteiro, nao para uma coordenada. Um D baixo aqui e
    o esperado, e reporta-lo sem dizer isso seria transformar uma escolha de
    desenho em fracasso medido.

    O que **e** a afirmacao esta em `block_mass`: a fracao da importancia de cada
    fator que cai em `z_e` contra `z_c`. E o desemaranhamento em blocos, que e o
    que a arquitetura promete e o que a fase 2 vai consumir na troca de codigos.
    """
    matrix, informativeness = importance_matrix(codes, frame, factors, seed, trees)
    factors = tuple(factors)

    # D por dimensao: quanto a importancia daquela dimensao se concentra num
    # fator so, ponderada por quanta importancia total a dimensao carrega.
    per_latent = np.array([1.0 - _entropy(row, len(factors)) for row in matrix])
    mass = matrix.sum(axis=1)
    disentanglement = float((per_latent * mass).sum() / mass.sum()) if mass.sum() else 0.0
    # C por fator: quanto o fator se concentra em poucas dimensoes.
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
    sabem de cada fator, normalizada pela entropia do fator.

    Nao depende de classificador -- e a diferenca util em relacao ao DCI, que
    depende. Se os dois discordarem, e a floresta que esta opinando.

    Vale a mesma ressalva do D: o MIG mede alinhamento **por eixo**. Aqui ele
    entra como descricao, nao como criterio.
    """
    from sklearn.metrics import mutual_info_score

    codes = np.asarray(codes, dtype=np.float64)
    if len(codes) != len(frame):
        raise ValueError(f"{len(codes)} codigos e {len(frame)} linhas")

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
        if factor not in frame.columns:
            raise KeyError(f"fator ausente no sidecar: {factor!r}")
        labels = frame[factor].astype("category").cat.codes.to_numpy()
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
    """DCI e MIG sobre `[z_e | z_c]` de uma execucao ja treinada.

    Sobre os **dois** blocos concatenados, e nao so sobre `z_e`: as metricas
    perguntam onde cada fator esta, e uma pergunta dessas nao se responde olhando
    metade do codigo. Onde o bloco comeca fica registrado em `blocks`.
    """
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.train import embed_blocks, split_frames

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    store = FeatureStore(dataset_root, frame, feature)
    z_e, z_c = embed_blocks(model, store, PixelStandardizer.load(run_dir / "standardizer.npz"))
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

    results_dir = Path(results_dir)
    wanted = list(techniques) if techniques else list(STUDY_ORDER)
    rows: List[Dict[str, object]] = []
    for name in wanted:
        run_dir = results_dir / name
        if not (run_dir / "run.json").exists():
            continue
        report = structure_run(run_dir, dataset_root, split=split, factors=factors,
                               seed=seed, trees=trees)
        dci = report["dci"]  # type: ignore[index]
        mig = report["mig"]  # type: ignore[index]
        dims = report["dims"]  # type: ignore[index]
        # A referencia da massa nao e 50%: os blocos tem tamanhos diferentes (32
        # contra 64), entao um codigo que nao separa nada espalha a importancia
        # na proporcao das dimensoes. Medido no encoder nao treinado, a massa em
        # `z_e` fica entre 0,33 e 0,42 -- exatamente em cima desta linha.
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
    if not rows:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")
    return pd.DataFrame(rows)


# --- a barra que decide se duas tecnicas sao diferentes ------------------------
#: As 5.600 consultas do split nao sao 5.600 amostras independentes: sao 20
#: conteudos x 40 configuracoes x 7 implementacoes, e o conteudo e o fator que
#: domina a dificuldade (etapa 4: e ele que derruba o B0, nao a implementacao).
#: Um IC por linha assume independencia que nao existe e sai **3,5x estreito** --
#: foi o que transformou "+1,25 ponto" numa diferenca aparente entre tecnicas que
#: o IC agrupado nao sustenta. Reamostrar **conteudos inteiros** e o conserto.
BOOTSTRAP_REPS = 4000

#: Chave de uma consulta. O nome do arquivo **se repete entre implementacoes** --
#: e o mesmo conteudo e a mesma configuracao renderizados por cada arm -- entao
#: juntar duas execucoes so por `file_name` multiplica as linhas por 7 em
#: silencio. O par e a chave.
QUERY_KEY = ("file_name", "query_arm")


def read_predictions(run_dir: Path, axis: str = "drive_level") -> pd.DataFrame:
    """`predictions.csv` de uma execucao, com a coluna de acerto ja derivada.

    O eixo e um parametro porque os dois nao se comportam igual: a dispersao
    entre execucoes e de ~1 ponto no drive e de ~5 no tom, entao uma diferenca
    lida no drive nao autoriza a mesma leitura no tom.
    """
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

    Pareada linha a linha (as duas execucoes respondem exatamente as mesmas
    consultas) e reamostrada por `cluster`, nao por linha.
    """
    junto = a.join(b["acerto"].rename("acerto_b"), how="inner")
    if len(junto) != len(a) or len(junto) != len(b):
        raise ValueError(
            f"as duas execucoes nao respondem as mesmas consultas: "
            f"{len(a)} e {len(b)} linhas dao {len(junto)} pareadas"
        )
    grupos = [grupo for _, grupo in junto.groupby(cluster)]
    rng = np.random.default_rng(seed)
    observado = float(junto["acerto"].mean() - junto["acerto_b"].mean()) * 100
    amostras = np.empty(reps)
    for indice in range(reps):
        escolha = rng.integers(0, len(grupos), len(grupos))
        bloco = pd.concat([grupos[posicao] for posicao in escolha])
        amostras[indice] = (bloco["acerto"].mean() - bloco["acerto_b"].mean()) * 100
    baixo, alto = (float(valor) for valor in np.percentile(amostras, [2.5, 97.5]))
    return {
        "diferenca_pontos": observado,
        "ic_baixo": baixo,
        "ic_alto": alto,
        "grupos": len(grupos),
        "n": int(len(junto)),
        # A leitura que interessa: um IC que cruza zero nao sustenta um ranking.
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
    """Execucoes com `predictions.csv` sob o diretorio, um nivel de subpasta.

    As variantes de peso moram numa subpasta justamente para nao entrarem no
    estudo pre-declarado; esta funcao e o unico lugar que as junta de volta, e
    so para efeito de comparacao.
    """
    results_dir = Path(results_dir)
    encontradas: Dict[str, Path] = {}
    for base in [results_dir, *(results_dir / nome for nome in extra)]:
        if not base.is_dir():
            continue
        for pasta in sorted(base.iterdir()):
            # As pastas de `extra` sao varridas como base, nunca como execucao:
            # sem isto a descida de um nivel abaixo transformaria a propria
            # `pesos/` numa "execucao" quando ela tivesse uma variante so.
            if not pasta.is_dir() or pasta.name in encontradas or pasta.name in extra:
                continue
            if (pasta / "predictions.csv").exists():
                encontradas[pasta.name] = pasta
                continue
            # Uma varredura escreve `<recorte>/<tecnica>/`, porque `--output-dir`
            # e a base e a tecnica vira subpasta. O nome util e o do recorte -- a
            # tecnica e a mesma nas quatro execucoes e nomearia todas igual.
            dentro = [filho for filho in sorted(pasta.iterdir())
                      if filho.is_dir() and (filho / "predictions.csv").exists()]
            if len(dentro) == 1:
                encontradas[pasta.name] = dentro[0]
    return encontradas
