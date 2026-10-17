"""Sondas lineares sobre o `z_e` (e o `z_c`, se houver) de execucoes ja treinadas.

A evidencia de desemaranhamento do POC II: o que o codigo de efeito ainda deixa
ler de cada fator, contra o encoder nao treinado. Configuracao deve ficar mais
legivel; conteudo e implementacao, menos. No codigo de conteudo, o contrario
para conteudo e configuracao.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair)",
    "content_id": "conteudo (deve sair)",
    "drive_level": "configuracao (deve ficar)",
}

#: O mesmo, para o codigo de conteudo.
CONTENT_PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair)",
    "content_id": "conteudo (deve ficar)",
    "drive_level": "configuracao (deve sair)",
}

DEFAULT_FOLDS = 3

#: Os fatores que devem sair sao sondados dentro de cada nivel (ver `linear_probes`).
WITHIN = "drive_level"


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
    within: Optional[str] = WITHIN,
) -> Dict[str, Dict[str, float]]:
    """Acerto de uma regressao logistica por fator, com o acaso ao lado.

    Linear de proposito: pergunta se o fator esta legivel, nao se e recuperavel.
    A padronizacao fica dentro da validacao cruzada: com L2 de `C` fixo, sem ela
    a sonda mediria a escala do codigo junto com a legibilidade.

    Os fatores que devem sair sao sondados dentro de cada valor de `within` (o
    nivel de drive), e o acerto e a media entre eles. Sondados sobre tudo, o
    nivel domina a variancia de `z_e` e a direcao do conteudo muda de nivel
    para nivel: a sonda global dava 13% de conteudo onde, dentro de um nivel,
    ele ainda se lia a ~50%.
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
        scope = within if within in frame.columns and factor != within else None
        groups = (frame[scope].to_numpy() if scope else np.zeros(len(frame)))
        accuracy = float(np.mean([
            cross_val_score(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                codes[groups == group], labels[groups == group], cv=partition, n_jobs=folds,
            ).mean()
            for group in np.unique(groups)
        ]))
        out[factor] = {
            "within": scope or "",
            "accuracy": accuracy,
            "chance": chance,
            "classes": classes,
            "above_chance": (accuracy - chance) / (1.0 - chance),
        }
    return out


def embed_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent_v2"),
    split: str = "validacao",
    feature: str = "Spec",
    code: str = "z_e",
):
    """`(manifesto, frame, codigo)` do catalogo de `split`, com os arms e o
    padronizador da execucao; o codigo e a cabeca `code`."""
    from gefx.data.dataset import FeatureStore
    from gefx.data.grid import EVAL_SPLITS
    from gefx.evaluation.retrieval import embed
    from gefx.training.inference import open_run

    model, manifest, standardizer, frames = open_run(run_dir, dataset_root, code=code)
    frame = frames[EVAL_SPLITS[split][1]]
    codes = embed(model, FeatureStore(dataset_root, frame, feature), standardizer)
    return manifest, frame, codes


def probe_study(
    results_dir: Path,
    dataset_root: Path = Path("datasets/disent_v2"),
    runs: Optional[Sequence[str]] = None,
    split: str = "validacao",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por (execucao, codigo, fator), sobre o catalogo de `split`: o
    `z_e` de toda execucao, e o `z_c` das que tem cabeca de conteudo. Sem `runs`,
    toda subpasta com `run.json`."""
    results_dir = Path(results_dir)
    if runs is None:
        from gefx.training.inference import list_runs

        runs = list_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")

    rows: List[Dict[str, object]] = []
    for name in runs:
        manifest, frame, codes = embed_run(results_dir / name, dataset_root, split)
        heads = [("z_e", codes, PROBE_FACTORS)]
        if manifest["config"].get("architecture", {}).get("content_dim"):
            _, _, content = embed_run(results_dir / name, dataset_root, split, code="z_c")
            heads.append(("z_c", content, CONTENT_PROBE_FACTORS))
        for code, values, meanings in heads:
            for factor, numbers in linear_probes(values, frame, folds=folds,
                                                 seed=seed).items():
                rows.append({"run": name, "technique": manifest["config"]["technique"],
                             "code": code, "factor": factor, "meaning": meanings[factor],
                             "split": split, "n": int(len(frame)), **numbers})
    return pd.DataFrame(rows)
