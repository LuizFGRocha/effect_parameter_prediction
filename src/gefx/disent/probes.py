"""Sondas lineares sobre o `z_e` de execucoes ja treinadas.

A evidencia de desemaranhamento do POC II: o que o codigo de efeito ainda deixa
ler de cada fator, contra o encoder nao treinado. Configuracao deve ficar mais
legivel; conteudo e implementacao, menos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair)",
    "content_id": "conteudo (deve sair)",
    "drive_level": "configuracao (deve ficar)",
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


def load_run(run_dir: Path):
    """`(modelo, manifesto)` de uma execucao gravada."""
    from gefx.disent.model import WEIGHTS_FILE, EncoderConfig, build_model

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = build_model(EncoderConfig.from_dict(manifest["config"]["encoder"]),
                        manifest["config"]["technique"])
    model.load_weights(run_dir / WEIGHTS_FILE)
    return model, manifest


def embed_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent_v2"),
    split: str = "catalog",
    feature: str = "Spec",
):
    """`(manifesto, frame, z_e)` de uma execucao, com os arms e o padronizador dela."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.sidecar import split_frames
    from gefx.disent.train import embed

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    codes = embed(model, FeatureStore(dataset_root, frame, feature), standardizer)
    return manifest, frame, codes


def probe_study(
    results_dir: Path,
    dataset_root: Path = Path("datasets/disent_v2"),
    runs: Optional[Sequence[str]] = None,
    split: str = "catalog",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por (execucao, fator). Sem `runs`, toda subpasta com `run.json`."""
    results_dir = Path(results_dir)
    if runs is None:
        runs = sorted(path.parent.name for path in results_dir.glob("*/run.json"))
    if not runs:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")

    rows: List[Dict[str, object]] = []
    for name in runs:
        manifest, frame, codes = embed_run(results_dir / name, dataset_root, split)
        for factor, numbers in linear_probes(codes, frame, folds=folds, seed=seed).items():
            rows.append({"run": name, "technique": manifest["config"]["technique"],
                         "factor": factor, "meaning": PROBE_FACTORS[factor],
                         "split": split, "n": int(len(frame)), **numbers})
    return pd.DataFrame(rows)
