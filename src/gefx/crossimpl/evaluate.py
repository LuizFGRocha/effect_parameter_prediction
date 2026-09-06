"""Avaliacao cross-implementation: os modelos generalizam para outros plugins?

Reporta tres numeros por parametro:

- `mae`: erro direto do modelo treinado, sem retreino.
- `mae_cal`: MAE depois de uma calibracao isotonica ajustada nos proprios dados
  do teste. E otimista de proposito — serve como **teto** do que qualquer
  calibracao monotonica compraria, nao como resultado.
- `spearman`: correlacao de postos, unica leitura que faz sentido nos parametros
  sem equivalencia fisica (`exato=nao`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from gefx.data.dataset import load_chain_dataset
from gefx.effects.catalog import EFFECT_PARAMETER_RANGES, effect_predictable_params
from gefx.effects.registry import REFERENCE_ARM, REGISTRY, is_exact
from gefx.training.inference import clear_session, load_trained_chain


@dataclass
class EvalOptions:
    feature: str = "Spec"
    models_root: str = "results/second_main_run"
    output_root: str = "datasets/cross_impl"


def evaluate(options: EvalOptions) -> pd.DataFrame:
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    root = Path(options.output_root)
    arms = sorted(path.name for path in root.iterdir() if path.is_dir())
    rows: List[dict] = []

    for effect in sorted(EFFECT_PARAMETER_RANGES):
        model_dir = Path(options.models_root) / options.feature / effect
        if not model_dir.exists():
            continue

        trained = load_trained_chain(model_dir)
        names = [param["name"] for param in effect_predictable_params(effect)]

        for arm in arms:
            if arm != REFERENCE_ARM and REGISTRY.get(effect, {}).get(arm) is None:
                continue
            if not (root / arm / effect).exists():
                continue

            features, targets, _ = load_chain_dataset(
                root / arm, effect, feature_name=options.feature
            )
            predictions = trained.predict(features)

            for index, name in enumerate(names):
                y_true, y_pred = targets[:, index], predictions[:, index]
                calibrated = IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)
                rows.append(
                    {
                        "effect": effect,
                        "arm": arm,
                        "parameter": name,
                        "exato": is_exact(effect, arm, name),
                        "n": len(y_true),
                        "mae": float(np.mean(np.abs(y_pred - y_true))),
                        "mae_cal": float(np.mean(np.abs(calibrated.predict(y_pred) - y_true))),
                        "spearman": float(spearmanr(y_true, y_pred).statistic),
                    }
                )

        clear_session()

    frame = pd.DataFrame(rows)
    out_path = root / f"cross_impl_{options.feature}.csv"
    frame.to_csv(out_path, index=False)
    print_report(frame)
    print(f"\nSalvo em {out_path}")
    return frame


def print_report(frame: pd.DataFrame) -> None:
    """Tabela com o delta de MAE de cada braco contra a referencia in-domain."""
    reference = frame[frame.arm == REFERENCE_ARM].set_index(["effect", "parameter"])
    print(
        f"\n{'efeito':<16} {'parametro':<24} {'braco':<12} {'exato':<6} "
        f"{'MAE':>7} {'dMAE':>7} {'MAEcal':>7} {'rho':>6}"
    )
    print("-" * 96)
    for (effect, param), group in frame.groupby(["effect", "parameter"], sort=True):
        for row in group.sort_values("arm").itertuples(index=False):
            base = (
                reference.loc[(effect, param), "mae"]
                if (effect, param) in reference.index
                else np.nan
            )
            delta = "" if row.arm == REFERENCE_ARM else f"{row.mae - base:+.3f}"
            print(
                f"{effect:<16} {param:<24} {row.arm:<12} {'sim' if row.exato else 'nao':<6} "
                f"{row.mae:>7.3f} {delta:>7} {row.mae_cal:>7.3f} {row.spearman:>6.2f}"
            )
