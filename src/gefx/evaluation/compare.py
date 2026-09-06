"""Comparacao entre features.

Fica um nivel acima do `runner`: espera `<results-base>/<FEATURE>/chain_metrics.csv`,
que e o layout produzido pela varredura completa de features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from gefx.evaluation.metrics import standard_error
from gefx.evaluation.plots import grouped_bars


def _chain_metrics_path(feature_dir: Path) -> Path:
    return feature_dir / "chain_metrics.csv"


def _load_feature_metrics(feature_dir: Path) -> pd.DataFrame:
    path = _chain_metrics_path(feature_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty metrics file: {path}")
    for column in ("mae", "chain_key"):
        if column not in frame.columns:
            raise ValueError(f"Missing '{column}' column in {path}")
    return frame


def build_feature_comparison(results_base: Path) -> pd.DataFrame:
    feature_dirs = sorted(
        path for path in Path(results_base).iterdir() if path.is_dir() and _chain_metrics_path(path).exists()
    )
    if not feature_dirs:
        raise RuntimeError(f"No feature runs found under {results_base}")

    rows = []
    for feature_dir in feature_dirs:
        frame = _load_feature_metrics(feature_dir)
        best = frame.loc[frame["mae"].idxmin()]
        rows.append(
            {
                "feature": feature_dir.name,
                "n_chains": int(len(frame)),
                "mean_mae": float(frame["mae"].mean()),
                "mae_sem": standard_error(frame["mae"]),
                "min_mae": float(frame["mae"].min()),
                "best_chain_key": str(best["chain_key"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["mean_mae", "min_mae", "feature"])
        .reset_index(drop=True)
    )


def compare_features(
    results_base: str | Path,
    output_csv: Optional[str | Path] = None,
    output_plot: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    base = Path(results_base).resolve()
    comparison = build_feature_comparison(base)

    csv_path = Path(output_csv).resolve() if output_csv else base / "feature_comparison.csv"
    plot_path = Path(output_plot).resolve() if output_plot else base / "feature_comparison_mae.png"

    comparison.to_csv(csv_path, index=False)
    grouped_bars(
        labels=comparison["feature"].tolist(),
        left_values=comparison["mean_mae"],
        right_values=comparison["min_mae"],
        left_errors=comparison["mae_sem"],
        left_label="Mean MAE",
        right_label="Min MAE",
        title="Feature comparison by MAE",
        xlabel="Feature",
        ylabel="MAE",
        out_path=plot_path,
        width=0.38,
    )

    print(f"Tabela: {csv_path}")
    print(f"Grafico: {plot_path}")
    return csv_path, plot_path
