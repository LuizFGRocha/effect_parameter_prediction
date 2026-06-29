from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import standard_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature runs using MAE statistics.")
    parser.add_argument(
        "--results-base",
        required=True,
        help="Folder containing one subfolder per feature, each with chain_metrics.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the aggregated CSV report. Defaults to <results-base>/feature_comparison.csv.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Path for the MAE comparison plot. Defaults to <results-base>/feature_comparison_mae.png.",
    )
    return parser.parse_args()


def _feature_metrics_path(feature_dir: Path) -> Path:
    return feature_dir / "chain_metrics.csv"


def _load_feature_metrics(feature_dir: Path) -> pd.DataFrame:
    metrics_path = _feature_metrics_path(feature_dir)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError(f"Empty metrics file: {metrics_path}")
    if "mae" not in metrics_df.columns:
        raise ValueError(f"Missing 'mae' column in {metrics_path}")
    if "chain_key" not in metrics_df.columns:
        raise ValueError(f"Missing 'chain_key' column in {metrics_path}")
    return metrics_df


def build_feature_comparison(results_base: Path) -> pd.DataFrame:
    feature_dirs = sorted(
        path
        for path in results_base.iterdir()
        if path.is_dir() and _feature_metrics_path(path).exists()
    )

    if not feature_dirs:
        raise RuntimeError(f"No feature runs found under {results_base}")

    rows = []
    for feature_dir in feature_dirs:
        metrics_df = _load_feature_metrics(feature_dir)
        best_row = metrics_df.loc[metrics_df["mae"].idxmin()]

        rows.append(
            {
                "feature": feature_dir.name,
                "n_chains": int(len(metrics_df)),
                "mean_mae": float(metrics_df["mae"].mean()),
                "mae_sem": standard_error(metrics_df["mae"]),
                "min_mae": float(metrics_df["mae"].min()),
                "best_chain_key": str(best_row["chain_key"]),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(["mean_mae", "min_mae", "feature"]).reset_index(drop=True)
    return comparison_df


def plot_feature_comparison(comparison_df: pd.DataFrame, out_path: Path) -> None:
    ordered_df = comparison_df.sort_values(["mean_mae", "min_mae", "feature"]).reset_index(drop=True)
    x_positions = list(range(len(ordered_df)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [index - width / 2 for index in x_positions],
        ordered_df["mean_mae"],
        width,
        label="Mean MAE",
        yerr=ordered_df["mae_sem"],
        capsize=4,
        error_kw={"elinewidth": 1, "ecolor": "#444444"},
    )
    ax.bar([index + width / 2 for index in x_positions], ordered_df["min_mae"], width, label="Min MAE")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_df["feature"], rotation=30, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("MAE")
    ax.set_title("Feature comparison by MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_base = Path(args.results_base).resolve()

    comparison_df = build_feature_comparison(results_base)

    output_csv = Path(args.output_csv).resolve() if args.output_csv else results_base / "feature_comparison.csv"
    output_plot = Path(args.output_plot).resolve() if args.output_plot else results_base / "feature_comparison_mae.png"

    comparison_df.to_csv(output_csv, index=False)
    plot_feature_comparison(comparison_df, output_plot)

    print(f"Saved feature comparison table: {output_csv}")
    print(f"Saved feature comparison plot:  {output_plot}")


if __name__ == "__main__":
    main()