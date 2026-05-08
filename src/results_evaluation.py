"""Evaluate fixed-chain models and compare L1/L2/L3 baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Paper fixed-order chain models.")
    parser.add_argument("--results-root", required=True, help="Folder containing L1/L2/L3 model outputs.")
    return parser.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return pd.read_csv(path)


def _load_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parameter_columns(df: pd.DataFrame) -> List[str]:
    return sorted(col.replace("y_true_", "") for col in df.columns if col.startswith("y_true_"))


def evaluate_length(length_dir: Path, chain_length: int) -> pd.DataFrame:
    pred_df = _load_predictions(length_dir / "predictions.csv")
    metrics = _load_metrics(length_dir / "metrics.json")

    params = _parameter_columns(pred_df)
    rows = []
    for param in params:
        y_true = pred_df[f"y_true_{param}"].to_numpy(dtype=np.float64)
        y_pred = pred_df[f"y_pred_{param}"].to_numpy(dtype=np.float64)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        mse = float(np.mean((y_pred - y_true) ** 2))
        rows.append(
            {
                "chain_length": chain_length,
                "parameter": param,
                "mae": mae,
                "mse": mse,
                "global_mae": float(metrics["mae"]),
                "global_mse": float(metrics["mse"]),
            }
        )
    return pd.DataFrame(rows)


def plot_length_baseline(summary_df: pd.DataFrame, out_path: Path) -> None:
    grouped = summary_df.groupby("chain_length").agg({"mae": "mean", "mse": "mean"}).reset_index()

    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, grouped["mae"], width, label="Mean MAE")
    ax.bar(x + width / 2, grouped["mse"], width, label="Mean MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{int(v)}" for v in grouped["chain_length"]])
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Error")
    ax.set_title("Fixed-order chain baseline comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_estimated_vs_real(results_root: Path, out_path: Path) -> None:
    frames = []
    for chain_length in (1, 2, 3):
        pred_df = _load_predictions(results_root / f"L{chain_length}" / "predictions.csv")
        params = _parameter_columns(pred_df)
        for param in params:
            frames.append(
                pd.DataFrame(
                    {
                        "chain_length": chain_length,
                        "parameter": param,
                        "real": pred_df[f"y_true_{param}"].to_numpy(dtype=np.float64),
                        "estimated": pred_df[f"y_pred_{param}"].to_numpy(dtype=np.float64),
                    }
                )
            )

    if not frames:
        raise ValueError(f"No prediction data found under {results_root}")

    plot_df = pd.concat(frames, ignore_index=True)
    parameters = sorted(plot_df["parameter"].unique())
    ncols = min(2, len(parameters))
    nrows = math.ceil(len(parameters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    min_value = float(min(plot_df["real"].min(), plot_df["estimated"].min()))
    max_value = float(max(plot_df["real"].max(), plot_df["estimated"].max()))
    padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
    axis_min = min_value - padding
    axis_max = max_value + padding

    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for axis, param in zip(axes.flat, parameters):
        param_df = plot_df[plot_df["parameter"] == param]
        for chain_length, chain_df in param_df.groupby("chain_length"):
            axis.scatter(
                chain_df["real"],
                chain_df["estimated"],
                s=18,
                alpha=0.65,
                color=colors.get(int(chain_length), "#1f77b4"),
                label=f"L{int(chain_length)}",
            )

        axis.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1, label="Ideal")
        axis.set_title(param)
        axis.set_xlabel("Real value")
        axis.set_ylabel("Estimated value")
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")

    for axis in axes.flat[len(parameters) :]:
        fig.delaxes(axis)

    fig.suptitle("Estimated vs real parameter values", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()

    all_rows = []
    for chain_length in (1, 2, 3):
        length_dir = results_root / f"L{chain_length}"
        df = evaluate_length(length_dir, chain_length)
        all_rows.append(df)

    summary_df = pd.concat(all_rows, ignore_index=True)
    summary_csv = results_root / "comparison_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_path = results_root / "length_baseline_mae_mse.png"
    plot_length_baseline(summary_df, plot_path)

    parity_plot_path = results_root / "estimated_vs_real.png"
    plot_estimated_vs_real(results_root, parity_plot_path)

    print(f"Saved comparison table: {summary_csv}")
    print(f"Saved baseline plot:     {plot_path}")
    print(f"Saved parity plot:       {parity_plot_path}")


if __name__ == "__main__":
    main()
