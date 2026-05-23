from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-order chain models.")
    parser.add_argument("--results-root", required=True, help="Folder containing per-chain model outputs.")
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


def evaluate_chain(chain_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = _load_predictions(chain_dir / "predictions.csv")
    metrics = _load_metrics(chain_dir / "metrics.json")
    chain_key_value = metrics.get("chain_key", chain_dir.name)

    params = _parameter_columns(pred_df)
    param_rows = []
    for param in params:
        y_true = pred_df[f"y_true_{param}"].to_numpy(dtype=np.float64)
        y_pred = pred_df[f"y_pred_{param}"].to_numpy(dtype=np.float64)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        mse = float(np.mean((y_pred - y_true) ** 2))
        param_rows.append(
            {
                "chain_key": chain_key_value,
                "parameter": param,
                "mae": mae,
                "mse": mse,
                "global_mae": float(metrics["mae"]),
                "global_mse": float(metrics["mse"]),
            }
        )
    effect_order = metrics.get("effect_order", [])
    chain_length = len(effect_order) if effect_order else int(metrics.get("chain_length", 0))
    chain_row = {
        "chain_key": chain_key_value,
        "chain_length": chain_length,
        "effect_order": json.dumps(effect_order),
        "feature": metrics.get("feature"),
        "n_train": int(metrics.get("n_train", 0)),
        "n_test": int(metrics.get("n_test", 0)),
        "output_dim": int(metrics.get("output_dim", 0)),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
    }
    return pd.DataFrame([chain_row]), pd.DataFrame(param_rows)


def plot_chain_baseline(chain_df: pd.DataFrame, out_path: Path) -> None:
    grouped = chain_df.sort_values("chain_key")

    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, grouped["mae"], width, label="Mean MAE")
    ax.bar(x + width / 2, grouped["mse"], width, label="Mean MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["chain_key"], rotation=30, ha="right")
    ax.set_xlabel("Chain")
    ax.set_ylabel("Error")
    ax.set_title("Fixed-order chain comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _safe_filename(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def build_prediction_df(results_root: Path, chain_dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for chain_dir in chain_dirs:
        pred_df = _load_predictions(chain_dir / "predictions.csv")
        metrics = _load_metrics(chain_dir / "metrics.json")
        chain_key_value = metrics.get("chain_key", chain_dir.name)
        params = _parameter_columns(pred_df)
        for param in params:
            frames.append(
                pd.DataFrame(
                    {
                        "chain_key": chain_key_value,
                        "parameter": param,
                        "real": pred_df[f"y_true_{param}"].to_numpy(dtype=np.float64),
                        "estimated": pred_df[f"y_pred_{param}"].to_numpy(dtype=np.float64),
                    }
                )
            )

    if not frames:
        raise ValueError(f"No prediction data found under {results_root}")

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["chain_key"] = plot_df["chain_key"].astype(str)
    return plot_df


def plot_estimated_vs_real(results_root: Path, chain_dirs: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = build_prediction_df(results_root, chain_dirs)
    parameters = sorted(plot_df["parameter"].unique())

    for param in parameters:
        param_df = plot_df[plot_df["parameter"] == param]
        chain_keys = sorted(param_df["chain_key"].unique())
        ncols = min(2, len(chain_keys))
        nrows = math.ceil(len(chain_keys) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

        for axis, chain_key_value in zip(axes.flat, chain_keys):
            chain_df = param_df[param_df["chain_key"] == chain_key_value]
            real = chain_df["real"].to_numpy(dtype=np.float64)
            estimated = chain_df["estimated"].to_numpy(dtype=np.float64)
            min_value = float(min(real.min(), estimated.min()))
            max_value = float(max(real.max(), estimated.max()))
            padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
            axis_min = min_value - padding
            axis_max = max_value + padding

            axis.scatter(real, estimated, s=18, alpha=0.65, color="#1f77b4")
            axis.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1, label="Ideal")
            axis.set_title(chain_key_value)
            axis.set_xlabel("Real value")
            axis.set_ylabel("Estimated value")
            axis.set_xlim(axis_min, axis_max)
            axis.set_ylim(axis_min, axis_max)
            axis.set_aspect("equal", adjustable="box")
            axis.legend(loc="best")

        for axis in axes.flat[len(chain_keys) :]:
            fig.delaxes(axis)

        fig.suptitle(f"Estimated vs real | {param}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_filename(param)}.png", bbox_inches="tight")
        plt.close(fig)


def plot_best_worst(results_root: Path, chain_dirs: List[Path], out_path: Path, top_n: int = 3) -> None:
    plot_df = build_prediction_df(results_root, chain_dirs)
    metrics_df = (
        plot_df.assign(abs_error=(plot_df["estimated"] - plot_df["real"]).abs())
        .groupby(["chain_key", "parameter"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )

    best = metrics_df.nsmallest(top_n, "mae")
    worst = metrics_df.nlargest(top_n, "mae")
    selected = pd.concat([best, worst], ignore_index=True)
    if selected.empty:
        raise ValueError(f"No prediction data found under {results_root}")

    ncols = min(3, len(selected))
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for axis, row in zip(axes.flat, selected.itertuples(index=False)):
        chain_df = plot_df[(plot_df["chain_key"] == row.chain_key) & (plot_df["parameter"] == row.parameter)]
        real = chain_df["real"].to_numpy(dtype=np.float64)
        estimated = chain_df["estimated"].to_numpy(dtype=np.float64)
        min_value = float(min(real.min(), estimated.min()))
        max_value = float(max(real.max(), estimated.max()))
        padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
        axis_min = min_value - padding
        axis_max = max_value + padding

        axis.scatter(real, estimated, s=18, alpha=0.65, color="#1f77b4")
        axis.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1, label="Ideal")
        axis.set_title(f"{row.chain_key} | {row.parameter}\nMAE={row.mae:.4f}")
        axis.set_xlabel("Real value")
        axis.set_ylabel("Estimated value")
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")

    for axis in axes.flat[len(selected) :]:
        fig.delaxes(axis)

    fig.suptitle("Best 3 and worst 3 parameter estimates", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()

    chain_dirs = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "metrics.json").exists() and (path / "predictions.csv").exists()
    )

    if not chain_dirs:
        raise RuntimeError(f"No chain outputs found under {results_root}")

    chain_rows = []
    param_rows = []
    for chain_dir in chain_dirs:
        chain_df, params_df = evaluate_chain(chain_dir)
        chain_rows.append(chain_df)
        param_rows.append(params_df)

    chain_summary_df = pd.concat(chain_rows, ignore_index=True)
    param_summary_df = pd.concat(param_rows, ignore_index=True)

    chain_summary_csv = results_root / "chain_metrics.csv"
    chain_summary_df.to_csv(chain_summary_csv, index=False)

    param_summary_csv = results_root / "parameter_metrics.csv"
    param_summary_df.to_csv(param_summary_csv, index=False)

    plot_path = results_root / "chain_baseline_mae_mse.png"
    plot_chain_baseline(chain_summary_df, plot_path)

    parity_plot_dir = results_root / "estimated_vs_real"
    plot_estimated_vs_real(results_root, chain_dirs, parity_plot_dir)

    best_worst_plot_path = results_root / "estimated_vs_real_best_worst.png"
    plot_best_worst(results_root, chain_dirs, best_worst_plot_path)

    print(f"Saved chain table:       {chain_summary_csv}")
    print(f"Saved parameter table:   {param_summary_csv}")
    print(f"Saved chain plot:        {plot_path}")
    print(f"Saved parity plots:      {parity_plot_dir}")
    print(f"Saved best/worst plot:   {best_worst_plot_path}")


if __name__ == "__main__":
    main()
