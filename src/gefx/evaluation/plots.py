"""Primitivas de grafico compartilhadas pela avaliacao e pela comparacao."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCATTER_COLOR = "#1f77b4"
ERROR_BAR_STYLE = {"elinewidth": 1, "ecolor": "#444444"}


def safe_filename(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def grouped_bars(
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    left_label: str,
    right_label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    left_errors: Optional[Sequence[float]] = None,
    right_errors: Optional[Sequence[float]] = None,
    width: float = 0.35,
) -> None:
    """Duas series de barras lado a lado, com barras de erro opcionais."""
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        positions - width / 2,
        left_values,
        width,
        label=left_label,
        yerr=left_errors,
        capsize=4 if left_errors is not None else 0,
        error_kw=ERROR_BAR_STYLE,
    )
    ax.bar(
        positions + width / 2,
        right_values,
        width,
        label=right_label,
        yerr=right_errors,
        capsize=4 if right_errors is not None else 0,
        error_kw=ERROR_BAR_STYLE,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parity_axis(axis, real: np.ndarray, estimated: np.ndarray, title: str) -> None:
    """Dispersao estimado-vs-real com a diagonal ideal e eixos quadrados iguais."""
    min_value = float(min(real.min(), estimated.min()))
    max_value = float(max(real.max(), estimated.max()))
    padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
    axis_min, axis_max = min_value - padding, max_value + padding

    axis.scatter(real, estimated, s=18, alpha=0.65, color=SCATTER_COLOR)
    axis.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1, label="Ideal")
    axis.set_title(title)
    axis.set_xlabel("Real value")
    axis.set_ylabel("Estimated value")
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")


def parity_grid(panels: Sequence[tuple], out_path: Path, suptitle: str, ncols_cap: int = 2) -> None:
    """Grade de paineis de paridade. Cada painel e (titulo, real, estimated)."""
    ncols = min(ncols_cap, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for axis, (title, real, estimated) in zip(axes.flat, panels):
        parity_axis(axis, real, estimated, title)
    for axis in axes.flat[len(panels) :]:
        fig.delaxes(axis)

    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_chain_baseline(chain_frame: pd.DataFrame, out_path: Path) -> None:
    grouped = chain_frame.sort_values("chain_key")
    grouped_bars(
        labels=grouped["chain_key"].tolist(),
        left_values=grouped["mae"],
        right_values=grouped["mse"],
        left_errors=grouped["mae_sem"] if "mae_sem" in grouped else None,
        right_errors=grouped["mse_sem"] if "mse_sem" in grouped else None,
        left_label="Mean MAE",
        right_label="Mean MSE",
        title="Fixed-order chain comparison",
        xlabel="Chain",
        ylabel="Error",
        out_path=out_path,
    )


def plot_estimated_vs_real(plot_frame: pd.DataFrame, out_dir: Path) -> None:
    """Um png por parametro, com um painel de paridade por cadeia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for param in sorted(plot_frame["parameter"].unique()):
        param_frame = plot_frame[plot_frame["parameter"] == param]
        panels = [
            (
                chain_key_value,
                param_frame.loc[param_frame["chain_key"] == chain_key_value, "real"].to_numpy(np.float64),
                param_frame.loc[param_frame["chain_key"] == chain_key_value, "estimated"].to_numpy(np.float64),
            )
            for chain_key_value in sorted(param_frame["chain_key"].unique())
        ]
        parity_grid(panels, out_dir / f"{safe_filename(param)}.png", f"Estimated vs real | {param}")


def plot_best_worst(plot_frame: pd.DataFrame, out_path: Path, top_n: int = 3) -> None:
    """Os `top_n` pares (cadeia, parametro) de menor e de maior MAE."""
    per_pair = (
        plot_frame.assign(abs_error=(plot_frame["estimated"] - plot_frame["real"]).abs())
        .groupby(["chain_key", "parameter"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    selected = pd.concat(
        [per_pair.nsmallest(top_n, "mae"), per_pair.nlargest(top_n, "mae")], ignore_index=True
    )
    if selected.empty:
        raise ValueError("Nenhuma predicao para o grafico de melhores/piores.")

    panels = []
    for row in selected.itertuples(index=False):
        pair = plot_frame[
            (plot_frame["chain_key"] == row.chain_key) & (plot_frame["parameter"] == row.parameter)
        ]
        panels.append(
            (
                f"{row.chain_key} | {row.parameter}\nMAE={row.mae:.4f}",
                pair["real"].to_numpy(np.float64),
                pair["estimated"].to_numpy(np.float64),
            )
        )

    parity_grid(panels, out_path, "Best 3 and worst 3 parameter estimates", ncols_cap=3)
