"""Avaliacao de um run: consome so os arquivos por cadeia deixados pelo treino."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from gefx.evaluation.metrics import build_prediction_frame, evaluate_chain, find_chain_dirs
from gefx.evaluation.plots import plot_best_worst, plot_chain_baseline, plot_estimated_vs_real


def evaluate_run(results_root: str | Path) -> Dict[str, Path]:
    """Gera chain_metrics.csv, parameter_metrics.csv e os graficos do run."""
    root = Path(results_root).resolve()
    chain_dirs = find_chain_dirs(root)
    if not chain_dirs:
        raise RuntimeError(f"No chain outputs found under {root}")

    chain_frames, param_frames = zip(*(evaluate_chain(chain_dir) for chain_dir in chain_dirs))
    chain_summary = pd.concat(chain_frames, ignore_index=True)
    param_summary = pd.concat(param_frames, ignore_index=True)

    outputs = {
        "chain_metrics": root / "chain_metrics.csv",
        "parameter_metrics": root / "parameter_metrics.csv",
        "chain_plot": root / "chain_baseline_mae_mse.png",
        "parity_plots": root / "estimated_vs_real",
        "best_worst_plot": root / "estimated_vs_real_best_worst.png",
    }

    chain_summary.to_csv(outputs["chain_metrics"], index=False)
    param_summary.to_csv(outputs["parameter_metrics"], index=False)

    plot_chain_baseline(chain_summary, outputs["chain_plot"])
    plot_frame = build_prediction_frame(chain_dirs)
    plot_estimated_vs_real(plot_frame, outputs["parity_plots"])
    plot_best_worst(plot_frame, outputs["best_worst_plot"])

    for label, path in outputs.items():
        print(f"{label:<20} {path}")
    return outputs
