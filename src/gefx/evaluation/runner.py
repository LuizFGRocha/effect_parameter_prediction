"""Avaliacao de um run: consome so os arquivos que o treino deixou.

- POC I (`evaluate_run`): os arquivos por cadeia, em tabelas e graficos.
- POC II (`compare`, `evaluate_runs`): a escada (baselines e execucoes do
  encoder) numa particao, e a passada final sobre o teste.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from gefx.config import git_revision
from gefx.data.grid import EVAL_SPLITS, split_file
from gefx.data.dataset import FeatureStore
from gefx.evaluation.retrieval import evaluate, metrics_file, predictions_file

from gefx.evaluation.metrics import build_prediction_frame, evaluate_chain, find_chain_dirs
from gefx.evaluation.plots import plot_best_worst, plot_chain_baseline, plot_estimated_vs_real


def evaluate_run(results_root: str | Path, legacy: bool = False) -> Dict[str, Path]:
    """Gera chain_metrics.csv, parameter_metrics.csv e os graficos do run.

    `legacy=True` ordena os parametros alfabeticamente, como nas tabelas ja
    commitadas, em vez de seguir a ordem do vetor alvo.
    """
    root = Path(results_root).resolve()
    chain_dirs = find_chain_dirs(root)
    if not chain_dirs:
        raise RuntimeError(f"No chain outputs found under {root}")

    chain_frames, param_frames = zip(
        *(evaluate_chain(chain_dir, legacy=legacy) for chain_dir in chain_dirs)
    )
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
    plot_frame = build_prediction_frame(chain_dirs, legacy=legacy)
    plot_estimated_vs_real(plot_frame, outputs["parity_plots"])
    plot_best_worst(plot_frame, outputs["best_worst_plot"])

    for label, path in outputs.items():
        print(f"{label:<20} {path}")
    return outputs


# --- POC II -------------------------------------------------------------------
ENCODER_RESULTS = Path("results/disent/v2/validacao/encoder")

#: Arquivos de `gefx disent retrieve`, na pasta acima das execucoes: `<nome>_<particao>.json`.
BASELINES: Dict[str, str] = {"B0": "b0", "B1": "b1"}


def load_baselines(baselines_dir: Path, split: str = "validacao") -> List[Dict[str, object]]:
    """Uma linha por baseline gravado, mais o acaso de drive tirado do alfabeto dele."""
    rows: List[Dict[str, object]] = []
    for name, stem in BASELINES.items():
        path = Path(baselines_dir) / split_file(stem, split, "json")
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["overall"]
        if not rows:
            rows.append({"run": "chance", "technique": "baseline",
                         "drive_exact": 1.0 / metrics["alphabet"]["drive_level"]})
        rows.append({"run": name, "technique": "baseline",
                     "drive_exact": float(overall["drive_level"]["exact"]),
                     "mae_db": float(overall["mae_db"])})
    return rows


def compare(
    results_dir: Path = ENCODER_RESULTS,
    runs: Optional[Sequence[str]] = None,
    baselines_dir: Optional[Path] = None,
    split: str = "validacao",
) -> pd.DataFrame:
    """A escada numa particao: baselines por cima, depois uma linha por execucao.

    Sem `runs`, pega toda subpasta com `run.json`, em ordem alfabetica; entram as
    que ja foram avaliadas em `split`. Os baselines saem de `baselines_dir`, por
    padrao a pasta acima de `results_dir`.
    """
    from gefx.training.encoder import summarize
    from gefx.training.inference import list_runs

    results_dir = Path(results_dir)
    if runs is None:
        runs = list_runs(results_dir)
    rows = load_baselines(baselines_dir or results_dir.parent, split)
    for name in runs:
        manifest_path = results_dir / name / "run.json"
        metrics_path = results_dir / name / metrics_file(split)
        if not (manifest_path.exists() and metrics_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "run": name,
            "technique": manifest["config"]["technique"],
            "seed": manifest["config"]["seed"],
            **summarize(metrics),
            "steps": manifest["steps_executed"],
            "best_step": manifest["best_step"],
        })
    return pd.DataFrame(rows, columns=None if rows else ["run", "technique"])


def evaluate_runs(
    results_dir: Path = ENCODER_RESULTS,
    runs: Optional[Sequence[str]] = None,
    split: str = "teste",
    dataset_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Avalia execucoes gravadas numa particao, com os pesos que o treino deixou.

    E a passada final sobre o teste: grava `metrics_<split>.json` e
    `predictions_<split>.csv` em cada execucao e devolve a escada dessa particao.
    Le cada execucao com os arms e a validacao com que ela treinou.
    """
    from gefx.training.inference import list_runs, open_run

    results_dir = Path(results_dir)
    if runs is None:
        runs = list_runs(results_dir)
    for name in runs:
        run_dir = results_dir / name
        model, manifest, standardizer, frames = open_run(run_dir, dataset_root)
        run_config = manifest["config"]
        root = Path(dataset_root or run_config["dataset_root"])
        stores = {key: FeatureStore(root, frames[key], run_config["feature"])
                  for key in EVAL_SPLITS[split]}
        predictions, metrics = evaluate(model, frames, stores, standardizer,
                                        run_config["embed_batch"], split=split)
        predictions.to_csv(run_dir / predictions_file(split), index=False)
        metrics["git_revision"] = git_revision()
        (run_dir / metrics_file(split)).write_text(json.dumps(metrics, indent=2),
                                                  encoding="utf-8")
    return compare(results_dir, runs, split=split)
