"""Metricas agregadas a partir dos arquivos que o treino deixou em disco.

MAE/MSE por parametro sao recomputados de `predictions.csv`, e nao lidos do
agregado em `metrics.json` — assim o erro padrao acompanha cada numero.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def standard_error(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))


def load_predictions(chain_dir: Path) -> pd.DataFrame:
    path = Path(chain_dir) / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return pd.read_csv(path)


def load_metrics(chain_dir: Path) -> Dict:
    path = Path(chain_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_columns(frame: pd.DataFrame, legacy: bool = False) -> List[str]:
    """Nomes dos parametros na ordem em que as colunas foram gravadas.

    Essa e a ordem do vetor alvo, a mesma de `metrics.json["parameter_names"]`.
    `legacy=True` volta a ordenar alfabeticamente, como nas tabelas ja geradas.
    """
    names = [col[len("y_true_") :] for col in frame.columns if col.startswith("y_true_")]
    return sorted(names) if legacy else names


def find_chain_dirs(results_root: Path) -> List[Path]:
    """Subpastas de `results_root` que tem saida de treino completa."""
    return sorted(
        path
        for path in Path(results_root).iterdir()
        if path.is_dir()
        and (path / "metrics.json").exists()
        and (path / "predictions.csv").exists()
    )


def evaluate_chain(chain_dir: Path, legacy: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (linha da cadeia, linhas por parametro) para uma cadeia."""
    predictions = load_predictions(chain_dir)
    metrics = load_metrics(chain_dir)
    chain_key_value = metrics.get("chain_key", Path(chain_dir).name)

    param_rows = []
    abs_errors: List[np.ndarray] = []
    squared_errors: List[np.ndarray] = []
    for param in parameter_columns(predictions, legacy=legacy):
        y_true = predictions[f"y_true_{param}"].to_numpy(dtype=np.float64)
        y_pred = predictions[f"y_pred_{param}"].to_numpy(dtype=np.float64)
        abs_error = np.abs(y_pred - y_true)
        squared_error = (y_pred - y_true) ** 2
        abs_errors.append(abs_error)
        squared_errors.append(squared_error)
        param_rows.append(
            {
                "chain_key": chain_key_value,
                "parameter": param,
                "mae": float(np.mean(abs_error)),
                "mse": float(np.mean(squared_error)),
                "mae_sem": standard_error(abs_error),
                "mse_sem": standard_error(squared_error),
                "global_mae": float(metrics["mae"]),
                "global_mse": float(metrics["mse"]),
            }
        )

    effect_order = metrics.get("effect_order", [])
    chain_row = {
        "chain_key": chain_key_value,
        "chain_length": len(effect_order) if effect_order else int(metrics.get("chain_length", 0)),
        "effect_order": json.dumps(effect_order),
        "feature": metrics.get("feature"),
        "n_train": int(metrics.get("n_train", 0)),
        "n_test": int(metrics.get("n_test", 0)),
        "output_dim": int(metrics.get("output_dim", 0)),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "mae_sem": standard_error(np.concatenate(abs_errors)) if abs_errors else 0.0,
        "mse_sem": standard_error(np.concatenate(squared_errors)) if squared_errors else 0.0,
    }
    return pd.DataFrame([chain_row]), pd.DataFrame(param_rows)


def build_prediction_frame(chain_dirs: List[Path], legacy: bool = False) -> pd.DataFrame:
    """Junta todas as predicoes em formato longo: chain_key, parameter, real, estimated."""
    frames = []
    for chain_dir in chain_dirs:
        predictions = load_predictions(chain_dir)
        metrics = load_metrics(chain_dir)
        chain_key_value = metrics.get("chain_key", Path(chain_dir).name)
        for param in parameter_columns(predictions, legacy=legacy):
            frames.append(
                pd.DataFrame(
                    {
                        "chain_key": chain_key_value,
                        "parameter": param,
                        "real": predictions[f"y_true_{param}"].to_numpy(dtype=np.float64),
                        "estimated": predictions[f"y_pred_{param}"].to_numpy(dtype=np.float64),
                    }
                )
            )
    if not frames:
        raise ValueError("Nenhuma predicao encontrada.")

    plot_frame = pd.concat(frames, ignore_index=True)
    plot_frame["chain_key"] = plot_frame["chain_key"].astype(str)
    return plot_frame
