"""Treino de um regressor por cadeia.

Cada cadeia grava em `<results-root>/<chain_key>/`: `model.keras`,
`feature_scalers.pkl`, `history.json`, `metrics.json`, `predictions.csv` e
`split_indices.npz`. O run inteiro grava `all_metrics.json` e `run.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gefx.config import ExperimentConfig, set_global_seeds, write_run_manifest
from gefx.data.dataset import load_chain_dataset
from gefx.data.metadata import list_chain_keys, validate_sidecar_integrity
from gefx.effects.catalog import chain_key_to_effects, parameter_names_for_chain
from gefx.training.architecture import build_model
from gefx.training.inference import MODEL_FILENAME, SCALERS_FILENAME, clear_session
from gefx.training.scaling import fit_transform_split


def _predictions_frame(
    file_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    parameter_names: List[str],
) -> pd.DataFrame:
    columns: Dict[str, object] = {"file_name": [str(name) for name in file_names]}
    for index, name in enumerate(parameter_names):
        columns[f"y_true_{name}"] = y_true[:, index].astype(float)
        columns[f"y_pred_{name}"] = y_pred[:, index].astype(float)
    return pd.DataFrame(columns)


def train_one_chain(
    chain_key_value: str,
    config: ExperimentConfig,
) -> Dict[str, object]:
    """Treina, avalia e persiste uma cadeia. Devolve o dicionario de metricas."""
    train_config = config.train
    features, targets, file_names = load_chain_dataset(
        train_config.dataset_root,
        chain_key_value,
        feature_name=train_config.feature,
        force_rebuild_cache=train_config.rebuild_cache,
    )

    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=train_config.test_size,
        random_state=train_config.split_seed,
        shuffle=True,
    )

    x_train, x_test, scalers = fit_transform_split(features[train_idx], features[test_idx])
    y_train, y_test = targets[train_idx], targets[test_idx]

    model = build_model(x_train.shape[1:], y_train.shape[1], config.architecture)
    history = model.fit(
        x_train,
        y_train,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        verbose=2,
        validation_data=(x_test, y_test),
    )

    predictions = model.predict(x_test, verbose=0)
    mse = float(np.mean((predictions - y_test) ** 2))
    mae = float(np.mean(np.abs(predictions - y_test)))

    out_dir = Path(train_config.results_root) / chain_key_value
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / MODEL_FILENAME)
    joblib.dump(scalers, out_dir / SCALERS_FILENAME)
    np.savez(
        out_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        train_files=file_names[train_idx],
        test_files=file_names[test_idx],
    )
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8",
    )

    parameter_names = parameter_names_for_chain(chain_key_value)
    metrics = {
        "chain_key": chain_key_value,
        "chain_length": len(chain_key_to_effects(chain_key_value)),
        "effect_order": chain_key_to_effects(chain_key_value),
        "feature": train_config.feature,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "output_dim": int(targets.shape[1]),
        "mae": mae,
        "mse": mse,
        "parameter_names": parameter_names,
        # Registrado para saber depois qual arquitetura gerou este numero.
        "architecture": config.architecture.__dict__ | {"kernel_size": list(config.architecture.kernel_size)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _predictions_frame(file_names[test_idx], y_test, predictions, parameter_names).to_csv(
        out_dir / "predictions.csv", index=False
    )

    del features, targets, x_train, x_test, y_train, y_test, predictions
    return metrics


def train(config: ExperimentConfig) -> List[Dict[str, object]]:
    """Treina todas as cadeias do dataset (ou so a de `--chain-key`)."""
    train_config = config.train
    print("Sidecar integrity check passed:", validate_sidecar_integrity(train_config.dataset_root))

    set_global_seeds(train_config.seed, deterministic=train_config.deterministic)

    results_root = Path(train_config.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    chain_keys = (
        [train_config.chain_key]
        if train_config.chain_key
        else list_chain_keys(train_config.dataset_root)
    )
    if not chain_keys:
        raise RuntimeError("No matching chains found in dataset metadata.")

    all_metrics: List[Dict[str, object]] = []
    for chain_key_value in chain_keys:
        print(f"Treinando cadeia {chain_key_value}")
        metrics = train_one_chain(chain_key_value, config)
        all_metrics.append(metrics)
        print(f"{chain_key_value} -> MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}")
        if train_config.clear_session:
            clear_session()

    (results_root / "all_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    manifest = write_run_manifest(
        results_root,
        config,
        [{k: m[k] for k in ("chain_key", "feature", "mae", "mse")} for m in all_metrics],
    )
    print(f"Manifesto: {manifest}")
    return all_metrics
