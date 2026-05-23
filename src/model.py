from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers # type: ignore
from chain_definitions import chain_key_to_effects
from metadata_utils import (
    chain_keys_for_length,
    list_chain_keys,
    load_chain_dataset,
    parameter_names_for_chain,
    validate_sidecar_integrity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed-order chain parameter models.")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing metadata.csv and chain folders.")
    parser.add_argument("--feature", default="MFCC40", choices=["Spec", "MFCC40", "Chroma", "GFCC40"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--results-root", default='results/default_results')
    parser.add_argument(
        "--chain-key",
        type=str,
        default=None,
        help="Train only a specific chain key (e.g. distortion__chorus).",
    )
    parser.add_argument(
        "--no-clear-session",
        action="store_true",
        help="Disable TensorFlow session cleanup between chain runs.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute feature cache files before training (useful to migrate old float64 caches).",
    )
    return parser.parse_args()


def choose_path(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_conv(model, i, kernel_size, n_filters):
    model.add(layers.Conv2D(n_filters * (i + 1), kernel_size=kernel_size, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    return model


def add_full(model, n_nodes):
    model.add(layers.Dense(n_nodes, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    return model


def scale_data(X_train: np.ndarray, X_test: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    scalers = {}
    for i in range(X_train.shape[1]):
        scalers[i] = StandardScaler()
        X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
    for i in range(X_test.shape[1]):
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    return X_train, X_test, scalers


def create_model(
    input_dim,
    output_dim,
    kernel_size=(3, 3),
    n_conv=2,
    n_full=3,
    n_nodes=64,
    n_filters=6,
):
    model = models.Sequential()
    model.add(layers.Conv2D(n_filters, kernel_size=kernel_size, activation="relu", input_shape=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    for i in range(n_conv - 1):
        model = add_conv(model, i + 1, kernel_size, n_filters)

    model.add(layers.Flatten())
    for _ in range(n_full - 1):
        model = add_full(model, n_nodes)

    model.add(layers.Dense(output_dim, activation="sigmoid"))
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["mse", "mae"],
    )
    return model


def train_one_chain(
    dataset_root: str,
    chain_key_value: str,
    feature: str,
    epochs: int,
    batch_size: int,
    test_size: float,
    split_seed: int,
    results_root: Path,
    rebuild_cache: bool,
) -> Dict[str, float]:
    chain_effects = chain_key_to_effects(chain_key_value)
    chain_length = len(chain_effects)
    X, y, file_names = load_chain_dataset(
        dataset_root,
        chain_key_value,
        feature_name=feature,
        force_rebuild_cache=rebuild_cache,
    )

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=split_seed, shuffle=True)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test, scalers = scale_data(X_train, X_test)

    model = create_model(input_dim=X_train.shape[1:], output_dim=y_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(X_test, y_test),
    )

    pred = model.predict(X_test, verbose=0)
    mse = float(np.mean((pred - y_test) ** 2))
    mae = float(np.mean(np.abs(pred - y_test)))

    out_dir = results_root / chain_key_value
    choose_path(out_dir)

    model.save(out_dir / "model.keras")
    joblib.dump(scalers, out_dir / "feature_scalers.pkl")

    np.savez(
        out_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        train_files=file_names[train_idx],
        test_files=file_names[test_idx],
    )

    with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, handle, indent=2)

    metrics = {
        "chain_key": chain_key_value,
        "chain_length": chain_length,
        "effect_order": chain_effects,
        "feature": feature,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "output_dim": int(y.shape[1]),
        "mae": mae,
        "mse": mse,
        "parameter_names": parameter_names_for_chain(chain_key_value),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    rows = []
    param_names = parameter_names_for_chain(chain_key_value)
    for index, file_name in enumerate(file_names[test_idx]):
        row = {"file_name": str(file_name)}
        for p_i, p_name in enumerate(param_names):
            row[f"y_true_{p_name}"] = float(y_test[index, p_i])
            row[f"y_pred_{p_name}"] = float(pred[index, p_i])
        rows.append(row)

    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    # Free large arrays as soon as this chain finishes to reduce peak RAM usage.
    del X, y, X_train, X_test, y_train, y_test, pred, rows

    return metrics


def cleanup_after_chain() -> None:
    from tensorflow.keras import backend as K # type: ignore

    K.clear_session()
    gc.collect()


def main() -> None:
    args = parse_args()

    integrity = validate_sidecar_integrity(args.dataset_root)
    print("Sidecar integrity check passed:", integrity)

    if args.results_root:
        results_root = Path(args.results_root).resolve()
    else:
        results_root = Path(__file__).resolve().parent.parent.parent / "Results" / "Parameter Estimation" / "PaperFixedChains"

    choose_path(results_root)

    if args.chain_key:
        chain_keys = [args.chain_key]
    else:
        chain_keys = list_chain_keys(args.dataset_root)

    if not chain_keys:
        raise RuntimeError("No matching chains found in dataset metadata.")

    all_metrics = []
    for chain_key_value in chain_keys:
        print(f"Training chain {chain_key_value}")
        metrics = train_one_chain(
            dataset_root=args.dataset_root,
            chain_key_value=chain_key_value,
            feature=args.feature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            split_seed=args.split_seed,
            results_root=results_root,
            rebuild_cache=args.rebuild_cache,
        )
        all_metrics.append(metrics)
        print(f"{chain_key_value} -> MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}")
        if not args.no_clear_session:
            cleanup_after_chain()

    with (results_root / "all_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)


if __name__ == "__main__":
    # Keep TensorFlow logs quieter for long experiments.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
