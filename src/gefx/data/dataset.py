"""Leitura de um dataset: features em cache casadas com os alvos do sidecar."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from gefx.data.cache import ensure_feature_cache
from gefx.data.features import FEATURE_NAMES
from gefx.data.metadata import chain_folder, read_metadata, target_lookup


def load_chain_dataset(
    dataset_root: str | Path,
    chain_key_value: str,
    feature_name: str = "MFCC40",
    force_rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (X, y, file_names) de uma cadeia.

    A ordem das linhas vem do `file_names.json` do cache; `y` e montado
    consultando o sidecar por nome de arquivo, e nao pela ordem do CSV.
    """
    if feature_name not in FEATURE_NAMES:
        raise ValueError(
            f"Unsupported feature_name={feature_name}. Expected one of {sorted(FEATURE_NAMES)}"
        )

    root = Path(dataset_root).resolve()
    metadata = read_metadata(root)
    features, file_names = ensure_feature_cache(
        chain_folder(root, chain_key_value),
        feature_name=feature_name,
        force_rebuild=force_rebuild_cache,
    )

    lookup = target_lookup(metadata, chain_key_value)
    missing = [name for name in file_names if name not in lookup]
    if missing:
        raise RuntimeError(
            "Metadata mismatch: some wav files have no target row. "
            f"Example missing file: {missing[0]}"
        )

    targets = np.asarray([lookup[name] for name in file_names], dtype=np.float32)
    if len(features) != len(targets):
        raise RuntimeError(f"Feature/target size mismatch: X={len(features)} y={len(targets)}")

    return features, targets, file_names
