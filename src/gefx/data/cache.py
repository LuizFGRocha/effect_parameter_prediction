"""Cache de features, gravado ao lado do audio, dentro do proprio dataset.

`file_names.json` e a ordenacao canonica das linhas: e por ele que
`data/dataset.py` monta `y` a partir do sidecar. Cache e sidecar precisam ficar
em sincronia — apagar os arquivos, ou passar `--rebuild-cache`, e o caminho
depois de mexer na extracao.

Como o cache mora dentro do dataset, copiar um dataset copia caches velhos junto.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from gefx.data.features import extract_feature, stack_features

FILE_NAMES_CACHE = "file_names.json"


def feature_cache_file(chain_folder: Path, feature_name: str) -> Path:
    return chain_folder / f"{feature_name}.npz"


def file_names_cache_file(chain_folder: Path) -> Path:
    return chain_folder / FILE_NAMES_CACHE


def cache_exists(chain_folder: Path, feature_name: str) -> bool:
    return feature_cache_file(chain_folder, feature_name).exists() and file_names_cache_file(
        chain_folder
    ).exists()


def build_cache(chain_folder: Path, feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
    wav_files = sorted(path for path in chain_folder.glob("*.wav") if path.is_file())
    if not wav_files:
        raise RuntimeError(f"No wav files found in {chain_folder}")

    features = [extract_feature(wav_file, feature_name) for wav_file in wav_files]
    return stack_features(features, feature_name), np.array([wav.name for wav in wav_files])


def save_cache(
    chain_folder: Path,
    feature_name: str,
    features: np.ndarray,
    file_names: np.ndarray,
) -> None:
    np.savez(feature_cache_file(chain_folder, feature_name), features)
    file_names_cache_file(chain_folder).write_text(
        json.dumps(file_names.tolist(), indent=2), encoding="utf-8"
    )


def load_cache(chain_folder: Path, feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
    features = np.load(feature_cache_file(chain_folder, feature_name))["arr_0"]
    file_names = np.array(
        json.loads(file_names_cache_file(chain_folder).read_text(encoding="utf-8"))
    )
    return features.astype(np.float32, copy=False), file_names


def ensure_feature_cache(
    chain_folder: Path,
    feature_name: str,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Devolve (features, file_names), extraindo e gravando o cache na primeira vez."""
    if cache_exists(chain_folder, feature_name) and not force_rebuild:
        return load_cache(chain_folder, feature_name)

    features, file_names = build_cache(chain_folder, feature_name)
    save_cache(chain_folder, feature_name, features, file_names)
    return features, file_names
