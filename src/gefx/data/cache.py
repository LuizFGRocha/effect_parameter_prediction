"""Cache de features, gravado ao lado do audio, dentro do proprio dataset.

`file_names.json` e a ordenacao canonica das linhas: e por ele que
`data/dataset.py` monta `y` a partir do sidecar. Cache e sidecar precisam ficar
em sincronia — apagar os arquivos, ou passar `--rebuild-cache`, e o caminho
depois de mexer na extracao.

Como o cache mora dentro do dataset, copiar um dataset copia caches velhos junto.

As features ficam num `.npy`, e nao num `.npz`: o POC II abre os caches de todos
os arms juntos, que nao cabem na RAM, e um `.npy` se mapeia direto do disco
(`mmap_mode="r"`). Um `.npz` antigo converte sem reextrair, porque o `np.savez`
nao comprime: `unzip -p Spec.npz arr_0.npy > Spec.npy`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from gefx.data.features import extract_feature, stack_features

FILE_NAMES_CACHE = "file_names.json"


def feature_cache_file(chain_folder: Path, feature_name: str) -> Path:
    return chain_folder / f"{feature_name}.npy"


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
    np.save(feature_cache_file(chain_folder, feature_name), features)
    file_names_cache_file(chain_folder).write_text(
        json.dumps(file_names.tolist(), indent=2), encoding="utf-8"
    )


def load_cache(
    chain_folder: Path, feature_name: str, mmap_mode: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """`mmap_mode="r"` mapeia as features do disco em vez de carrega-las."""
    features = np.load(feature_cache_file(chain_folder, feature_name), mmap_mode=mmap_mode)
    file_names = np.array(
        json.loads(file_names_cache_file(chain_folder).read_text(encoding="utf-8"))
    )
    if len(file_names) != len(features):
        raise RuntimeError(
            f"{chain_folder}: cache com {len(features)} linhas e "
            f"{FILE_NAMES_CACHE} com {len(file_names)}"
        )
    return features.astype(np.float32, copy=False), file_names


def ensure_feature_cache(
    chain_folder: Path,
    feature_name: str,
    force_rebuild: bool = False,
    mmap_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Devolve (features, file_names), extraindo e gravando o cache na primeira vez."""
    if force_rebuild or not cache_exists(chain_folder, feature_name):
        features, file_names = build_cache(chain_folder, feature_name)
        save_cache(chain_folder, feature_name, features, file_names)
    return load_cache(chain_folder, feature_name, mmap_mode)
