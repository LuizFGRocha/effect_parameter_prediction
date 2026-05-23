from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import pandas as pd
from skimage.transform import rescale
from spafe.features.gfcc import gfcc as sgfcc

from chain_definitions import EFFECT_PARAMETER_RANGES, chain_key, chain_key_to_effects

FEATURE_NAMES = {"Spec", "MFCC40", "Chroma", "GFCC40"}


def _extract_features(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, sr = librosa.load(str(file_path), sr=None)
    y = librosa.util.normalize(y)
    y1, sr1 = librosa.load(str(file_path), sr=16000)
    y1 = librosa.util.normalize(y1)

    spectrogram = np.abs(librosa.stft(y))
    spectrogram = rescale(spectrogram, scale=(0.25, 1.0))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    gfcc = sgfcc(y1, num_ceps=40, nfilts=80)

    return spectrogram, mfcc, chroma, gfcc


def _read_metadata(dataset_root: Path) -> pd.DataFrame:
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata sidecar: {metadata_path}")
    return pd.read_csv(metadata_path)


def _ensure_chain_key(metadata: pd.DataFrame) -> pd.DataFrame:
    if "chain_key" in metadata.columns:
        return metadata

    def _build_key(effect_order_value: str) -> str:
        effects = json.loads(effect_order_value)
        return chain_key(effects)

    metadata = metadata.copy()
    metadata["chain_key"] = metadata["effect_order"].apply(_build_key)
    return metadata


def _chain_folder(dataset_root: Path, chain_key_value: str) -> Path:
    folder = dataset_root / chain_key_value
    if not folder.exists():
        raise FileNotFoundError(f"Missing chain folder: {folder}")
    return folder


def _feature_cache_files(chain_folder: Path) -> Dict[str, Path]:
    return {
        "Spec": chain_folder / "Spec.npz",
        "MFCC40": chain_folder / "MFCC40.npz",
        "Chroma": chain_folder / "Chroma.npz",
        "GFCC40": chain_folder / "GFCC40.npz",
        "file_names": chain_folder / "file_names.json",
    }


def _all_cache_exists(cache_files: Dict[str, Path]) -> bool:
    return all(path.exists() for path in cache_files.values())


def _build_feature_cache(chain_folder: Path) -> Dict[str, np.ndarray]:
    specs: List[np.ndarray] = []
    mfccs: List[np.ndarray] = []
    chromas: List[np.ndarray] = []
    gfccs: List[np.ndarray] = []

    wav_files = sorted(path for path in chain_folder.glob("*.wav") if path.is_file())
    if not wav_files:
        raise RuntimeError(f"No wav files found in {chain_folder}")

    for wav_file in wav_files:
        spec, mfcc, chroma, gfcc = _extract_features(wav_file)
        specs.append(spec)
        mfccs.append(mfcc)
        chromas.append(chroma)
        gfccs.append(gfcc)

    return {
        "Spec": np.asarray(specs, dtype=np.float32),
        "MFCC40": np.asarray(mfccs, dtype=np.float32),
        "Chroma": np.asarray(chromas, dtype=np.float32),
        "GFCC40": np.swapaxes(np.asarray(gfccs, dtype=np.float32), 1, 2),
        "file_names": np.array([wav.name for wav in wav_files]),
    }


def _save_cache(chain_folder: Path, cache: Dict[str, np.ndarray]) -> None:
    cache_files = _feature_cache_files(chain_folder)
    for feat in FEATURE_NAMES:
        np.savez(cache_files[feat], cache[feat])
    cache_files["file_names"].write_text(
        json.dumps(cache["file_names"].tolist(), indent=2),
        encoding="utf-8",
    )


def _load_cache(chain_folder: Path) -> Dict[str, np.ndarray]:
    cache_files = _feature_cache_files(chain_folder)
    return {
        "Spec": np.load(cache_files["Spec"])["arr_0"].astype(np.float32, copy=False),
        "MFCC40": np.load(cache_files["MFCC40"])["arr_0"].astype(np.float32, copy=False),
        "Chroma": np.load(cache_files["Chroma"])["arr_0"].astype(np.float32, copy=False),
        "GFCC40": np.load(cache_files["GFCC40"])["arr_0"].astype(np.float32, copy=False),
        "file_names": np.array(json.loads(cache_files["file_names"].read_text(encoding="utf-8"))),
    }


def ensure_feature_cache(
    dataset_root: Path,
    chain_key_value: str,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    chain_folder = _chain_folder(dataset_root, chain_key_value)
    cache_files = _feature_cache_files(chain_folder)
    if _all_cache_exists(cache_files) and not force_rebuild:
        return _load_cache(chain_folder)

    cache = _build_feature_cache(chain_folder)
    _save_cache(chain_folder, cache)
    return cache


def _build_target_lookup(metadata: pd.DataFrame, chain_key_value: str) -> Dict[str, List[float]]:
    subset = metadata[metadata["chain_key"] == chain_key_value]
    lookup: Dict[str, List[float]] = {}
    for _, row in subset.iterrows():
        file_name = row["file_name"]
        vector = json.loads(row["normalized_parameter_vector"])
        lookup[file_name] = [float(v) for v in vector]
    return lookup


def load_chain_dataset(
    dataset_root: str,
    chain_key_value: str,
    feature_name: str = "MFCC40",
    force_rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_name not in FEATURE_NAMES:
        raise ValueError(f"Unsupported feature_name={feature_name}. Expected one of {sorted(FEATURE_NAMES)}")

    root = Path(dataset_root).resolve()
    metadata = _ensure_chain_key(_read_metadata(root))
    cache = ensure_feature_cache(root, chain_key_value, force_rebuild=force_rebuild_cache)

    file_names = cache["file_names"]
    lookup = _build_target_lookup(metadata, chain_key_value)

    missing = [name for name in file_names if name not in lookup]
    if missing:
        raise RuntimeError(
            "Metadata mismatch: some wav files have no target row. "
            f"Example missing file: {missing[0]}"
        )

    y = np.asarray([lookup[name] for name in file_names], dtype=np.float32)
    X = cache[feature_name].astype(np.float32, copy=False)

    if len(X) != len(y):
        raise RuntimeError(f"Feature/target size mismatch: X={len(X)} y={len(y)}")

    return X, y, np.array(file_names)


def validate_sidecar_integrity(dataset_root: str) -> Dict[str, int]:
    root = Path(dataset_root).resolve()
    metadata = _ensure_chain_key(_read_metadata(root))

    required_cols = {
        "file_name",
        "chain_key",
        "chain_length",
        "effect_order",
        "effect_presence",
        "normalized_parameter_vector",
        "raw_parameter_dict",
        "source_audio_id",
        "random_seed",
    }
    missing_cols = sorted(required_cols - set(metadata.columns))
    if missing_cols:
        raise RuntimeError(f"Missing required metadata columns: {missing_cols}")

    counts: Dict[str, int] = {}
    chain_keys = sorted(metadata["chain_key"].unique().tolist())
    for chain_key_value in chain_keys:
        folder = _chain_folder(root, chain_key_value)
        wavs = sorted(path.name for path in folder.glob("*.wav") if path.is_file())
        rows = metadata[metadata["chain_key"] == chain_key_value]

        if len(wavs) != len(rows):
            raise RuntimeError(
                f"Chain {chain_key_value}: wav/metadata mismatch wav={len(wavs)} rows={len(rows)}"
            )

        row_names = set(rows["file_name"].tolist())
        missing_rows = [name for name in wavs if name not in row_names]
        if missing_rows:
            raise RuntimeError(
                f"Chain {chain_key_value}: metadata missing for wav {missing_rows[0]}"
            )

        for _, row in rows.iterrows():
            vec = json.loads(row["normalized_parameter_vector"])
            if not all(0.0 <= float(v) <= 1.0 for v in vec):
                raise RuntimeError(
                    f"Chain {chain_key_value}: normalized value outside [0,1] in {row['file_name']}"
                )

        counts[chain_key_value] = len(wavs)

    return counts


def list_chain_keys(dataset_root: str) -> List[str]:
    root = Path(dataset_root).resolve()
    metadata = _ensure_chain_key(_read_metadata(root))
    return sorted(metadata["chain_key"].unique().tolist())


def chain_keys_for_length(dataset_root: str, chain_length: int) -> List[str]:
    root = Path(dataset_root).resolve()
    metadata = _ensure_chain_key(_read_metadata(root))
    rows = metadata[metadata["chain_length"] == chain_length]
    return sorted(rows["chain_key"].unique().tolist())


def parameter_names_for_chain(chain_key_value: str) -> List[str]:
    chain_effects = chain_key_to_effects(chain_key_value)
    names: List[str] = []
    for effect in chain_effects:
        for param in EFFECT_PARAMETER_RANGES[effect]:
            names.append(f"{effect}_{param['name']}")
    return names
