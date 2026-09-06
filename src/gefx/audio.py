"""Entrada/saida de audio, normalizacao de loudness e recorte de segmento."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pyloudnorm as pyln
from pedalboard.io import AudioFile

DEFAULT_LOUDNESS_LEVEL = -26.0


@functools.lru_cache(maxsize=8)
def get_meter(sr: int) -> pyln.Meter:
    return pyln.Meter(sr)


def load_audio_file(file_path: Path) -> Tuple[np.ndarray, int]:
    with AudioFile(str(file_path), "r") as handle:
        audio = handle.read(handle.frames)
        sample_rate = handle.samplerate
    return audio, sample_rate


def export_audio(audio: np.ndarray, sr: int, path: Path) -> None:
    with AudioFile(str(path), "w", sr, audio.shape[0]) as handle:
        handle.write(audio)


def normalize_loudness(
    audio: np.ndarray,
    sr: int,
    loudness_level: float = DEFAULT_LOUDNESS_LEVEL,
) -> np.ndarray:
    flat = np.reshape(audio, np.shape(audio)[1])
    meter = get_meter(sr)
    loudness = meter.integrated_loudness(flat)
    normalized = pyln.normalize.loudness(flat, loudness, loudness_level)
    return np.reshape(normalized, (1, np.shape(audio)[1]))


def select_random_segment(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    segment_seconds: float,
    ignore_start_seconds: float,
    ignore_end_seconds: float,
) -> np.ndarray:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if ignore_start_seconds < 0 or ignore_end_seconds < 0:
        raise ValueError("ignore_start_seconds and ignore_end_seconds must be >= 0")

    total_frames = audio.shape[1]
    segment_frames = int(round(segment_seconds * sr))
    if segment_frames <= 0:
        raise ValueError("segment_seconds is too small for the current sample rate")

    min_start = int(round(ignore_start_seconds * sr))
    max_end = total_frames - int(round(ignore_end_seconds * sr))
    max_start = max_end - segment_frames
    if max_start < min_start:
        total_seconds = total_frames / sr
        raise ValueError(
            "Segment selection exceeds audio length. "
            f"segment_seconds={segment_seconds}, ignore_start_seconds={ignore_start_seconds}, "
            f"ignore_end_seconds={ignore_end_seconds}, audio_seconds={total_seconds:.3f}"
        )

    start = int(rng.integers(min_start, max_start + 1))
    end = start + segment_frames
    return audio[:, start:end]


def iter_wav_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.wav")):
        if path.is_file():
            yield path
