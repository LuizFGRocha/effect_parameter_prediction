"""Extracao das features de audio.

A semantica aqui e contrato com os artefatos ja em disco: os caches `.npz` e os
`feature_scalers.pkl` do `second_main_run` foram construidos com exatamente estas
transformacoes. Mudar qualquer uma invalida os caches (use `--rebuild-cache`) e
torna os modelos treinados incomparaveis.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from skimage.transform import rescale
from spafe.features.gfcc import gfcc as sgfcc

FEATURE_NAMES = ("Spec", "MFCC40", "Chroma", "GFCC40")

# Taxa fixa a que o GFCC e extraido (o spafe espera banda estreita).
GFCC_SAMPLE_RATE = 16000


def extract_feature(file_path: Path, feature_name: str) -> np.ndarray:
    if feature_name not in FEATURE_NAMES:
        raise ValueError(
            f"Unsupported feature_name={feature_name}. Expected one of {sorted(FEATURE_NAMES)}"
        )

    if feature_name == "GFCC40":
        audio, _ = librosa.load(str(file_path), sr=GFCC_SAMPLE_RATE)
        audio = librosa.util.normalize(audio)
        return sgfcc(audio, num_ceps=40, nfilts=80)

    audio, sr = librosa.load(str(file_path), sr=None)
    audio = librosa.util.normalize(audio)

    if feature_name == "Spec":
        return rescale(np.abs(librosa.stft(audio)), scale=(0.25, 1.0))
    if feature_name == "MFCC40":
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return librosa.feature.chroma_stft(y=audio, sr=sr)


def stack_features(features: list[np.ndarray], feature_name: str) -> np.ndarray:
    """Empilha as features de uma pasta no formato (n, altura, largura).

    O GFCC sai do spafe transposto em relacao as demais, por isso a troca de eixos.
    """
    stacked = np.asarray(features, dtype=np.float32)
    if feature_name == "GFCC40":
        stacked = np.swapaxes(stacked, 1, 2)
    return stacked
