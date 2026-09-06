"""Carga de um modelo treinado e predicao.

Existe para que o treino e o estudo cross-implementation passem pelo mesmo
caminho de escala e predicao. Antes o `cross_impl` reimplementava o laco de
`transform`, e uma divergencia silenciosa ali invalidaria a comparacao.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from gefx.training.scaling import apply_scalers

MODEL_FILENAME = "model.keras"
SCALERS_FILENAME = "feature_scalers.pkl"


@dataclass
class TrainedChain:
    """Modelo de uma cadeia e os scalers com que ele foi treinado."""

    model: object
    scalers: Dict[int, StandardScaler]
    directory: Path

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Escala e prediz. `features` vem cru, no formato (n, altura, largura)."""
        return self.model.predict(apply_scalers(features, self.scalers), verbose=0)


def load_trained_chain(model_dir: str | Path) -> TrainedChain:
    directory = Path(model_dir)
    model_path = directory / MODEL_FILENAME
    scalers_path = directory / SCALERS_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers nao encontrados: {scalers_path}")

    # Depois das checagens: um caminho errado nao precisa pagar o TensorFlow.
    import keras

    return TrainedChain(
        model=keras.models.load_model(model_path),
        scalers=joblib.load(scalers_path),
        directory=directory,
    )


def clear_session() -> None:
    """Limpa a sessao Keras entre cadeias, para limitar o pico de memoria."""
    import gc

    import keras

    keras.backend.clear_session()
    gc.collect()
