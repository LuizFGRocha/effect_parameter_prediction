"""Padronizacao das features, uma `StandardScaler` por linha do espectrograma.

Os scalers sao ajustados so no treino e persistidos, porque a inferencia em
outro lugar (`crossimpl`) precisa reproduzir exatamente a mesma escala. Este
modulo e o unico ponto que aplica escala — se treino e cross-impl divergirem
aqui, a comparacao entre implementacoes deixa de significar alguma coisa.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scalers(features: np.ndarray) -> Dict[int, StandardScaler]:
    """Ajusta uma scaler por linha (eixo 1) sobre o conjunto de treino."""
    scalers: Dict[int, StandardScaler] = {}
    for index in range(features.shape[1]):
        scalers[index] = StandardScaler().fit(features[:, index, :])
    return scalers


def apply_scalers(features: np.ndarray, scalers: Dict[int, StandardScaler]) -> np.ndarray:
    """Aplica os scalers e acrescenta o eixo de canal. Nao altera `features`.

    Exige uma scaler por linha. Sem essa checagem, um array com menos linhas do
    que scalers seria escalado so em parte e devolvido sem erro — e o `crossimpl`
    reusa scalers persistidos, entao a comparacao entre implementacoes sairia
    errada em silencio.
    """
    if features.shape[1] != len(scalers):
        raise ValueError(
            f"Numero de linhas incompativel com os scalers: features={features.shape[1]}, "
            f"scalers={len(scalers)}"
        )

    scaled = features.copy()
    for index in range(scaled.shape[1]):
        scaled[:, index, :] = scalers[index].transform(scaled[:, index, :])
    return np.expand_dims(scaled, axis=3)


def fit_transform_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[int, StandardScaler]]:
    """Ajusta no treino, aplica nos dois. Devolve copias; os originais ficam intactos."""
    scalers = fit_scalers(x_train)
    return apply_scalers(x_train, scalers), apply_scalers(x_test, scalers), scalers
