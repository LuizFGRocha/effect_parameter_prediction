"""Padronizacao das features, uma `StandardScaler` por linha do espectrograma.

Os scalers sao ajustados so no treino e persistidos, porque a inferencia em
outro lugar (`crossimpl`) precisa reproduzir exatamente a mesma escala. Este
modulo e o unico ponto que aplica escala — se treino e cross-impl divergirem
aqui, a comparacao entre implementacoes deixa de significar alguma coisa.
"""
from __future__ import annotations

from pathlib import Path
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


class PixelStandardizer:
    """Media e desvio por pixel do espectrograma, acumulados numa passada.

    E a mesma conta de `fit_scalers`: uma `StandardScaler` por linha ja da media e
    desvio por coluna, isto e, por pixel. Aqui ela e alimentada por `partial_fit`
    em blocos, para nao materializar o treino do POC II, que nao cabe na RAM.
    Ajustado so no treino e persistido com a execucao, em `.npz`. O POC I segue
    com os `feature_scalers.pkl` ja gravados.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, n: int) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.n = int(n)

    @classmethod
    def fit(cls, store, rows=None, chunk: int = 256,
            floor: float = 1e-6) -> "PixelStandardizer":
        from sklearn.preprocessing import StandardScaler

        rows = np.arange(len(store)) if rows is None else np.asarray(rows, dtype=np.int64)
        if not len(rows):
            raise ValueError("nao ha linhas para ajustar a padronizacao")
        scaler = StandardScaler()
        for _, block in store.stream(rows, chunk=chunk):
            scaler.partial_fit(block.reshape(len(block), -1))
        shape = store.feature_shape
        # Pixels constantes (silencio nas bandas altas) teriam desvio zero.
        std = np.maximum(np.sqrt(scaler.var_), floor)
        return cls(scaler.mean_.reshape(shape), std.reshape(shape), len(rows))

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Padroniza e acrescenta o eixo de canal, como espera a Conv2D."""
        if features.shape[1:] != self.mean.shape:
            raise ValueError(
                f"shape {features.shape[1:]} incompativel com a padronizacao {self.mean.shape}"
            )
        scaled = (features.astype(np.float32) - self.mean) / self.std
        return scaled[..., None]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, n=np.array(self.n))

    @classmethod
    def load(cls, path: Path) -> "PixelStandardizer":
        with np.load(Path(path)) as data:
            return cls(data["mean"], data["std"], int(data["n"]))
