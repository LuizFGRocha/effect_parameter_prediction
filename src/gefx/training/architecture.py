"""Arquitetura da CNN de regressao: Conv2D -> Dense, saida sigmoide.

A saida tem largura igual ao numero de parametros previstos da cadeia, e a perda
e MSE sobre os valores normalizados em [0,1] — dai a sigmoide.
"""
from __future__ import annotations

from typing import Sequence

from gefx.config import ArchitectureConfig


def build_model(input_dim: Sequence[int], output_dim: int, config: ArchitectureConfig):
    from keras import layers, models, optimizers

    model = models.Sequential()
    model.add(layers.Input(shape=tuple(input_dim)))
    model.add(layers.Conv2D(config.n_filters, kernel_size=config.kernel_size, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # O i-esimo bloco convolucional extra alarga os filtros proporcionalmente.
    for index in range(1, config.n_conv):
        model.add(
            layers.Conv2D(
                config.n_filters * (index + 1),
                kernel_size=config.kernel_size,
                activation="relu",
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(config.dropout))

    model.add(layers.Flatten())
    for _ in range(config.n_full - 1):
        model.add(layers.Dense(config.n_nodes, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config.dropout))

    model.add(layers.Dense(output_dim, activation="sigmoid"))
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        metrics=["mse", "mae"],
    )
    return model
