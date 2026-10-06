"""A CNN do POC I e o encoder do POC II, que e a mesma rede com outra saida.

O tronco: `n_conv` blocos Conv2D -> BatchNorm -> MaxPooling (o i-esimo com
`n_filters * (i + 1)` filtros e Dropout a partir do segundo), a reducao do eixo
do tempo (`time_pool`) e `n_full - 1` camadas densas com BatchNorm e Dropout.

- `build_model`, o regressor do POC I: saida sigmoide com uma unidade por
  parametro, MSE sobre os valores normalizados em [0, 1]. No POC II e o controle
  escalar (`regressao`), com uma unidade so, o drive.
- `build_encoder`, o encoder do POC II: uma projecao linear para `z_e` com
  `effect_dim` dimensoes, L2-normalizada porque a busca e por cosseno.

`ARCHITECTURES` e a escada do POC I ao POC II: cada degrau muda um campo da CNN
do POC I, e cada mudanca foi medida (`scripts/poc2_escada.sh`).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence

from gefx.config import ArchitectureConfig

#: Os degraus, na ordem. `poc1` e `experiments/base.yaml`; `poc2`, o encoder do
#: relatorio. As duas densas de 64 do POC I ficam: uma densa de 256 dava +1,5
#: ponto e mais que dobrava o conteudo legivel em `z_e` (19% -> 44%; protocolo
#: antigo, no teste).
ARCHITECTURES: Dict[str, ArchitectureConfig] = {"poc1": ArchitectureConfig()}
ARCHITECTURES["poc1_media"] = replace(ARCHITECTURES["poc1"], time_pool="mean")
ARCHITECTURES["poc1_media_largo"] = replace(ARCHITECTURES["poc1_media"], n_filters=32)
ARCHITECTURES["poc2"] = replace(ARCHITECTURES["poc1_media_largo"], n_conv=4)


def build_trunk(input_dim: Sequence[int], config: ArchitectureConfig):
    """`(entrada, saida do tronco)`, para as duas cabecas."""
    from keras import layers

    inputs = layers.Input(shape=tuple(input_dim))
    x = inputs
    for index in range(config.n_conv):
        x = layers.Conv2D(config.n_filters * (index + 1), kernel_size=config.kernel_size,
                          activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        if index:
            x = layers.Dropout(config.dropout)(x)

    if config.time_pool == "mean":
        # A media no eixo do tempo, (frequencia, tempo, canal) -> (frequencia, 1, canal).
        x = layers.AveragePooling2D(pool_size=(1, x.shape[2]))(x)
    x = layers.Flatten()(x)
    for _ in range(config.n_full - 1):
        x = layers.Dense(config.n_nodes, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout)(x)
    return inputs, x


def build_model(input_dim: Sequence[int], output_dim: int, config: ArchitectureConfig):
    """O regressor: tronco + saida sigmoide, compilado com MSE como no POC I."""
    from keras import layers, models, optimizers

    inputs, x = build_trunk(input_dim, config)
    model = models.Model(inputs, layers.Dense(output_dim, activation="sigmoid")(x))
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        metrics=["mse", "mae"],
    )
    return model


def build_encoder(input_dim: Sequence[int], config: ArchitectureConfig):
    """O encoder: tronco + projecao para `z_e`, na esfera unitaria."""
    from keras import layers, models

    inputs, x = build_trunk(input_dim, config)
    z_e = layers.Dense(config.effect_dim)(x)
    return models.Model(inputs, layers.UnitNormalization()(z_e))


#: As tecnicas do POC II (`training/encoder.py`); so `REGRESSION` usa o regressor.
REGRESSION = "regressao"
TECHNIQUES = ("random_encoder", "bn_only", "supcon", REGRESSION)


def build_network(input_dim: Sequence[int], config: ArchitectureConfig, technique: str):
    """O regressor (uma saida, o drive) para `REGRESSION`, o encoder para as outras."""
    if technique == REGRESSION:
        return build_model(input_dim, 1, config)
    return build_encoder(input_dim, config)
