"""O encoder do POC II: espectrograma -> `z_e`, o codigo de efeito.

`z_e` e L2-normalizado porque a busca e por cosseno. O tronco tira a media so no
eixo do tempo: o ajuste de distorcao e estacionario num segmento de dois
segundos, e o que varia no tempo e o que foi tocado.

A media no tempo entrou por esse argumento, sem medicao. `time_pool` existe para
a ablacao: `max` reduz o tempo com o mesmo tamanho de saida, e `flatten` o
mantem inteiro (a `trunk_dense` fica ~8x maior), como a CNN do POC I.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_FILTERS: Tuple[int, ...] = (32, 64, 96, 128)

#: Como o tronco trata o eixo do tempo antes da camada densa.
TIME_POOLS: Tuple[str, ...] = ("mean", "max", "flatten")

#: Onde cada execucao grava os pesos do encoder, relativo a pasta dela.
WEIGHTS_FILE = Path("weights") / "encoder.weights.h5"


@dataclass
class EncoderConfig:
    """Forma da rede. Vai inteira para o `run.json`, como no POC I."""

    input_shape: Tuple[int, int, int] = (256, 173, 1)
    filters: Tuple[int, ...] = DEFAULT_FILTERS
    kernel_size: int = 3
    trunk_units: int = 256
    dropout: float = 0.2
    effect_dim: int = 32
    #: Manifestos anteriores a ablacao nao tem o campo: rodaram com a media.
    time_pool: str = "mean"

    def __post_init__(self) -> None:
        if self.time_pool not in TIME_POOLS:
            raise ValueError(
                f"time_pool desconhecido: {self.time_pool!r}. Ha {list(TIME_POOLS)}"
            )

    def as_dict(self) -> Dict[str, object]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EncoderConfig":
        return cls(**{k: tuple(v) if isinstance(v, list) else v  # type: ignore[arg-type]
                      for k, v in data.items()})


def _mean_over_time(tensor):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia, canal)."""
    import tensorflow as tf

    return tf.reduce_mean(tensor, axis=2)


def _max_over_time(tensor):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia, canal)."""
    import tensorflow as tf

    return tf.reduce_max(tensor, axis=2)


_TIME_REDUCERS = {"mean": _mean_over_time, "max": _max_over_time}


def build_trunk(config: EncoderConfig):
    """Conv2D -> BN -> pool empilhados, reducao do tempo, denso. Um `keras.Model`.

    Com `time_pool="flatten"` nao ha reducao: o `Flatten` leva frequencia e tempo.
    """
    from keras import layers, models

    inputs = layers.Input(shape=tuple(config.input_shape), name="spec")
    x = inputs
    for index, width in enumerate(config.filters):
        x = layers.Conv2D(
            width, config.kernel_size, activation="relu", name=f"conv{index}"
        )(x)
        x = layers.BatchNormalization(name=f"bn{index}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{index}")(x)
        if index:
            x = layers.Dropout(config.dropout, name=f"drop{index}")(x)

    if config.time_pool in _TIME_REDUCERS:
        x = layers.Lambda(
            _TIME_REDUCERS[config.time_pool],
            output_shape=lambda shape: (shape[0], shape[1], shape[3]),
            name="time_pool",
        )(x)
    x = layers.Flatten(name="flat")(x)
    x = layers.Dense(config.trunk_units, activation="relu", name="trunk_dense")(x)
    x = layers.BatchNormalization(name="trunk_bn")(x)
    x = layers.Dropout(config.dropout, name="trunk_drop")(x)
    return models.Model(inputs, x, name="trunk")


def build_encoder(config: EncoderConfig):
    """Tronco + projecao para `z_e`, L2-normalizado.

    O treino usa este `keras.Model` direto; os pesos vao para `WEIGHTS_FILE`.
    """
    from keras import layers, models

    trunk = build_trunk(config)
    z_e = layers.Dense(config.effect_dim, name="effect_dense")(trunk.output)
    z_e = layers.UnitNormalization(name="effect_code")(z_e)
    return models.Model(trunk.input, z_e, name="encoder")
