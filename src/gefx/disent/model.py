"""O encoder do POC II: espectrograma -> `z_e`, o codigo de efeito.

`z_e` e L2-normalizado porque a busca e por cosseno. O tronco tira a media so no
eixo do tempo: o ajuste de distorcao e estacionario num segmento de dois
segundos, e o que varia no tempo e o que foi tocado.

A media no tempo entrou por esse argumento, sem medicao. `time_pool` existe para
a ablacao: `max` reduz o tempo com o mesmo tamanho de saida, e `flatten` o
mantem inteiro (a `trunk_dense` fica ~8x maior), como a CNN do POC I.

`ARCHITECTURES["poc1"]` e a CNN do POC I (`training/architecture.py` com
`experiments/base.yaml`) escrita neste tronco: com a cabeca do regressor, e a
mesma rede camada a camada. E o ponto de partida da escada do POC I ao POC II,
em que cada degrau muda uma coisa so.
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
    #: Camadas densas ocultas do tronco (o POC I tem duas). Idem: antes, uma.
    trunk_layers: int = 1

    def __post_init__(self) -> None:
        if self.time_pool not in TIME_POOLS:
            raise ValueError(
                f"time_pool desconhecido: {self.time_pool!r}. Ha {list(TIME_POOLS)}"
            )
        if self.trunk_layers < 1:
            raise ValueError("o tronco precisa de ao menos uma camada densa")

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
    # A primeira camada guarda os nomes antigos, para os pesos gravados carregarem.
    for index in range(config.trunk_layers):
        suffix = str(index) if index else ""
        x = layers.Dense(config.trunk_units, activation="relu",
                         name=f"trunk_dense{suffix}")(x)
        x = layers.BatchNormalization(name=f"trunk_bn{suffix}")(x)
        x = layers.Dropout(config.dropout, name=f"trunk_drop{suffix}")(x)
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


#: A tecnica cujo modelo e o regressor escalar, e nao o encoder.
REGRESSION = "regressao"


def build_regressor(config: EncoderConfig):
    """Controle escalar: o mesmo tronco, uma saida sigmoide com o drive em [0, 1].

    Responde se o encoder precisa de `effect_dim` dimensoes para o nivel: com o
    mesmo tronco, dados e amostrador, so muda a cabeca. `effect_dim` e ignorado.
    """
    from keras import layers, models

    trunk = build_trunk(config)
    drive = layers.Dense(1, activation="sigmoid", name="drive")(trunk.output)
    return models.Model(trunk.input, drive, name="regressor")


def build_model(config: EncoderConfig, technique: str):
    """O regressor para `REGRESSION`, o encoder para as outras tecnicas."""
    return build_regressor(config) if technique == REGRESSION else build_encoder(config)


#: Arquiteturas nomeadas. `poc2` e o encoder do relatorio; `poc1`, a CNN do POC I:
#: dois blocos de 6 e 12 filtros, sem reducao do tempo, duas densas de 64.
ARCHITECTURES: Dict[str, EncoderConfig] = {
    "poc2": EncoderConfig(),
    "poc1": EncoderConfig(filters=(6, 12), time_pool="flatten", trunk_units=64,
                          trunk_layers=2),
}
