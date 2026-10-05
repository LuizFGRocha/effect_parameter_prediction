"""O encoder do POC II: espectrograma -> `z_e`, o codigo de efeito.

`z_e` e L2-normalizado porque a busca e por cosseno.

O encoder e a CNN do POC I (`ARCHITECTURES["poc1"]`, a rede de
`training/architecture.py` com `experiments/base.yaml`, conferida camada a camada)
com tres mudancas, cada uma medida num degrau da escada do POC I ao POC II
(SupCon, 3 sementes, recuperacao entre implementacoes):

- media no eixo do tempo depois das convolucoes: +8,6 pontos, em toda semente;
- filtros de (6, 12) para (32, 64): +3,1;
- quatro blocos, (32, 64, 96, 128): +1,5.

As duas densas de 64 do POC I ficam. Uma densa de 256 (o encoder anterior) dava
+1,5 ponto e mais que dobrava o conteudo legivel em `z_e` (19% -> 44%).
`time_pool="flatten"` (sem reducao) fica para a CNN do POC I.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_FILTERS: Tuple[int, ...] = (32, 64, 96, 128)

#: Campos que os manifestos antigos nao tem, com o valor com que eles rodaram.
LEGACY_FIELDS: Dict[str, object] = {"time_pool": "mean", "trunk_layers": 1}

#: Como o tronco trata o eixo do tempo antes da camada densa. O `max` da ablacao
#: (`results/disent/v2/ablacao_tempo/`) saiu: perdeu para a media.
TIME_POOLS: Tuple[str, ...] = ("mean", "flatten")

#: Onde cada execucao grava os pesos do encoder, relativo a pasta dela.
WEIGHTS_FILE = Path("weights") / "encoder.weights.h5"


@dataclass
class EncoderConfig:
    """Forma da rede. Vai inteira para o `run.json`, como no POC I."""

    input_shape: Tuple[int, int, int] = (256, 173, 1)
    filters: Tuple[int, ...] = DEFAULT_FILTERS
    kernel_size: int = 3
    trunk_units: int = 64
    dropout: float = 0.2
    effect_dim: int = 32
    time_pool: str = "mean"
    #: Camadas densas ocultas do tronco, como as `n_full - 1` do POC I.
    trunk_layers: int = 2

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
        data = {**LEGACY_FIELDS, **data}
        return cls(**{k: tuple(v) if isinstance(v, list) else v  # type: ignore[arg-type]
                      for k, v in data.items()})


def _mean_over_time(tensor):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia, canal)."""
    import tensorflow as tf

    return tf.reduce_mean(tensor, axis=2)


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

    if config.time_pool == "mean":
        x = layers.Lambda(
            _mean_over_time,
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

#: As tecnicas de `gefx disent train` (ver `train.py`).
TECHNIQUES: Tuple[str, ...] = ("random_encoder", "bn_only", "supcon", REGRESSION)


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


#: Arquiteturas nomeadas. `poc2` e o encoder; `poc1`, a CNN do POC I: dois blocos
#: de 6 e 12 filtros, sem reducao do tempo, duas densas de 64.
ARCHITECTURES: Dict[str, EncoderConfig] = {
    "poc2": EncoderConfig(),
    "poc1": EncoderConfig(filters=(6, 12), time_pool="flatten", trunk_units=64,
                          trunk_layers=2),
}
