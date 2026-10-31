"""A CNN do POC I e o encoder do POC II, que e a mesma rede com outra saida.

O tronco: `n_conv` blocos Conv2D -> BatchNorm -> MaxPooling (o i-esimo com
`n_filters * (i + 1)` filtros e Dropout a partir do segundo), a reducao do eixo
do tempo (`time_pool`) e `n_full - 1` camadas densas com BatchNorm e Dropout.

- `build_model`, o regressor do POC I: saida sigmoide com uma unidade por
  parametro, MSE sobre os valores normalizados em [0, 1]. No POC II e o controle
  escalar (`regressao`), com uma unidade so, o drive.
- `build_encoder`, o encoder do POC II: uma projecao linear para `z_e` com
  `effect_dim` dimensoes, L2-normalizada porque a busca e por cosseno. Com
  `content_dim`, uma segunda cabeca, `z_c`, sobre as mesmas convolucoes: o
  desemaranhamento em dois blocos, um subespaco por nocao de similaridade. E a
  variante "fixed disjoint masks" das Conditional Similarity Networks (Veit,
  Belongie e Karaletsos, CVPR 2017; andreasveit/conditional-similarity-networks,
  BSD-3), que o artigo descreve como "fully disentangled", levada ao audio por
  Lee et al. (ICASSP 2020). Aqui cada bloco tem as suas densas, e nao so as suas
  dimensoes, porque as duas cabecas reduzem o tempo de jeitos diferentes. `z_c` nao tira a media
  no tempo, que e o que tirava o conteudo de `z_e` (a ablacao de 2026-09-30,
  no CLAUDE.md).

`ARCHITECTURES` e a escada do POC I ao POC II: cada degrau muda um campo da CNN
do POC I, e cada mudanca foi medida (`scripts/poc2_escada.sh`); o ultimo degrau
foi medido e ficou de fora.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence

from gefx.config import ArchitectureConfig

#: Os degraus, na ordem. `poc1` e `experiments/base.yaml`; `poc2`, o encoder do
#: relatorio. As duas densas de 64 do POC I ficam (uma de 256 mais que dobrava o
#: conteudo legivel em `z_e`), e `poc2_fundo` (quatro blocos, filtros 32, 64, 96,
#: 128) foi medido e nao entrou; os numeros estao no CLAUDE.md.
ARCHITECTURES: Dict[str, ArchitectureConfig] = {"poc1": ArchitectureConfig()}
ARCHITECTURES["poc1_media"] = replace(ARCHITECTURES["poc1"], time_pool="mean")
ARCHITECTURES["poc2"] = replace(ARCHITECTURES["poc1_media"], n_filters=32)
ARCHITECTURES["poc2_fundo"] = replace(ARCHITECTURES["poc2"], n_conv=4)


def _conv_blocks(inputs, config: ArchitectureConfig):
    """Os blocos convolucionais, comuns as cabecas."""
    from keras import layers

    x = inputs
    for index in range(config.n_conv):
        x = layers.Conv2D(config.n_filters * (index + 1), kernel_size=config.kernel_size,
                          activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        if index:
            x = layers.Dropout(config.dropout)(x)
    return x


def _dense_stack(x, config: ArchitectureConfig, time_pool: str):
    """A reducao do eixo do tempo e as `n_full - 1` densas de uma cabeca."""
    from keras import layers

    if time_pool == "mean":
        # A media no eixo do tempo, (frequencia, tempo, canal) -> (frequencia, 1, canal).
        x = layers.AveragePooling2D(pool_size=(1, x.shape[2]))(x)
    x = layers.Flatten()(x)
    for _ in range(config.n_full - 1):
        x = layers.Dense(config.n_nodes, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout)(x)
    return x


def build_trunk(input_dim: Sequence[int], config: ArchitectureConfig):
    """`(entrada, saida do tronco)`, para as duas cabecas."""
    from keras import layers

    inputs = layers.Input(shape=tuple(input_dim))
    return inputs, _dense_stack(_conv_blocks(inputs, config), config, config.time_pool)


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


#: As saidas do encoder, na ordem.
HEADS = ("z_e", "z_c")


def build_encoder(input_dim: Sequence[int], config: ArchitectureConfig):
    """O encoder: tronco + projecao para `z_e`, na esfera unitaria; com
    `content_dim`, tambem `z_c`, numa segunda saida."""
    from keras import layers, models

    inputs = layers.Input(shape=tuple(input_dim))
    features = _conv_blocks(inputs, config)
    z_e = layers.Dense(config.effect_dim)(_dense_stack(features, config, config.time_pool))
    z_e = layers.UnitNormalization()(z_e)
    if not config.content_dim:
        return models.Model(inputs, z_e)
    z_c = layers.Dense(config.content_dim)(_dense_stack(features, config, "flatten"))
    return models.Model(inputs, [z_e, layers.UnitNormalization()(z_c)])


def head(model, name: str = "z_e"):
    """Uma saida so do encoder, com os mesmos pesos: a busca e as sondas leem um
    codigo de cada vez."""
    from keras import models

    if name not in HEADS:
        raise KeyError(f"cabeca desconhecida: {name!r}. Ha {list(HEADS)}")
    index = HEADS.index(name)
    if index >= len(model.outputs):
        raise ValueError(f"o modelo nao tem a cabeca {name!r}")
    if len(model.outputs) == 1:
        return model
    return models.Model(model.inputs, model.outputs[index])


#: As tecnicas do POC II (`training/encoder.py`); so `REGRESSION` usa o regressor.
REGRESSION = "regressao"
TECHNIQUES = ("random_encoder", "bn_only", "supcon", REGRESSION)


def build_network(input_dim: Sequence[int], config: ArchitectureConfig, technique: str):
    """O regressor (uma saida, o drive) para `REGRESSION`, o encoder para as outras."""
    if technique == REGRESSION:
        if config.content_dim:
            raise ValueError("o regressor nao tem cabeca de conteudo (content_dim=0)")
        return build_model(input_dim, 1, config)
    return build_encoder(input_dim, config)
