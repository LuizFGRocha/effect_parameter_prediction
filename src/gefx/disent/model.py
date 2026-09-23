"""Encoder bifurcado do POC II e o controle beta-VAE.

O contrato que o resto do pacote enxerga e `z_e, z_c = model.encode(x)`:

- `z_e` e o codigo de efeito, L2-normalizado porque a busca e por cosseno. So
  ele entra no catalogo.
- `z_c` e o codigo de conteudo, o destino do que deve sair de `z_e`.

O tronco tira a media so no eixo do tempo: o ajuste de distorcao e estacionario
num segmento de dois segundos, e o que varia no tempo e o que foi tocado.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

from gefx.disent.losses import gradient_reversal

DEFAULT_FILTERS: Tuple[int, ...] = (32, 64, 96, 128)


@dataclass
class EncoderConfig:
    """Forma da rede. Vai inteira para o `run.json`, como no POC I."""

    input_shape: Tuple[int, int, int] = (256, 173, 1)
    filters: Tuple[int, ...] = DEFAULT_FILTERS
    kernel_size: int = 3
    trunk_units: int = 256
    dropout: float = 0.2
    effect_dim: int = 32
    content_dim: int = 64
    #: Experimento da assimetria dos adversarios: poe `z_c` na esfera tambem.
    normalize_content: bool = False
    #: BatchNorm na entrada dos adversarios, sem tocar nos codigos: as
    #: coordenadas de `z_e` sao pequenas demais para uma cabeca crua ler.
    adversary_input_norm: bool = False
    #: Camadas ocultas do decoder da fase 2; vazio = sem decoder. Fica aqui
    #: porque e esta dataclass que o `run.json` grava para reconstruir o modelo.
    decoder_units: Tuple[int, ...] = ()
    adversary_units: int = 128

    def as_dict(self) -> Dict[str, object]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EncoderConfig":
        # Chave ausente cai no padrao: execucoes antigas continuam recarregando.
        return cls(**{k: tuple(v) if isinstance(v, list) else v  # type: ignore[arg-type]
                      for k, v in data.items()})


@dataclass
class HeadConfig:
    """Quantas classes cada cabeca enxerga, contadas no recorte de treino."""

    n_arms: int
    n_contents: int
    n_configs: int
    n_aux: int = 2  # drive e tone, normalizados em [0, 1]

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def _mean_over_time(tensor):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia, canal)."""
    import tensorflow as tf

    return tf.reduce_mean(tensor, axis=2)


def build_trunk(config: EncoderConfig):
    """Conv2D -> BN -> pool empilhados, media no tempo, denso. Um `keras.Model`."""
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

    x = layers.Lambda(
        _mean_over_time,
        output_shape=lambda shape: (shape[0], shape[1], shape[3]),
        name="time_pool",
    )(x)
    x = layers.Flatten(name="flat")(x)
    x = layers.Dense(config.trunk_units, activation="relu", name="trunk_dense")(x)
    x = layers.BatchNormalization(name="trunk_bn")(x)
    x = layers.Dropout(config.dropout, name="trunk_drop")(x)
    return models.Model(inputs, x, name="trunk")


def build_encoder(config: EncoderConfig):
    """Tronco + a bifurcacao. Saidas: `z_e` (L2-normalizado) e `z_c`."""
    from keras import layers, models

    trunk = build_trunk(config)
    features = trunk.output
    z_e = layers.Dense(config.effect_dim, name="effect_dense")(features)
    z_e = layers.UnitNormalization(name="effect_code")(z_e)
    # O nome `content_code` fica na Dense para os pesos gravados recarregarem.
    z_c = layers.Dense(config.content_dim, name="content_code")(features)
    if config.normalize_content:
        z_c = layers.UnitNormalization(name="content_norm")(z_c)
    return models.Model(trunk.input, [z_e, z_c], name="encoder")


class DisentModel:
    """Encoder mais as cabecas de treino, com a reversao de gradiente no meio.

    Um recipiente de submodelos `keras.Model`, nao uma `keras.Model`: o laco de
    treino e customizado.
    """

    def __init__(self, encoder_config: EncoderConfig, head_config: HeadConfig) -> None:
        from keras import layers, models

        self.encoder_config = encoder_config
        self.head_config = head_config

        self.encoder = build_encoder(encoder_config)

        effect_in = layers.Input(shape=(encoder_config.effect_dim,), name="aux_in")
        self.aux_head = models.Model(
            effect_in,
            layers.Dense(head_config.n_aux, activation="sigmoid", name="aux_out")(
                layers.Dense(64, activation="relu", name="aux_hidden")(effect_in)
            ),
            name="aux_head",
        )

        def adversary(name: str, input_dim: int, n_classes: int):
            return _adversary_on(f"adv_{name}", input_dim, encoder_config.adversary_units,
                                 n_classes, encoder_config.dropout,
                                 encoder_config.adversary_input_norm)

        self.adversaries = {
            "arm": adversary("arm", encoder_config.effect_dim, head_config.n_arms),
            "content": adversary("content", encoder_config.effect_dim, head_config.n_contents),
            "config": adversary("config", encoder_config.content_dim, head_config.n_configs),
        }

        self.decoder = (
            build_decoder(encoder_config, head_config.n_arms)
            if encoder_config.decoder_units else None
        )

    def decode(self, z_e, z_c, arm_onehot, training: bool = False):
        """Espectro medio reconstruido; para trocar codigos, passe blocos de linhas diferentes."""
        if self.decoder is None:
            raise ValueError(
                "esta execucao nao tem decoder. Use `decoder_units` nao vazio "
                "no EncoderConfig (a tecnica `swap` ja o faz)."
            )
        import tensorflow as tf

        # O Keras recusa misturar tensores e arrays numpy na mesma chamada.
        entradas = [tf.convert_to_tensor(v, dtype=tf.float32)
                    for v in (z_e, z_c, arm_onehot)]
        return self.decoder(entradas, training=training)

    def encode(self, x, training: bool = False):
        """`(z_e, z_c)`."""
        z_e, z_c = self.encoder(x, training=training)
        return z_e, z_c

    def __call__(self, x, lam: float = 0.0, training: bool = True) -> Dict[str, object]:
        """Um passo para frente completo, ja com o contexto que as perdas esperam."""
        z_e, z_c = self.encode(x, training=training)
        return {
            "z_e": z_e,
            "z_c": z_c,
            "aux_prediction": self.aux_head(z_e, training=training),
            "adv_arm_logits": self.adversaries["arm"](
                gradient_reversal(z_e, lam), training=training
            ),
            "adv_content_logits": self.adversaries["content"](
                gradient_reversal(z_e, lam), training=training
            ),
            "adv_config_logits": self.adversaries["config"](
                gradient_reversal(z_c, lam), training=training
            ),
        }

    @property
    def trainable_variables(self):
        return [v for part in self.parts().values() for v in part.trainable_variables]

    def parts(self) -> Dict[str, object]:
        out: Dict[str, object] = {"encoder": self.encoder, "aux_head": self.aux_head}
        out.update({f"adv_{name}": head for name, head in self.adversaries.items()})
        if self.decoder is not None:
            out["decoder"] = self.decoder
        return out

    def save_weights(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, part in self.parts().items():
            part.save_weights(directory / f"{name}.weights.h5")

    def load_weights(self, directory: Path) -> None:
        directory = Path(directory)
        for name, part in self.parts().items():
            path = directory / f"{name}.weights.h5"
            if path.exists():
                part.load_weights(path)
            elif name == "encoder":  # as cabecas podem faltar em execucoes antigas
                raise FileNotFoundError(f"pesos do encoder ausentes em {directory}")


def build_decoder(config: EncoderConfig, n_arms: int):
    """`[z_e | z_c] + implementacao` -> espectro medio no tempo (fase 2).

    O alvo e o espectro medio porque o tronco ja tirou o tempo antes do gargalo.
    A implementacao entra como one-hot porque o alvo da troca a fixa, e os
    codigos a carregam so em parte.
    """
    from keras import layers, models

    z_e = layers.Input(shape=(config.effect_dim,), name="dec_z_e")
    z_c = layers.Input(shape=(config.content_dim,), name="dec_z_c")
    arm = layers.Input(shape=(n_arms,), name="dec_arm")
    x = layers.Concatenate(name="dec_concat")([z_e, z_c, arm])
    for index, width in enumerate(config.decoder_units):
        x = layers.Dense(width, activation="relu", name=f"dec_hidden{index}")(x)
    bins = int(config.input_shape[0])
    out = layers.Dense(bins, name="dec_out")(x)
    return models.Model([z_e, z_c, arm], out, name="decoder")


def _adversary_on(name: str, input_dim: int, units: int, n_classes: int,
                  dropout: float, input_norm: bool = False):
    """Cabeca adversaria rasa (duas camadas, como no DANN), logits crus."""
    from keras import layers, models

    inputs = layers.Input(shape=(input_dim,), name=f"{name}_in")
    x = layers.BatchNormalization(name=f"{name}_bn")(inputs) if input_norm else inputs
    x = layers.Dense(units, activation="relu", name=f"{name}_hidden")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return models.Model(inputs, layers.Dense(n_classes, name=f"{name}_logits")(x), name=name)


# --- controle que se espera falhar -------------------------------------------
class BetaVAE:
    """beta-VAE (Higgins et al. 2017) sobre o mesmo tronco: controle negativo.

    O codigo tem a largura de `z_e + z_c`; as `effect_dim` primeiras dimensoes
    fazem o papel de `z_e`.
    """

    def __init__(self, encoder_config: EncoderConfig, beta: float = 4.0) -> None:
        from keras import layers, models

        self.encoder_config = encoder_config
        self.beta = float(beta)
        self.latent_dim = encoder_config.effect_dim + encoder_config.content_dim

        trunk = build_trunk(encoder_config)
        stats = layers.Dense(2 * self.latent_dim, name="posterior")(trunk.output)
        self.encoder = models.Model(trunk.input, stats, name="vae_encoder")

        height, width, channels = encoder_config.input_shape
        code = layers.Input(shape=(self.latent_dim,), name="z")
        small_h, small_w = height // 16, width // 16
        x = layers.Dense(small_h * small_w * 32, activation="relu", name="dec_dense")(code)
        x = layers.Reshape((small_h, small_w, 32), name="dec_reshape")(x)
        for index, filters in enumerate((64, 48, 32, 16)):
            x = layers.Conv2DTranspose(
                filters, 3, strides=2, padding="same", activation="relu",
                name=f"deconv{index}",
            )(x)
        x = layers.Conv2D(channels, 3, padding="same", name="dec_out")(x)
        x = layers.Resizing(height, width, name="dec_resize")(x)
        self.decoder = models.Model(code, x, name="vae_decoder")

    def encode(self, x, training: bool = False):
        """`(z_e, z_c)` como no `DisentModel`; fora do treino, a media da posterior."""
        import tensorflow as tf

        stats = self.encoder(x, training=training)
        mean, log_variance = tf.split(stats, 2, axis=1)
        if training:
            noise = tf.random.normal(tf.shape(mean))
            code = mean + tf.exp(0.5 * log_variance) * noise
        else:
            code = mean
        split = self.encoder_config.effect_dim
        return tf.math.l2_normalize(code[:, :split], axis=1), code[:, split:]

    def losses(self, x, training: bool = True) -> Dict[str, object]:
        import tensorflow as tf

        stats = self.encoder(x, training=training)
        mean, log_variance = tf.split(stats, 2, axis=1)
        code = mean + tf.exp(0.5 * log_variance) * tf.random.normal(tf.shape(mean))
        reconstruction = self.decoder(code, training=training)
        recon = tf.reduce_mean(tf.square(reconstruction - x))
        kl = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1)
        )
        return {"reconstruction": recon, "kl": kl, "total": recon + self.beta * kl}

    def parts(self) -> Dict[str, object]:
        return {"encoder": self.encoder, "decoder": self.decoder}

    trainable_variables = DisentModel.trainable_variables
    save_weights = DisentModel.save_weights
    load_weights = DisentModel.load_weights
