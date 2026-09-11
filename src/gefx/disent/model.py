"""Encoder bifurcado do POC II e o controle beta-VAE.

O contrato que o resto do pacote enxerga e um so:

    z_e, z_c = model.encode(x)

`z_e` (32-d, L2-normalizado) e o **codigo de efeito** -- e ele, e so ele, que vai
para o catalogo e para a busca. `z_c` (64-d) e o **codigo de conteudo**: existe
para dar onde o resto morar, ja que sem um destino alternativo a informacao de
conteudo nao sai de `z_e`, so se esconde. Nenhuma cabeca faz parte desse
contrato, o que e a segunda costura do compromisso da fase 2 (memoria
`poc2-extensao-decoder-troca-de-codigos`): o decoder pluga em `encode`, nao no
modelo de treino.

`z_e` normalizado nao e detalhe: a recuperacao e por vizinho mais proximo e o
contrastivo opera em produto interno. Deixar a norma livre daria a rede um jeito
barato de mexer na perda sem mudar a direcao, que e a unica coisa que a busca le.

**Reducao no tempo, nao no espaco inteiro.** Depois das convolucoes a media e
tirada so no eixo temporal, preservando o de frequencia. E uma afirmacao sobre o
problema: dentro de um segmento de dois segundos o ajuste de distorcao e
estacionario e o que muda ao longo do tempo e o que foi tocado. Mediar o tempo
descarta conteudo de graca; mediar a frequencia descartaria justamente o efeito.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

from gefx.disent.losses import gradient_reversal

#: Larguras dos blocos convolucionais do tronco.
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
    #: L2-normalizar tambem o `z_c`. O padrao e False -- e assim que todas as
    #: execucoes publicadas rodaram, e assim que o desenho pede (so o `z_e`
    #: precisa viver na esfera, porque a busca e por cosseno). Existe para um
    #: experimento: dos tres adversarios, os dois que ficaram no acaso o treino
    #: inteiro penduram no bloco normalizado e o unico que se mexeu pendura no
    #: livre. Ligar isto poe o adversario de configuracao na mesma condicao dos
    #: outros dois.
    normalize_content: bool = False
    #: Padronizar a entrada das cabecas adversarias (BatchNorm), sem tocar nos
    #: codigos. O `z_e` e L2-normalizado, entao cada coordenada dele vive em
    #: torno de 1/sqrt(32) ~ 0,18: as pre-ativacoes da cabeca nascem minusculas e
    #: ela nao consegue ler o que uma regressao logistica padronizada le. Medido
    #: post-hoc: a mesma cabeca sobe de 17,5% para 39,1% no conteudo quando a
    #: entrada e padronizada. Isto testa se era esse o motivo de os adversarios
    #: de `z_e` nunca saírem do acaso -- e nao perturba a representacao, que e o
    #: confundidor de `normalize_content`.
    adversary_input_norm: bool = False
    #: Larguras das camadas ocultas do decoder da fase 2. Vazio = sem decoder, e
    #: e o padrao: toda execucao publicada rodou sem ele, e construi-lo sempre
    #: acrescentaria parametros a modelos que nao o usam. Fica no `EncoderConfig`
    #: -- e nao num argumento do `DisentModel` -- porque e o `run.json` que
    #: reconstroi o modelo nos diagnosticos, e ele grava esta dataclass.
    decoder_units: Tuple[int, ...] = ()
    adversary_units: int = 128

    def as_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["input_shape"] = list(self.input_shape)
        out["filters"] = list(self.filters)
        out["decoder_units"] = list(self.decoder_units)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EncoderConfig":
        data = dict(data)
        data["input_shape"] = tuple(data["input_shape"])  # type: ignore[arg-type]
        data["filters"] = tuple(data["filters"])  # type: ignore[arg-type]
        # Execucoes gravadas antes da fase 2 nao tem a chave; ausente e "sem
        # decoder", que e o que elas de fato eram.
        data["decoder_units"] = tuple(data.get("decoder_units", ()))  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]


@dataclass
class HeadConfig:
    """Quantas classes cada cabeca enxerga. Sai do recorte de treino, nao da grade.

    Os adversarios sao descartaveis: existem so para empurrar o encoder. Sao
    portanto as unicas partes do modelo cujo tamanho depende do split -- o numero
    de conteudos de treino nao tem por que ser o mesmo em outro recorte, e nada
    fora do treino consulta essas cabecas.
    """

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
    # O nome `content_code` fica na Dense, e nao na normalizacao: os pesos ja
    # gravados sao recarregados por estrutura, e renomear a camada quebraria a
    # releitura de toda execucao publicada.
    z_c = layers.Dense(config.content_dim, name="content_code")(features)
    if config.normalize_content:
        z_c = layers.UnitNormalization(name="content_norm")(z_c)
    return models.Model(trunk.input, [z_e, z_c], name="encoder")


class DisentModel:
    """Encoder mais as cabecas de treino, com a reversao de gradiente no meio.

    Nao e uma `keras.Model`: e um recipiente. O laco de treino e customizado
    (varias perdas, um lambda que muda a cada passo) e uma `Model.fit` nao daria
    nada de graca aqui -- daria so uma camada de indirecao entre a perda e o
    gradiente. Os submodelos continuam sendo `keras.Model`, entao pesos, resumo e
    serializacao seguem os do Keras.
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

        units = encoder_config.adversary_units
        drop = encoder_config.dropout
        norm = encoder_config.adversary_input_norm
        self.adversaries = {
            "arm": _adversary_on(
                "adv_arm", encoder_config.effect_dim, units, head_config.n_arms, drop, norm
            ),
            "content": _adversary_on(
                "adv_content", encoder_config.effect_dim, units, head_config.n_contents,
                drop, norm
            ),
            "config": _adversary_on(
                "adv_config", encoder_config.content_dim, units, head_config.n_configs,
                drop, norm
            ),
        }

        self.decoder = (
            build_decoder(encoder_config, head_config.n_arms)
            if encoder_config.decoder_units else None
        )

    # --- o contrato ----------------------------------------------------------
    def decode(self, z_e, z_c, arm_onehot, training: bool = False):
        """Espectro medio reconstruido. E aqui que a troca de codigos acontece:
        quem chama decide de qual linha vem cada bloco."""
        if self.decoder is None:
            raise ValueError(
                "esta execucao nao tem decoder. Use `decoder_units` nao vazio "
                "no EncoderConfig (a tecnica `swap` ja o faz)."
            )
        import tensorflow as tf

        # O Keras recusa uma chamada que misture tensores e arrays; quem chama
        # daqui vem tanto do laco de treino (tensores) quanto de um diagnostico
        # em numpy, e a conversao aqui evita que cada chamador tenha de lembrar.
        entradas = [tf.convert_to_tensor(v, dtype=tf.float32)
                    for v in (z_e, z_c, arm_onehot)]
        return self.decoder(entradas, training=training)

    def encode(self, x, training: bool = False):
        """`(z_e, z_c)`. E so isto que a recuperacao e a fase 2 usam."""
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

    # --- pesos ---------------------------------------------------------------
    @property
    def trainable_variables(self):
        out = list(self.encoder.trainable_variables)
        out += list(self.aux_head.trainable_variables)
        for head in self.adversaries.values():
            out += list(head.trainable_variables)
        if self.decoder is not None:
            out += list(self.decoder.trainable_variables)
        return out

    def parts(self) -> Dict[str, object]:
        # `load_weights` tolera arquivo ausente em tudo menos o encoder, entao
        # execucoes gravadas antes da fase 2 continuam recarregando.
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
            elif name == "encoder":
                raise FileNotFoundError(f"pesos do encoder ausentes em {directory}")


def build_decoder(config: EncoderConfig, n_arms: int):
    """`[z_e | z_c] + implementacao` -> espectro medio no tempo (fase 2).

    **O alvo e o espectro medio, e nao o espectrograma**, porque e o que o codigo
    pode conter: o tronco faz `time_pool` (media sobre o tempo) ANTES do gargalo,
    entao a resolucao temporal ja nao existe em `z_e` nem em `z_c`. Um decoder
    para o espectrograma inteiro so poderia inventar o eixo do tempo, e o que ele
    inventasse nao estaria vindo do codigo -- a demonstracao mediria o decoder, e
    nao a representacao. O espectro medio tambem e a grandeza com que o trabalho
    inteiro mede distorcao, entao nao e uma concessao: e o alvo certo.

    A implementacao entra como one-hot, e nao pelos codigos: o alvo da troca e
    `x[conteudo(a), configuracao(b), implementacao(a)]`, e sem condicionar em
    `arm` o problema fica mal posto (o `z_c` carrega implementacao so
    parcialmente). E a quarta costura registrada em
    `poc2-extensao-decoder-troca-de-codigos`.
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
    """Cabeca adversaria: duas camadas, logits crus, sem softmax.

    Rasa de proposito. Um adversario forte demais aprende a ler ruido e o encoder
    passa a lutar contra o ruido; um raso demais nao pressiona nada. Duas camadas
    e o meio-termo usado no DANN.
    """
    from keras import layers, models

    inputs = layers.Input(shape=(input_dim,), name=f"{name}_in")
    x = layers.BatchNormalization(name=f"{name}_bn")(inputs) if input_norm else inputs
    x = layers.Dense(units, activation="relu", name=f"{name}_hidden")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return models.Model(inputs, layers.Dense(n_classes, name=f"{name}_logits")(x), name=name)


# --- controle que se espera falhar -------------------------------------------
class BetaVAE:
    """beta-VAE (Higgins et al. 2017) sobre o mesmo tronco, sem rotulo nenhum.

    Entra como **controle negativo declarado**, nao como concorrente: Locatello
    et al. (2019) mostram que desemaranhamento nao supervisionado nao e
    identificavel sem vies indutivo, e aqui o fator dominante da variancia e o
    conteudo -- ou seja, espera-se que as dimensoes que o beta-VAE separa sejam
    as do conteudo, e que a recuperacao por configuracao nao melhore. Se ele
    ganhar, e a supervisao que esta mal usada, e isso tambem e resultado.

    Para deixar a comparacao justa, o codigo latente tem a mesma largura da
    concatenacao `z_e + z_c`, e a recuperacao usa as `effect_dim` primeiras
    dimensoes -- a escolha mais generosa possivel sem rotulo.
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
        """`(z_e, z_c)`, para casar com o contrato do `DisentModel`.

        No modo de avaliacao usa a media da posterior, nao uma amostra: amostrar
        na hora de montar catalogo introduziria ruido que nada tem a ver com o
        que o modelo aprendeu.
        """
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

    @property
    def trainable_variables(self):
        return list(self.encoder.trainable_variables) + list(self.decoder.trainable_variables)

    def parts(self) -> Dict[str, object]:
        return {"encoder": self.encoder, "decoder": self.decoder}

    save_weights = DisentModel.save_weights
    load_weights = DisentModel.load_weights
