"""As perdas e o registro que as compoe.

- contrastivo supervisionado (Khosla et al. 2020) sobre `z_e`, com a
  configuracao como classe;
- reversao de gradiente (Ganin & Lempitsky 2015) contra classificadores de
  implementacao e conteudo em `z_e` e de configuracao em `z_c`;
- ortogonalidade entre `z_e` e `z_c`;
- reconstrucao e troca de codigos (fase 2).

A perda total e a soma ponderada das entradas de `LOSS_REGISTRY`. Todo termo
recebe o mesmo contexto: um dicionario com codigos, logits e rotulos do batch.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional

EPSILON = 1e-8

DEFAULT_TEMPERATURE = 0.07  # SimCLR/SupCon
RAMP_GAMMA = 10.0  # Ganin & Lempitsky


# --- reversao a gradiente -----------------------------------------------------
#: Construida na primeira chamada: o pacote so importa TF dentro das funcoes.
_REVERSE: Optional[Any] = None


def _reverse_op():
    global _REVERSE
    if _REVERSE is None:
        import tensorflow as tf

        @tf.custom_gradient
        def reverse(x, lam):
            def grad(upstream):
                return -lam * upstream, None

            return tf.identity(x), grad

        _REVERSE = reverse
    return _REVERSE


def gradient_reversal(x, lam):
    """Identidade na ida, gradiente multiplicado por `-lam` na volta."""
    import tensorflow as tf

    return _reverse_op()(x, tf.cast(lam, x.dtype))


def lambda_ramp(progress: float, gamma: float = RAMP_GAMMA) -> float:
    """Rampa `2 / (1 + exp(-gamma * p)) - 1`, com `p` em [0, 1] (Ganin & Lempitsky).

    Comeca suave porque, com o adversario a todo peso desde o inicio, o encoder
    aprende a apagar `z_e` inteiro.
    """
    p = float(min(max(progress, 0.0), 1.0))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# --- contrastivo supervisionado ----------------------------------------------
def supervised_contrastive(z, labels, temperature: float = DEFAULT_TEMPERATURE):
    """L_out^sup de Khosla et al. (2020), com `z` ja L2-normalizado.

    Os positivos de uma ancora sao todas as outras entradas com a mesma
    configuracao, por isso o batch precisa ser balanceado por classe. Ancoras
    sem positivo ficam fora da media.
    """
    import tensorflow as tf

    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    batch = tf.shape(z)[0]

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    logits = logits - tf.stop_gradient(tf.reduce_max(logits, axis=1, keepdims=True))

    not_self = 1.0 - tf.eye(batch, dtype=logits.dtype)
    same = tf.cast(tf.equal(labels[:, None], labels[None, :]), logits.dtype)
    positives = same * not_self

    log_denominator = tf.math.log(
        tf.reduce_sum(tf.exp(logits) * not_self, axis=1) + EPSILON
    )
    log_probability = logits - log_denominator[:, None]

    n_positives = tf.reduce_sum(positives, axis=1)
    per_anchor = tf.reduce_sum(positives * log_probability, axis=1) / tf.maximum(
        n_positives, 1.0
    )
    valid = tf.cast(n_positives > 0.0, logits.dtype)
    return -tf.reduce_sum(per_anchor * valid) / tf.maximum(tf.reduce_sum(valid), 1.0)


# --- independencia entre os dois codigos --------------------------------------
def orthogonality(z_e, z_c):
    """Correlacao cruzada media ao quadrado entre `z_e` e `z_c` no batch.

    Em [0, 1] e independente da escala dos codigos. Mede so independencia linear.
    """
    import tensorflow as tf

    z_e = tf.cast(z_e, tf.float32)
    z_c = tf.cast(z_c, tf.float32)
    n = tf.cast(tf.shape(z_e)[0], tf.float32)

    def standardize(z):
        mean, variance = tf.nn.moments(z, axes=[0], keepdims=True)
        return (z - mean) / tf.sqrt(variance + EPSILON)

    cross = tf.matmul(standardize(z_e), standardize(z_c), transpose_a=True) / n
    return tf.reduce_mean(tf.square(cross))


# --- fase 2: reconstrucao e troca de codigos ----------------------------------
def loss_recon(ctx: Mapping[str, Any]):
    """Reconstrucao do proprio espectro medio, a partir dos codigos da ancora."""
    import tensorflow as tf

    return tf.reduce_mean(tf.square(ctx["recon_prediction"] - ctx["recon_target"]))


def loss_swap_recon(ctx: Mapping[str, Any]):
    """Troca de codigos: `z_e` do doador + `z_c` da ancora -> espectro do alvo.

    Supervisionada: na grade totalmente cruzada o alvo
    `x[conteudo(a), configuracao(b), implementacao(a)]` existe em disco.
    """
    import tensorflow as tf

    return tf.reduce_mean(tf.square(ctx["swap_prediction"] - ctx["swap_target_spec"]))


# --- o registro ---------------------------------------------------------------
LossFn = Callable[[Mapping[str, Any]], Any]


def _cross_entropy(logits, labels):
    import tensorflow as tf

    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, [-1]), tf.int32), logits=logits
        )
    )


def loss_contrastive(ctx: Mapping[str, Any]):
    return supervised_contrastive(
        ctx["z_e"], ctx["config_label"], ctx.get("temperature", DEFAULT_TEMPERATURE)
    )


def loss_aux_regression(ctx: Mapping[str, Any]):
    """Regressao dos niveis normalizados a partir de `z_e`: da a ordem dos eixos,
    que para o contrastivo sao classes sem vizinhanca."""
    import tensorflow as tf

    return tf.reduce_mean(tf.square(ctx["aux_prediction"] - ctx["aux_target"]))


def loss_adversary_arm(ctx: Mapping[str, Any]):
    return _cross_entropy(ctx["adv_arm_logits"], ctx["arm_label"])


def loss_adversary_content(ctx: Mapping[str, Any]):
    return _cross_entropy(ctx["adv_content_logits"], ctx["content_label"])


def loss_adversary_config(ctx: Mapping[str, Any]):
    """Adversario de configuracao em `z_c`, o simetrico dos outros dois."""
    return _cross_entropy(ctx["adv_config_logits"], ctx["config_label"])


def loss_orthogonality(ctx: Mapping[str, Any]):
    return orthogonality(ctx["z_e"], ctx["z_c"])


LOSS_REGISTRY: Dict[str, LossFn] = {
    "contrastive": loss_contrastive,
    "aux_regression": loss_aux_regression,
    "adversary_arm": loss_adversary_arm,
    "adversary_content": loss_adversary_content,
    "adversary_config": loss_adversary_config,
    "orthogonality": loss_orthogonality,
    "recon": loss_recon,
    "swap_recon": loss_swap_recon,
}


def total_loss(
    ctx: Mapping[str, Any], weights: Mapping[str, float]
) -> Dict[str, Any]:
    """Soma ponderada dos termos pedidos, mais cada termo cru para o historico.

    Peso zero pula o termo, que entao nem e calculado.
    """
    import tensorflow as tf

    unknown = set(weights) - set(LOSS_REGISTRY)
    if unknown:
        raise KeyError(
            f"perdas desconhecidas: {sorted(unknown)}. "
            f"Registradas: {sorted(LOSS_REGISTRY)}"
        )
    parts: Dict[str, Any] = {}
    total = tf.constant(0.0, dtype=tf.float32)
    for name, weight in weights.items():
        if weight == 0.0:
            continue
        value = tf.cast(LOSS_REGISTRY[name](ctx), tf.float32)
        parts[name] = value
        total = total + weight * value
    parts["total"] = total
    return parts
