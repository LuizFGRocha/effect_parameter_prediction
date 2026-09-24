"""As duas perdas do encoder e a soma que as compoe.

- contrastivo supervisionado (Khosla et al. 2020) sobre `z_e`, com a
  configuracao como classe: e ele que tira conteudo e implementacao do codigo;
- regressao auxiliar dos niveis a partir de `z_e`, que da a ordem dos eixos.
"""
from __future__ import annotations

from typing import Any, Dict

EPSILON = 1e-8

DEFAULT_TEMPERATURE = 0.07  # SimCLR/SupCon


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


def aux_regression(prediction, target):
    """Erro quadratico dos niveis normalizados: para o contrastivo as 40
    configuracoes sao classes sem vizinhanca."""
    import tensorflow as tf

    return tf.reduce_mean(tf.square(prediction - target))


def total_loss(
    z_e, config_label, aux_prediction, aux_target,
    temperature: float = DEFAULT_TEMPERATURE, aux_weight: float = 1.0,
) -> Dict[str, Any]:
    """`contrastive + aux_weight * aux_regression`, com cada termo para o historico."""
    contrastive = supervised_contrastive(z_e, config_label, temperature)
    aux = aux_regression(aux_prediction, aux_target)
    return {
        "contrastive": contrastive,
        "aux_regression": aux,
        "total": contrastive + aux_weight * aux,
    }
