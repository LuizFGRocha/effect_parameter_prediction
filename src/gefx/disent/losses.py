"""A perda do encoder: o contrastivo supervisionado (SupCon) de Khosla et al. (2020).

`sup_con_loss` e a traducao para TensorFlow do `SupConLoss` do repositorio oficial,
https://github.com/HobbitLong/SupContrast/blob/72fd9894f39023906308a21ec404b3b01527b8f5/losses.py
(BSD-2-Clause). Os passos e os comentarios em ingles seguem o original linha a
linha, para comparar lado a lado. Ficou de fora so o que nao usamos: `mask`
explicita e `contrast_mode='one'` (usamos `labels` e o modo 'all', o padrao).

A classe e o nivel de drive. Os positivos de uma ancora sao as outras vistas do
mesmo nivel, em outro conteudo e em outra implementacao (o amostrador sorteia os
dois de forma independente): e isso que tira conteudo e implementacao de `z_e`.
O uso para separar efeito de conteudo em audio segue o FXencoder de Koo et al.
(2023), que treina por contraste um codificador que guarda so os efeitos.

Os niveis entram como classes sem ordem, e ainda assim `z_e` sai ordenado (os
niveis vizinhos soam parecidos): o Rank-N-Contrast, que impoe a ordem, empatou
com ele na grade e entre os niveis, e foi removido.
"""
from __future__ import annotations

DEFAULT_TEMPERATURE = 0.07  # o padrao do SupConLoss


def sup_con_loss(features, labels, temperature: float = DEFAULT_TEMPERATURE,
                 base_temperature: float = DEFAULT_TEMPERATURE):
    """`SupConLoss.forward(features, labels)` com `contrast_mode='all'`.

    Args:
        features: hidden vector of shape [bsz, n_views, ...], ja L2-normalizado.
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    import tensorflow as tf

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = tf.reshape(features, [tf.shape(features)[0], features.shape[1], -1])

    batch_size = tf.shape(features)[0]
    labels = tf.reshape(labels, [-1, 1])
    if labels.shape[0] is not None and features.shape[0] is not None \
            and labels.shape[0] != features.shape[0]:
        raise ValueError('Num of labels does not match num of features')
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), features.dtype)

    contrast_count = features.shape[1]
    contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = tf.matmul(anchor_feature, contrast_feature,
                                    transpose_b=True) / temperature
    # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - tf.stop_gradient(logits_max)

    # tile mask
    mask = tf.tile(mask, [anchor_count, contrast_count])
    # mask-out self-contrast cases
    logits_mask = 1.0 - tf.eye(batch_size * anchor_count, dtype=features.dtype)
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point.
    mask_pos_pairs = tf.reduce_sum(mask, axis=1)
    mask_pos_pairs = tf.where(mask_pos_pairs < 1e-6, tf.ones_like(mask_pos_pairs),
                              mask_pos_pairs)
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_pos_pairs

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))

    return loss

