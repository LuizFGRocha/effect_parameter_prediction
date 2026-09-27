"""As perdas do encoder, cada uma traduzida do codigo oficial do artigo.

- `rnc_loss`: Rank-N-Contrast (Zha et al. 2023), a perda principal. O drive e
  ordenado, e ela poe essa ordem nas distancias de `z_e`.
- `sup_con_loss`: o contrastivo supervisionado (Khosla et al. 2020), que trata os
  niveis como classes sem ordem. Fica como controle: se o RnC for melhor, sai.

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


# --- Rank-N-Contrast ----------------------------------------------------------
#: O original usa 2 sobre features sem normalizar. Nosso `z_e` fica na esfera
#: (a busca e por cosseno), onde a distancia L2 vai de 0 a 2: com 2, os logits
#: ficariam em [-1, 0] e quase nao haveria contraste. 0.1 e da ordem do 0.07 do
#: SupCon, que tambem trabalha na esfera. E um desvio do original; conferir.
RNC_TEMPERATURE = 0.1


def rnc_loss(features, labels, temperature: float = RNC_TEMPERATURE):
    """`RnCLoss(label_diff='l1', feature_sim='l2').forward(features, labels)`.

    Traducao para TensorFlow do Rank-N-Contrast (Zha et al., NeurIPS 2023),
    https://github.com/kaiwenzha/Rank-N-Contrast/blob/6239bdfc42181e25d5e570a4d96aaafd04f2b573/loss.py
    (o repositorio nao declara licenca). Dois desvios, ambos de forma e nao de
    conta:

    - o original junta exatamente 2 vistas; aqui, `n_views` quaisquer, como no
      SupCon (usamos 1: as vistas sao linhas diferentes do batch);
    - o laco `for k in range(n - 1)` virou uma conta com um eixo a mais
      ([n, n, n]), porque o tamanho do batch nao e fixo dentro do `tf.function`.
      O teste compara com o laco transcrito em NumPy.

    Para cada ancora i e cada outra amostra j (o "positivo"), os negativos sao
    as amostras k cujo rotulo esta ao menos tao longe do de i quanto o de j.
    Com j do mesmo nivel que i, os negativos sao todos: e o termo do SupCon.

    Args:
        features: [bsz, n_views, feat_dim].
        labels: [bsz, label_dim] (ou [bsz]).
    Returns:
        A loss scalar.
    """
    import tensorflow as tf

    n_views = features.shape[1]
    features = tf.concat(tf.unstack(features, axis=1), axis=0)  # [n_views*bs, feat_dim]
    labels = tf.cast(tf.reshape(labels, [tf.shape(labels)[0], -1]), features.dtype)
    labels = tf.tile(labels, [n_views, 1])  # [n_views*bs, label_dim]

    # LabelDifference('l1')
    label_diffs = tf.reduce_sum(tf.abs(labels[:, None, :] - labels[None, :, :]), axis=-1)
    # FeatureSimilarity('l2'): -||f_i - f_j||. A raiz tem gradiente infinito em 0
    # (a diagonal); o `where` duplo a evita sem mudar nenhum valor.
    squared = tf.reduce_sum(tf.square(features[:, None, :] - features[None, :, :]), axis=-1)
    positive = squared > 0.0
    distance = tf.where(positive, tf.sqrt(tf.where(positive, squared, tf.ones_like(squared))),
                        tf.zeros_like(squared))
    logits = -distance / temperature
    logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
    logits = logits - tf.stop_gradient(logits_max)
    exp_logits = tf.exp(logits)

    n = tf.shape(logits)[0]  # n = n_views*bs

    # remove diagonal
    off_diagonal = 1.0 - tf.eye(n, dtype=logits.dtype)  # [i, j]: j != i

    # neg_mask[i, j, k] = label_diffs[i, k] >= label_diffs[i, j], com k != i
    neg_mask = tf.cast(label_diffs[:, None, :] >= label_diffs[:, :, None], logits.dtype)
    neg_mask = neg_mask * off_diagonal[:, None, :]
    denominator = tf.reduce_sum(neg_mask * exp_logits[:, None, :], axis=-1)  # [i, j]
    # Na diagonal (j == i) o denominador nunca e zero, mas o termo e descartado.
    pos_log_probs = logits - tf.math.log(denominator)
    n_pairs = tf.cast(n * (n - 1), logits.dtype)
    loss = -tf.reduce_sum(pos_log_probs * off_diagonal) / n_pairs

    return loss
