"""As perdas do encoder, o SupCon e o RnC traduzidos dos repositorios oficiais.

O que estes testes protegem sao propriedades, nao valores: a perda tem de descer
quando as classes se separam e tratar ancoras sem positivo como o original.
Fixar valores numericos aqui amarraria o treino a uma versao do TensorFlow sem
dizer nada sobre estar certo.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from gefx.disent.losses import rnc_loss, sup_con_loss


def _normalize(x):
    return tf.math.l2_normalize(tf.constant(np.asarray(x, dtype=np.float32)), axis=-1)


def _one_view(x):
    return _normalize(x)[:, None, :]


def test_the_loss_is_lower_when_the_classes_are_separated():
    labels = tf.constant([0, 0, 1, 1])
    separated = _one_view([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.02, 1.0]])
    mixed = _one_view([[1.0, 0.0], [0.0, 1.0], [1.0, 0.02], [0.02, 1.0]])
    assert float(sup_con_loss(separated, labels)) < float(sup_con_loss(mixed, labels))


def test_an_anchor_without_a_positive_counts_as_zero_like_the_original():
    """O original divide por 1 onde nao ha positivo: a ancora entra na media com
    termo zero. No treino nao acontece, porque o batch e balanceado por nivel."""
    z = _one_view([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    with_orphan = float(sup_con_loss(z, tf.constant([0, 0, 1])))
    assert np.isfinite(with_orphan)
    # A mesma conta a mao: as duas ancoras com positivo, somadas, sobre 3.
    logits = np.asarray(tf.squeeze(z, 1)) @ np.asarray(tf.squeeze(z, 1)).T / 0.07
    terms = []
    for i, j in ((0, 1), (1, 0)):
        others = [k for k in range(3) if k != i]
        log_denominator = np.log(np.exp(logits[i, others] - logits[i].max()).sum())
        terms.append(-(logits[i, j] - logits[i].max() - log_denominator))
    assert with_orphan == pytest.approx(sum(terms) / 3, rel=1e-4)


def test_a_batch_with_no_positive_at_all_is_zero_not_nan():
    value = float(sup_con_loss(_one_view([[1.0, 0.0], [0.0, 1.0]]), tf.constant([0, 1])))
    assert value == pytest.approx(0.0)


def test_two_views_equal_the_same_rows_stacked_as_one_view():
    """`[bsz, 2, d]` e o mesmo que `[2*bsz, 1, d]` com os rotulos repetidos: e
    assim que o original desempilha as vistas."""
    rng = np.random.default_rng(2)
    z = _normalize(rng.normal(size=(6, 2, 4)))
    labels = tf.constant([0, 0, 1, 1, 2, 2])
    stacked = tf.concat([z[:, 0], z[:, 1]], axis=0)[:, None, :]
    assert float(sup_con_loss(z, labels)) == pytest.approx(
        float(sup_con_loss(stacked, tf.concat([labels, labels], axis=0))), rel=1e-5)


def test_features_without_a_views_axis_are_refused():
    with pytest.raises(ValueError, match="n_views"):
        sup_con_loss(_normalize([[1.0, 0.0], [0.0, 1.0]]), tf.constant([0, 1]))


# --- Rank-N-Contrast ----------------------------------------------------------
def _rnc_loop(features, labels, t):
    """O `RnCLoss.forward` original transcrito em NumPy, com o laco em k."""
    features = np.concatenate([features[:, 0], features[:, 1]], axis=0)
    labels = np.tile(labels, (2, 1))
    label_diffs = np.abs(labels[:, None, :] - labels[None, :, :]).sum(-1)
    logits = -np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1) / t
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    n = logits.shape[0]
    keep = ~np.eye(n, dtype=bool)
    logits = logits[keep].reshape(n, n - 1)
    exp_logits = exp_logits[keep].reshape(n, n - 1)
    label_diffs = label_diffs[keep].reshape(n, n - 1)
    loss = 0.0
    for k in range(n - 1):
        pos_logits = logits[:, k]
        pos_label_diffs = label_diffs[:, k]
        neg_mask = (label_diffs >= pos_label_diffs[:, None]).astype(float)
        pos_log_probs = pos_logits - np.log((neg_mask * exp_logits).sum(-1))
        loss += -(pos_log_probs / (n * (n - 1))).sum()
    return loss


def test_rnc_matches_the_original_loop():
    rng = np.random.default_rng(3)
    features = np.asarray(_normalize(rng.normal(size=(6, 2, 4))), dtype=np.float64)
    labels = rng.integers(0, 4, size=(6, 1)).astype(np.float64)
    ours = float(rnc_loss(tf.constant(features), tf.constant(labels), temperature=0.5))
    assert ours == pytest.approx(_rnc_loop(features, labels, 0.5), rel=1e-6)


def test_rnc_prefers_an_embedding_ordered_by_the_level():
    """Mesmos pontos, mesmos aglomerados: so muda a ordem deles ao longo da reta.
    O SupCon nao ve diferenca; o RnC tem de ver."""
    labels = tf.constant([0, 0, 1, 1, 2, 2])
    position = {"ordered": [0.0, 1.0, 2.0], "shuffled": [0.0, 2.0, 1.0]}
    losses = {}
    for name, where in position.items():
        angles = np.repeat(np.asarray(where) * 0.4, 2) + np.tile([0.0, 0.02], 3)
        z = _one_view(np.stack([np.cos(angles), np.sin(angles)], axis=1))
        losses[name] = (float(rnc_loss(z, labels)), float(sup_con_loss(z, labels)))
    assert losses["ordered"][0] < losses["shuffled"][0]
    assert losses["ordered"][1] == pytest.approx(losses["shuffled"][1], rel=1e-4)


def test_rnc_has_finite_gradients_with_coincident_points():
    """Duas vistas identicas: a distancia zero fora da diagonal nao pode dar NaN."""
    z = tf.Variable(np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    with tf.GradientTape() as tape:
        loss = rnc_loss(z[:, None, :], tf.constant([0, 0, 1]))
    assert np.all(np.isfinite(np.asarray(tape.gradient(loss, z))))
