"""As perdas do encoder.

O que estes testes protegem sao propriedades, nao valores: o contrastivo tem de
descer quando as classes se separam e ignorar ancoras sem positivo. Fixar valores
numericos aqui amarraria o treino a uma versao do TensorFlow sem dizer nada
sobre estar certo.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from gefx.disent import losses


def _normalize(x):
    return tf.math.l2_normalize(tf.constant(np.asarray(x, dtype=np.float32)), axis=1)


def test_contrastive_is_lower_when_the_classes_are_separated():
    labels = tf.constant([0, 0, 1, 1])
    separated = _normalize([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.02, 1.0]])
    mixed = _normalize([[1.0, 0.0], [0.0, 1.0], [1.0, 0.02], [0.02, 1.0]])
    assert float(losses.supervised_contrastive(separated, labels)) < float(
        losses.supervised_contrastive(mixed, labels)
    )


def test_contrastive_ignores_anchors_without_a_positive():
    """Uma ancora sozinha na classe nao produz termo -- nem zero, nem NaN.

    Se ela entrasse com termo zero, a perda passaria a depender da composicao do
    batch: bastaria sortear classes raras para o numero cair.
    """
    z = _normalize([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    with_orphan = float(losses.supervised_contrastive(z, tf.constant([0, 0, 1])))
    without = float(losses.supervised_contrastive(z[:2], tf.constant([0, 0])))
    assert np.isfinite(with_orphan)
    assert with_orphan == pytest.approx(without, abs=1e-5)


def test_contrastive_of_a_batch_with_no_positive_at_all_is_zero_not_nan():
    z = _normalize([[1.0, 0.0], [0.0, 1.0]])
    value = float(losses.supervised_contrastive(z, tf.constant([0, 1])))
    assert value == pytest.approx(0.0)


def test_aux_regression_is_zero_on_the_target_and_grows_with_the_error():
    target = tf.constant([[0.0, 1.0], [0.5, 0.25]])
    assert float(losses.aux_regression(target, target)) == pytest.approx(0.0)
    assert float(losses.aux_regression(target + 0.2, target)) < float(
        losses.aux_regression(target + 0.4, target)
    )


def test_total_loss_is_the_weighted_sum_of_the_parts():
    rng = np.random.default_rng(2)
    z_e = _normalize(rng.normal(size=(6, 4)))
    labels = tf.constant([0, 0, 1, 1, 2, 2])
    prediction = tf.constant(rng.random((6, 2)).astype(np.float32))
    target = tf.constant(rng.random((6, 2)).astype(np.float32))
    parts = losses.total_loss(z_e, labels, prediction, target, aux_weight=0.5)
    assert set(parts) == {"contrastive", "aux_regression", "total"}
    assert float(parts["total"]) == pytest.approx(
        float(parts["contrastive"]) + 0.5 * float(parts["aux_regression"]), abs=1e-5
    )
