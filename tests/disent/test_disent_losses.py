"""A perda do encoder, o SupCon traduzido do repositorio oficial.

`test_the_translation_gives_the_values_of_the_original` prende a traducao ao
original: os valores esperados sairam do `SupConLoss` do repositorio oficial
(HobbitLong/SupContrast, commit 72fd989), rodado em PyTorch nas mesmas entradas.
Revisar a traducao se resume a este teste passar; a tolerancia (1e-5 relativo)
fica folgada para a diferenca de arredondamento em float32 entre os dois
frameworks, que foi de 1e-7. Os demais testes protegem propriedades.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from gefx.disent.losses import sup_con_loss


def _normalize(x):
    return tf.math.l2_normalize(tf.constant(np.asarray(x, dtype=np.float32)), axis=-1)


def _one_view(x):
    return _normalize(x)[:, None, :]


def _unit_features(seed, shape):
    z = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return tf.constant(z / np.linalg.norm(z, axis=-1, keepdims=True))


@pytest.mark.parametrize(
    "seed, shape, labels, temperature, base_temperature, original",
    [
        # uma vista por linha, 4 classes x 4: o formato do treino
        (0, (16, 1, 8), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 0.07, 0.07,
         9.643486976623535),
        # duas vistas por linha: o caminho que desempilha as vistas
        (1, (8, 2, 8), [0, 0, 1, 1, 2, 2, 3, 3], 0.07, 0.07, 10.859750747680664),
        # a ultima ancora nao tem positivo: o caso de borda que o original trata
        (2, (7, 1, 8), [0, 0, 1, 1, 2, 2, 3], 0.07, 0.07, 7.837521076202393),
        # temperatura diferente da base: o fator temperature / base_temperature
        (3, (16, 1, 8), [0, 1, 2, 3] * 4, 0.1, 0.07, 9.653303146362305),
    ],
)
def test_the_translation_gives_the_values_of_the_original(
        seed, shape, labels, temperature, base_temperature, original):
    value = sup_con_loss(_unit_features(seed, shape), tf.constant(labels),
                         temperature, base_temperature)
    assert float(value) == pytest.approx(original, rel=1e-5)


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

