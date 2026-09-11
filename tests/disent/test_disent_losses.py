"""As perdas da fase 1.

O que estes testes protegem sao propriedades, nao valores: o contrastivo tem de
descer quando as classes se separam, a reversao tem de trocar o sinal do
gradiente e so dele, a ortogonalidade tem de ser cega a escala. Fixar valores
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


def test_gradient_reversal_is_the_identity_going_forward():
    x = tf.constant([[1.0, -2.0, 3.0]])
    assert np.allclose(losses.gradient_reversal(x, 0.7).numpy(), x.numpy())


def test_gradient_reversal_flips_and_scales_the_gradient():
    x = tf.Variable([[1.0, 2.0]])
    with tf.GradientTape() as reversed_tape:
        reversed_tape.watch(x)
        y = tf.reduce_sum(losses.gradient_reversal(x, 0.5) * 3.0)
    with tf.GradientTape() as plain_tape:
        plain_tape.watch(x)
        z = tf.reduce_sum(x * 3.0)
    assert np.allclose(
        reversed_tape.gradient(y, x).numpy(), -0.5 * plain_tape.gradient(z, x).numpy()
    )


def test_lambda_ramp_starts_at_zero_and_saturates_at_one():
    assert losses.lambda_ramp(0.0) == pytest.approx(0.0)
    assert losses.lambda_ramp(1.0) == pytest.approx(1.0, abs=1e-3)
    values = [losses.lambda_ramp(p) for p in np.linspace(0, 1, 11)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


def test_lambda_ramp_clamps_outside_the_unit_interval():
    assert losses.lambda_ramp(-3.0) == pytest.approx(0.0)
    assert losses.lambda_ramp(9.0) == losses.lambda_ramp(1.0)


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


def test_orthogonality_is_zero_for_independent_codes_and_one_for_a_copy():
    rng = np.random.default_rng(0)
    a = tf.constant(rng.normal(size=(512, 3)).astype(np.float32))
    b = tf.constant(rng.normal(size=(512, 3)).astype(np.float32))
    assert float(losses.orthogonality(a, b)) < 0.02
    assert float(losses.orthogonality(a, a)) == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_orthogonality_does_not_move_with_the_scale_of_the_codes():
    """Invariancia a escala e o que torna o peso desta perda comparavel entre
    execucoes -- uma norma de Frobenius crua mudaria de significado sozinha.

    A invariancia e da padronizacao, e nao exata: o piso de variancia que evita
    divisao por zero morde quando a escala fica perto dele. Dai a tolerancia ser
    relativa, e as escalas do teste ficarem longe do piso.
    """
    rng = np.random.default_rng(1)
    a = tf.constant(rng.normal(size=(256, 4)).astype(np.float32))
    b = tf.constant(rng.normal(size=(256, 2)).astype(np.float32))
    assert float(losses.orthogonality(a, b)) == pytest.approx(
        float(losses.orthogonality(a * 100.0, b * 0.01)), rel=1e-3
    )


def _context(batch=6):
    rng = np.random.default_rng(2)
    return {
        "z_e": _normalize(rng.normal(size=(batch, 4))),
        "z_c": tf.constant(rng.normal(size=(batch, 3)).astype(np.float32)),
        "aux_prediction": tf.constant(rng.random((batch, 2)).astype(np.float32)),
        "aux_target": tf.constant(rng.random((batch, 2)).astype(np.float32)),
        "adv_arm_logits": tf.constant(rng.normal(size=(batch, 3)).astype(np.float32)),
        "adv_content_logits": tf.constant(rng.normal(size=(batch, 5)).astype(np.float32)),
        "adv_config_logits": tf.constant(rng.normal(size=(batch, 4)).astype(np.float32)),
        "config_label": tf.constant([0, 0, 1, 1, 2, 2]),
        "content_label": tf.constant([0, 1, 2, 3, 4, 0]),
        "arm_label": tf.constant([0, 1, 2, 0, 1, 2]),
        # Fase 2: o decoder devolve o espectro medio no tempo.
        "recon_prediction": tf.constant(rng.normal(size=(batch, 8)).astype(np.float32)),
        "recon_target": tf.constant(rng.normal(size=(batch, 8)).astype(np.float32)),
        "swap_prediction": tf.constant(rng.normal(size=(batch, 8)).astype(np.float32)),
        "swap_target_spec": tf.constant(rng.normal(size=(batch, 8)).astype(np.float32)),
    }


def test_every_registered_loss_runs_on_the_shared_context():
    context = _context()
    for name, function in losses.LOSS_REGISTRY.items():
        value = float(function(context))
        assert np.isfinite(value), name


def test_total_loss_is_the_weighted_sum_of_the_parts():
    context = _context()
    weights = {"contrastive": 2.0, "orthogonality": 0.5}
    parts = losses.total_loss(context, weights)
    assert float(parts["total"]) == pytest.approx(
        2.0 * float(parts["contrastive"]) + 0.5 * float(parts["orthogonality"]), abs=1e-5
    )


def test_a_zero_weight_removes_the_term_instead_of_multiplying_it():
    """Peso zero tem de sumir do relatorio, senao o historico registra um termo
    que nao entrou no gradiente -- e e por ai que se monta o estudo comparativo."""
    parts = losses.total_loss(_context(), {"contrastive": 1.0, "orthogonality": 0.0})
    assert "orthogonality" not in parts


def test_an_unknown_loss_name_is_refused():
    """Este teste nasceu usando `swap_recon` como exemplo de nome desconhecido --
    era o termo da fase 2 que ainda nao existia. Agora existe e esta no registro,
    o que e a prova mais direta de que a costura funcionou como prometido."""
    with pytest.raises(KeyError, match="perda_que_nao_existe"):
        losses.total_loss(_context(), {"perda_que_nao_existe": 1.0})


def test_the_phase_two_terms_are_in_the_registry():
    assert {"recon", "swap_recon"} <= set(losses.LOSS_REGISTRY)
