"""O encoder bifurcado.

O teste que importa aqui e `test_encode_is_independent_of_the_heads`: o contrato
da fase 2 e `encode(x) -> (z_e, z_c)`, e se ele passar a depender das cabecas de
treino o decoder deixa de plugar sem reescrita.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.model import BetaVAE, DisentModel, EncoderConfig, HeadConfig, build_encoder

SMALL = EncoderConfig(
    input_shape=(32, 24, 1), filters=(8, 16), trunk_units=16,
    effect_dim=4, content_dim=6, adversary_units=8,
)
HEADS = HeadConfig(n_arms=3, n_contents=5, n_configs=4)


def _batch(n=4, config=SMALL):
    return np.random.default_rng(0).random((n, *config.input_shape)).astype(np.float32)


def test_encoder_gives_the_two_codes_with_the_declared_widths():
    z_e, z_c = build_encoder(SMALL)(_batch())
    assert z_e.shape[1] == SMALL.effect_dim
    assert z_c.shape[1] == SMALL.content_dim


def test_the_effect_code_lives_on_the_unit_sphere():
    """A busca so le a direcao de `z_e`. Norma livre daria a rede um jeito de
    mexer na perda sem mudar nada do que a recuperacao enxerga."""
    z_e, _ = build_encoder(SMALL)(_batch(8))
    assert np.allclose(np.linalg.norm(np.asarray(z_e), axis=1), 1.0, atol=1e-5)


def test_the_content_code_is_not_normalized():
    _, z_c = build_encoder(SMALL)(_batch(8))
    assert not np.allclose(np.linalg.norm(np.asarray(z_c), axis=1), 1.0, atol=1e-3)


def test_encode_is_independent_of_the_heads():
    """A costura da fase 2: `encode` nao pode passar pelas cabecas de treino."""
    model = DisentModel(SMALL, HEADS)
    x = _batch(3)
    before = [np.asarray(code) for code in model.encode(x)]
    for head in model.adversaries.values():
        for weight in head.trainable_variables:
            weight.assign(weight + 1.0)
    for weight in model.aux_head.trainable_variables:
        weight.assign(weight + 1.0)
    after = [np.asarray(code) for code in model.encode(x)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_the_forward_pass_gives_every_key_the_losses_expect():
    output = DisentModel(SMALL, HEADS)(_batch(), lam=0.5, training=False)
    assert set(output) == {
        "z_e", "z_c", "aux_prediction",
        "adv_arm_logits", "adv_content_logits", "adv_config_logits",
    }
    assert output["adv_arm_logits"].shape[1] == HEADS.n_arms
    assert output["adv_content_logits"].shape[1] == HEADS.n_contents
    assert output["adv_config_logits"].shape[1] == HEADS.n_configs


def test_the_config_adversary_hangs_on_the_content_code_and_not_the_effect_one():
    """O adversario de configuracao e o lado simetrico: ele tem de ver `z_c`.

    Pendura-lo em `z_e` inverteria o desenho -- passaria a apagar a configuracao
    justamente do codigo que a recuperacao usa.
    """
    model = DisentModel(SMALL, HEADS)
    assert model.adversaries["config"].input_shape[-1] == SMALL.content_dim
    assert model.adversaries["arm"].input_shape[-1] == SMALL.effect_dim
    assert model.adversaries["content"].input_shape[-1] == SMALL.effect_dim


def test_the_gradient_of_the_arm_adversary_reaches_the_encoder_reversed():
    import tensorflow as tf

    model = DisentModel(SMALL, HEADS)
    x = tf.constant(_batch(6))
    labels = tf.constant([0, 1, 2, 0, 1, 2])
    variables = model.encoder.trainable_variables

    def arm_gradient(lam):
        with tf.GradientTape() as tape:
            logits = model(x, lam=lam, training=False)["adv_arm_logits"]
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )
        return [g for g in tape.gradient(loss, variables) if g is not None]

    forward = arm_gradient(1.0)
    blocked = arm_gradient(0.0)
    assert forward, "o adversario nao alcanca o encoder"
    assert all(np.allclose(np.asarray(g), 0.0) for g in blocked)


def test_weights_round_trip_through_disk(tmp_path):
    model = DisentModel(SMALL, HEADS)
    x = _batch(3)
    expected = np.asarray(model.encode(x)[0])
    model.save_weights(tmp_path / "w")

    other = DisentModel(SMALL, HEADS)
    assert not np.allclose(np.asarray(other.encode(x)[0]), expected)
    other.load_weights(tmp_path / "w")
    assert np.allclose(np.asarray(other.encode(x)[0]), expected, atol=1e-6)


def test_loading_from_a_directory_without_the_encoder_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="encoder"):
        DisentModel(SMALL, HEADS).load_weights(tmp_path / "empty")


def test_the_encoder_config_round_trips_through_its_dict():
    restored = EncoderConfig.from_dict(SMALL.as_dict())
    assert restored == SMALL


def test_the_beta_vae_control_answers_the_same_encode_contract():
    """O controle tem de ser avaliavel pelo mesmo caminho, senao a comparacao
    mede a diferenca de avaliacao e nao a de tecnica."""
    vae = BetaVAE(EncoderConfig(input_shape=(64, 48, 1), filters=(8, 16),
                                trunk_units=16, effect_dim=4, content_dim=6))
    z_e, z_c = vae.encode(np.random.default_rng(1).random((3, 64, 48, 1)).astype(np.float32))
    assert z_e.shape[1] == 4 and z_c.shape[1] == 6
    assert np.allclose(np.linalg.norm(np.asarray(z_e), axis=1), 1.0, atol=1e-5)


def test_the_beta_vae_reconstruction_has_the_shape_of_the_input():
    config = EncoderConfig(input_shape=(64, 48, 1), filters=(8, 16), trunk_units=16,
                           effect_dim=4, content_dim=6)
    vae = BetaVAE(config)
    parts = vae.losses(np.random.default_rng(2).random((2, 64, 48, 1)).astype(np.float32))
    assert set(parts) == {"reconstruction", "kl", "total"}
    assert all(np.isfinite(float(value)) for value in parts.values())
