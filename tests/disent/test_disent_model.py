"""O encoder: espectrograma -> `z_e` na esfera."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.model import EffectModel, EncoderConfig, build_encoder

SMALL = EncoderConfig(input_shape=(32, 24, 1), filters=(8, 16), trunk_units=16,
                      effect_dim=4)


def _batch(n=4, config=SMALL):
    return np.random.default_rng(0).random((n, *config.input_shape)).astype(np.float32)


def test_the_encoder_gives_the_effect_code_with_the_declared_width():
    assert build_encoder(SMALL)(_batch()).shape == (4, SMALL.effect_dim)


def test_the_effect_code_lives_on_the_unit_sphere():
    """A busca so le a direcao de `z_e`. Norma livre daria a rede um jeito de
    mexer na perda sem mudar nada do que a recuperacao enxerga."""
    z_e = build_encoder(SMALL)(_batch(8))
    assert np.allclose(np.linalg.norm(np.asarray(z_e), axis=1), 1.0, atol=1e-5)


def test_the_forward_pass_is_the_effect_code():
    model = EffectModel(SMALL)
    x = _batch()
    assert np.array_equal(np.asarray(model(x, training=False)), np.asarray(model.encode(x)))


def test_weights_round_trip_through_disk(tmp_path):
    model = EffectModel(SMALL)
    x = _batch(3)
    expected = np.asarray(model.encode(x))
    model.save_weights(tmp_path / "w")

    other = EffectModel(SMALL)
    assert not np.allclose(np.asarray(other.encode(x)), expected)
    other.load_weights(tmp_path / "w")
    assert np.allclose(np.asarray(other.encode(x)), expected, atol=1e-6)


def test_loading_from_a_directory_without_weights_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        EffectModel(SMALL).load_weights(tmp_path / "empty")


def test_the_encoder_config_round_trips_through_its_dict():
    assert EncoderConfig.from_dict(SMALL.as_dict()) == SMALL


def test_a_manifest_with_fields_this_encoder_does_not_have_is_refused():
    """Um `run.json` do encoder de dois blocos nao pode recarregar em silencio."""
    with pytest.raises(TypeError):
        EncoderConfig.from_dict({**SMALL.as_dict(), "content_dim": 64})
