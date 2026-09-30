"""O encoder: espectrograma -> `z_e` na esfera."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.model import (
    REGRESSION,
    EncoderConfig,
    build_encoder,
    build_model,
    build_regressor,
)

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


def test_weights_round_trip_through_disk(tmp_path):
    model = build_encoder(SMALL)
    x = _batch(3)
    expected = np.asarray(model(x, training=False))
    model.save_weights(tmp_path / "w.weights.h5")

    other = build_encoder(SMALL)
    assert not np.allclose(np.asarray(other(x, training=False)), expected)
    other.load_weights(tmp_path / "w.weights.h5")
    assert np.allclose(np.asarray(other(x, training=False)), expected, atol=1e-6)


def test_loading_weights_that_are_not_there_is_refused(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        build_encoder(SMALL).load_weights(tmp_path / "nada.weights.h5")


def test_the_encoder_config_round_trips_through_its_dict():
    assert EncoderConfig.from_dict(SMALL.as_dict()) == SMALL


def test_a_manifest_with_fields_this_encoder_does_not_have_is_refused():
    """Um `run.json` do encoder de dois blocos nao pode recarregar em silencio."""
    with pytest.raises(TypeError):
        EncoderConfig.from_dict({**SMALL.as_dict(), "content_dim": 64})


@pytest.mark.parametrize("time_pool", ["mean", "max", "flatten"])
def test_every_time_pool_gives_the_effect_code_on_the_sphere(time_pool):
    config = EncoderConfig(**{**SMALL.as_dict(), "time_pool": time_pool})
    z_e = np.asarray(build_encoder(config)(_batch()))
    assert z_e.shape == (4, SMALL.effect_dim)
    assert np.allclose(np.linalg.norm(z_e, axis=1), 1.0, atol=1e-5)


def test_flatten_keeps_the_time_axis_and_the_reductions_do_not():
    """Com `flatten` a camada densa ve frequencia x tempo; com `mean` e `max`,
    so frequencia, e as duas reducoes tem o mesmo numero de pesos."""
    def dense_inputs(time_pool):
        model = build_encoder(EncoderConfig(**{**SMALL.as_dict(), "time_pool": time_pool}))
        return model.get_layer("trunk_dense").kernel.shape[0]

    assert dense_inputs("mean") == dense_inputs("max")
    assert dense_inputs("flatten") > dense_inputs("mean")


def test_a_manifest_from_before_the_ablation_loads_with_the_mean():
    """Os `run.json` anteriores nao tem `time_pool`: rodaram com a media."""
    old = {k: v for k, v in SMALL.as_dict().items() if k != "time_pool"}
    assert EncoderConfig.from_dict(old).time_pool == "mean"


def test_an_unknown_time_pool_is_refused():
    with pytest.raises(ValueError):
        EncoderConfig(time_pool="median")


def test_the_regressor_gives_one_drive_value_in_the_unit_interval():
    drive = np.asarray(build_regressor(SMALL)(_batch(8)))
    assert drive.shape == (8, 1)
    assert np.all((drive >= 0) & (drive <= 1))


def test_the_regressor_shares_the_trunk_and_changes_only_the_head():
    """O controle escalar so vale se a unica diferenca for a cabeca."""
    def body(model, head):
        return [(layer.name, layer.count_params()) for layer in model.layers
                if layer.name not in head]

    assert (body(build_encoder(SMALL), {"effect_dense", "effect_code"})
            == body(build_regressor(SMALL), {"drive"}))


def test_build_model_picks_the_regressor_only_for_its_technique():
    assert build_model(SMALL, REGRESSION).output_shape == (None, 1)
    assert build_model(SMALL, "supcon").output_shape == (None, SMALL.effect_dim)
