"""A CNN do POC I e o encoder do POC II, que e a mesma rede com outra saida."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from gefx.config import ArchitectureConfig, load_config
from gefx.training.architecture import (
    ARCHITECTURES,
    REGRESSION,
    build_encoder,
    build_model,
    build_network,
)

SHAPE = (32, 24, 1)
SMALL = ArchitectureConfig(n_filters=8, n_nodes=16, effect_dim=4, time_pool="mean")


def _batch(n=4):
    return np.random.default_rng(0).random((n, *SHAPE)).astype(np.float32)


def test_the_encoder_gives_the_effect_code_with_the_declared_width():
    assert build_encoder(SHAPE, SMALL)(_batch()).shape == (4, SMALL.effect_dim)


@pytest.mark.parametrize("time_pool", ["mean", "flatten"])
def test_the_effect_code_lives_on_the_unit_sphere(time_pool):
    """A busca so le a direcao de `z_e`. Norma livre daria a rede um jeito de
    mexer na perda sem mudar nada do que a recuperacao enxerga."""
    z_e = build_encoder(SHAPE, replace(SMALL, time_pool=time_pool))(_batch(8))
    assert np.allclose(np.linalg.norm(np.asarray(z_e), axis=1), 1.0, atol=1e-5)


def test_the_time_pool_is_the_mean_over_the_time_axis():
    from keras import layers, models

    model = build_encoder(SHAPE, SMALL)
    pool = next(layer for layer in model.layers if isinstance(layer, layers.AveragePooling2D))
    probe = models.Model(model.input, [pool.input, pool.output])
    before, after = (np.asarray(t) for t in probe(_batch(2)))
    assert np.allclose(after[:, :, 0, :], before.mean(axis=2), atol=1e-6)


def test_flatten_keeps_the_time_axis_and_the_mean_does_not():
    def dense_inputs(time_pool):
        model = build_encoder(SHAPE, replace(SMALL, time_pool=time_pool))
        return next(layer for layer in model.layers
                    if type(layer).__name__ == "Dense").kernel.shape[0]

    assert dense_inputs("flatten") > dense_inputs("mean")


def test_an_unknown_time_pool_is_refused():
    with pytest.raises(ValueError, match="time_pool"):
        ArchitectureConfig(time_pool="max")


def test_weights_round_trip_through_disk(tmp_path):
    model = build_encoder(SHAPE, SMALL)
    x = _batch(3)
    expected = np.asarray(model(x, training=False))
    model.save_weights(tmp_path / "w.weights.h5")

    other = build_encoder(SHAPE, SMALL)
    assert not np.allclose(np.asarray(other(x, training=False)), expected)
    other.load_weights(tmp_path / "w.weights.h5")
    assert np.allclose(np.asarray(other(x, training=False)), expected, atol=1e-6)


def test_the_regressor_gives_one_drive_value_in_the_unit_interval():
    drive = np.asarray(build_network(SHAPE, SMALL, REGRESSION)(_batch(8)))
    assert drive.shape == (8, 1)
    assert np.all((drive >= 0) & (drive <= 1))


def _layers(model):
    """(tipo, forma de saida, parametros) de cada camada que calcula algo."""
    from keras import layers

    return [(type(layer).__name__, tuple(layer.output.shape[1:]), layer.count_params())
            for layer in model.layers if not isinstance(layer, layers.InputLayer)]


def test_the_encoder_and_the_regressor_differ_only_in_the_head():
    """O controle escalar so vale se a unica diferenca for a cabeca."""
    encoder, regressor = _layers(build_encoder(SHAPE, SMALL)), _layers(build_model(SHAPE, 1, SMALL))
    assert encoder[:-2] == regressor[:-1]
    assert encoder[-2][0] == "Dense" and encoder[-1][0] == "UnitNormalization"


def test_build_network_picks_the_regressor_only_for_its_technique():
    assert build_network(SHAPE, SMALL, REGRESSION).output_shape == (None, 1)
    assert build_network(SHAPE, SMALL, "supcon").output_shape == (None, SMALL.effect_dim)


def test_the_first_rung_is_the_poc1_cnn():
    assert ARCHITECTURES["poc1"] == load_config("experiments/base.yaml").architecture


def test_the_report_encoder_is_the_poc1_cnn_with_the_time_mean_wider_and_deeper():
    """(32, 64, 96, 128) filtros sao a regra do POC I, `n_filters * (i + 1)`."""
    from keras import layers

    model = build_encoder((256, 173, 1), ARCHITECTURES["poc2"])
    widths = [layer.filters for layer in model.layers if isinstance(layer, layers.Conv2D)]
    assert widths == [32, 64, 96, 128]


def test_each_rung_of_the_ladder_changes_one_thing():
    """A escada so mede uma mudanca por degrau se cada um diferir do anterior num
    campo so, e ela vai do POC I ao encoder."""
    rungs = [asdict(config) for config in ARCHITECTURES.values()]
    for before, after in zip(rungs, rungs[1:]):
        assert sum(before[key] != after[key] for key in before) == 1
    assert list(ARCHITECTURES)[0] == "poc1" and list(ARCHITECTURES)[-1] == "poc2"
