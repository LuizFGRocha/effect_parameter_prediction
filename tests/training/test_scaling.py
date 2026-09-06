"""Padronizacao das features.

E o unico ponto que aplica escala, e os scalers ajustados aqui sao persistidos e
reusados pelo `crossimpl` — uma divergencia silenciosa aqui invalida a comparacao
entre implementacoes.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.training.scaling import apply_scalers, fit_scalers, fit_transform_split


@pytest.fixture
def features():
    return np.random.default_rng(0).normal(loc=5.0, scale=2.0, size=(20, 4, 7)).astype(np.float32)


def test_fit_scalers_is_one_scaler_per_row(features):
    scalers = fit_scalers(features)
    assert sorted(scalers) == [0, 1, 2, 3]
    # Cada scaler ve uma linha do espectrograma: 7 colunas.
    assert scalers[0].mean_.shape == (7,)


def test_apply_scalers_adds_the_channel_axis(features):
    scaled = apply_scalers(features, fit_scalers(features))
    assert scaled.shape == (20, 4, 7, 1)
    assert scaled.dtype == np.float32


def test_apply_scalers_does_not_mutate_the_input(features):
    original = features.copy()
    apply_scalers(features, fit_scalers(features))
    assert np.array_equal(features, original)


def test_apply_scalers_standardizes_each_row_independently(features):
    scaled = apply_scalers(features, fit_scalers(features))[..., 0]
    for row in range(scaled.shape[1]):
        assert np.allclose(scaled[:, row, :].mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(scaled[:, row, :].std(axis=0), 1.0, atol=1e-5)


def test_fit_transform_split_fits_on_train_only():
    rng = np.random.default_rng(1)
    x_train = rng.normal(loc=0.0, scale=1.0, size=(30, 3, 5))
    x_test = rng.normal(loc=10.0, scale=1.0, size=(10, 3, 5))

    scaled_train, scaled_test, scalers = fit_transform_split(x_train, x_test)

    assert scaled_train.shape == (30, 3, 5, 1)
    assert scaled_test.shape == (10, 3, 5, 1)
    assert np.allclose(scaled_train.mean(axis=0), 0.0, atol=1e-6)
    # O teste nao entra no ajuste, entao ele nao fica centrado.
    assert abs(float(scaled_test.mean())) > 5.0
    assert sorted(scalers) == [0, 1, 2]


@pytest.mark.parametrize("rows", [2, 6])
def test_apply_scalers_rejects_a_row_count_that_does_not_match_the_scalers(features, rows):
    # Com menos linhas do que scalers o laco parava antes e devolvia um array
    # escalado so em parte, sem erro. Como o `crossimpl` reusa scalers
    # persistidos, era por ai que uma comparacao errada passaria despercebida.
    scalers = fit_scalers(features)
    with pytest.raises(ValueError, match=f"features={rows}, scalers=4"):
        apply_scalers(np.zeros((5, rows, 7), dtype=np.float32), scalers)
