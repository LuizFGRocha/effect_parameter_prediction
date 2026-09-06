"""Primitivas de grafico: so o que tem risco de quebrar em dados degenerados.

O backend Agg e fixado no conftest, entao nada aqui abre janela.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gefx.evaluation.plots import parity_axis, parity_grid, plot_best_worst, safe_filename


@pytest.mark.parametrize(
    "value, expected",
    [
        ("chorus_rate_hz", "chorus_rate_hz"),
        ("a/b", "a__b"),
        ("a\\b", "a__b"),
        ("com espaco", "com_espaco"),
        ("a/b c", "a__b_c"),
    ],
)
def test_safe_filename(value, expected):
    assert safe_filename(value) == expected


def test_parity_axis_sets_square_equal_limits():
    fig, axis = plt.subplots()
    try:
        parity_axis(axis, np.array([0.0, 1.0]), np.array([0.2, 0.8]), "t")
        assert axis.get_xlim() == axis.get_ylim()
        assert axis.get_xlim() == (pytest.approx(-0.05), pytest.approx(1.05))
    finally:
        plt.close(fig)


def test_parity_axis_survives_a_constant_series():
    # max == min: o padding cai para a base 1.0 em vez de virar zero.
    fig, axis = plt.subplots()
    try:
        parity_axis(axis, np.array([0.5, 0.5]), np.array([0.5, 0.5]), "t")
        low, high = axis.get_xlim()
        assert high > low
    finally:
        plt.close(fig)


def make_plot_frame(pairs):
    rows = []
    for chain_key_value, parameter, error in pairs:
        for real in (0.2, 0.5, 0.8):
            rows.append(
                {
                    "chain_key": chain_key_value,
                    "parameter": parameter,
                    "real": real,
                    "estimated": real + error,
                }
            )
    return pd.DataFrame(rows)


def test_plot_best_worst_requires_data(tmp_path):
    # Colunas tipadas, como sai de `build_prediction_frame`: com object dtype o
    # `nsmallest` do pandas levantaria TypeError antes da checagem de vazio.
    empty = pd.DataFrame(
        {
            "chain_key": pd.Series(dtype=str),
            "parameter": pd.Series(dtype=str),
            "real": pd.Series(dtype="float64"),
            "estimated": pd.Series(dtype="float64"),
        }
    )
    with pytest.raises(ValueError, match="Nenhuma predicao"):
        plot_best_worst(empty, tmp_path / "bw.png")


def test_parity_grid_rejects_an_empty_panel_list(tmp_path):
    # Antes: ZeroDivisionError sem contexto, vindo de `ceil(0 / 0)`.
    with pytest.raises(ValueError, match="pelo menos um painel"):
        parity_grid([], tmp_path / "g.png", "titulo")

