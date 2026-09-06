"""Nucleo do estudo cross-implementation: qual valor cada mapeamento consome.

Errar isso nao quebra nada visivelmente — o render sai, o modelo prediz, e o
numero final e que fica errado.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.crossimpl import render as render_module
from gefx.crossimpl.render import _apply_mapping
from gefx.effects.catalog import effect_predictable_params
from gefx.effects.parameters import raw_params_for_effect
from gefx.effects.registry import REGISTRY, ident, onto, to_db


@pytest.fixture
def written(monkeypatch):
    """Registra o que `_apply_mapping` escreveria, sem passar por `set_parameter`."""
    calls = {}
    monkeypatch.setattr(
        render_module, "set_parameter", lambda plugin, name, value: calls.__setitem__(name, value)
    )
    return calls


def test_exact_mapping_consumes_the_raw_physical_value(written):
    spec = {"map": {"drive_db": ("input_gain_db", ident, True)}}
    raw = {"drive_db": 22.5}

    _apply_mapping(object(), spec, raw, np.array([0.5]), ["drive_db"])

    assert written == {"input_gain_db": pytest.approx(22.5)}


def test_non_exact_mapping_consumes_the_normalized_value(written):
    spec = {"map": {"depth": ("depth_ms", onto(0.5, 12.0), False)}}
    # `raw["depth"]` existe e e diferente; o mapeamento nao-exato deve ignora-lo.
    raw = {"depth": 0.4}

    _apply_mapping(object(), spec, raw, np.array([0.1, 0.5]), ["rate_hz", "depth"])

    assert written == {"depth_ms": pytest.approx(0.5 + 11.5 * 0.5)}


def test_non_exact_mapping_indexes_by_position_in_the_predictable_names(written):
    spec = {"map": {"feedback": ("fb", onto(0.0, 10.0), False)}}
    names = ["rate_hz", "depth", "feedback", "centre_delay_ms"]

    _apply_mapping(object(), spec, {}, np.array([0.0, 0.0, 0.25, 0.0]), names)

    assert written == {"fb": pytest.approx(2.5)}


def test_mix_dry_writes_the_complementary_gain(written):
    # Chave especial da metade seca de um crossfade. Nenhuma entrada do REGISTRY
    # atual a usa, mas o codigo esta vivo.
    spec = {"map": {"_mix_dry": ("dry_amount_db", ident, True)}}

    _apply_mapping(object(), spec, {"mix": 0.75}, np.array([]), [])

    assert written == {"dry_amount_db": pytest.approx(to_db(0.25))}


def test_every_value_written_is_a_float(written):
    spec = {"map": {"drive_db": ("input_gain_db", ident, True)}}
    _apply_mapping(object(), spec, {"drive_db": np.float64(3.0)}, np.array([0.5]), ["drive_db"])
    assert type(written["input_gain_db"]) is float


def test_a_non_exact_mapping_of_a_fixed_param_would_fail_loudly(written):
    # Guarda o invariante que `test_registry` checa estruturalmente: um nao-exato
    # so pode apontar para um parametro previsto, porque a resolucao e por indice
    # na lista de previstos.
    spec = {"map": {"mix": ("dry_wet", onto(0.0, 1.0), False)}}
    with pytest.raises(ValueError, match="not in list"):
        _apply_mapping(object(), spec, {"mix": 0.5}, np.array([0.1]), ["rate_hz"])


@pytest.mark.parametrize(
    "effect, arm",
    [(effect, arm) for effect in sorted(REGISTRY) for arm in sorted(REGISTRY[effect])],
)
def test_real_registry_specs_apply_without_error(effect, arm, written):
    # Roda o mapeamento real de cada braco com um vetor normalizado plausivel.
    names = [param["name"] for param in effect_predictable_params(effect)]
    norm_row = np.full(len(names), 0.5)
    raw = raw_params_for_effect(effect, list(norm_row))

    _apply_mapping(object(), REGISTRY[effect][arm], raw, norm_row, names)

    expected = {plugin_param for plugin_param, _, _ in REGISTRY[effect][arm]["map"].values()}
    assert set(written) == expected
    assert all(np.isfinite(value) for value in written.values())
