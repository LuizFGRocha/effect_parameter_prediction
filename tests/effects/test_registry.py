"""Registro de implementacoes de terceiros: conversoes puras e coerencia com o catalogo."""
from __future__ import annotations

import pytest

from gefx.effects.catalog import EFFECT_PARAMETER_RANGES, effect_predictable_params
from gefx.effects.registry import (
    REFERENCE_ARM,
    REGISTRY,
    all_arms,
    ident,
    is_exact,
    onto,
    to_db,
    to_pct,
)


def test_ident_to_pct():
    assert ident(3.5) == 3.5
    assert to_pct(0.35) == pytest.approx(35.0)


@pytest.mark.parametrize(
    "value, expected",
    [(1.0, 0.0), (0.1, -20.0), (0.5, pytest.approx(-6.0206, abs=1e-4))],
)
def test_to_db_converts_linear_gain(value, expected):
    assert to_db(value) == expected


def test_to_db_floors_at_minus_60():
    # O clamp em 1e-4 daria -80 dB; o piso corta em -60.
    assert to_db(0.0) == -60.0
    assert to_db(1e-9) == -60.0


def test_onto_maps_normalized_range():
    convert = onto(2.0, 10.0)
    assert convert(0.0) == pytest.approx(2.0)
    assert convert(0.5) == pytest.approx(6.0)
    assert convert(1.0) == pytest.approx(10.0)


def test_all_arms():
    assert all_arms() == ["pedalboard", "chowcentaur", "chowphaser", "dragonfly", "lsp"]


def test_is_exact_reference_arm_is_always_exact():
    assert is_exact("distortion", REFERENCE_ARM, "drive_db") is True
    assert is_exact("efeito_inexistente", REFERENCE_ARM, "seja_o_que_for") is True


def test_is_exact_reads_the_registry_flag():
    assert is_exact("distortion", "lsp", "drive_db") is True
    assert is_exact("distortion", "chowcentaur", "drive_db") is False
    assert is_exact("chorus", "lsp", "depth") is False


def test_is_exact_unknown_arm_or_parameter_is_false():
    assert is_exact("distortion", "braco_inexistente", "drive_db") is False
    assert is_exact("distortion", "lsp", "parametro_inexistente") is False


def test_delays_are_deliberately_absent():
    # Nenhum delay de terceiros passou na checagem de sanidade; se algum entrar,
    # este teste avisa que o docstring do modulo precisa ser atualizado.
    assert "slapback_delay" not in REGISTRY
    assert "feedback_delay" not in REGISTRY


@pytest.mark.parametrize("effect", sorted(REGISTRY))
def test_registry_effects_exist_in_catalog(effect):
    assert effect in EFFECT_PARAMETER_RANGES


@pytest.mark.parametrize("effect", sorted(REGISTRY))
def test_mapped_reference_params_exist_in_catalog(effect):
    known = {param["name"] for param in EFFECT_PARAMETER_RANGES[effect]} | {"_mix_dry"}
    for spec in REGISTRY[effect].values():
        assert set(spec["map"]) <= known


@pytest.mark.parametrize("effect", sorted(REGISTRY))
def test_non_exact_mappings_reference_predictable_params(effect):
    # Invariante estrutural: `_apply_mapping` resolve um mapeamento nao-exato por
    # `names.index(ref_name)` sobre os parametros *previstos*. Apontar um nao-exato
    # para um parametro fixo levantaria ValueError so na hora de renderizar.
    predictable = {param["name"] for param in effect_predictable_params(effect)}
    for spec in REGISTRY[effect].values():
        for ref_name, (_, _, exact) in spec["map"].items():
            if not exact and ref_name != "_mix_dry":
                assert ref_name in predictable


@pytest.mark.parametrize("effect", sorted(REGISTRY))
def test_specs_are_well_formed(effect):
    for arm, spec in REGISTRY[effect].items():
        assert arm != REFERENCE_ARM
        assert set(spec) <= {"path", "plugin_name", "fixed", "map"}
        assert spec["path"].endswith(".vst3")
        assert isinstance(spec["fixed"], dict)
        for plugin_param, convert, exact in spec["map"].values():
            assert isinstance(plugin_param, str)
            assert callable(convert)
            assert isinstance(exact, bool)
