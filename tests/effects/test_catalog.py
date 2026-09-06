"""O catalogo e a fonte unica de verdade: reordenar entradas aqui muda o vetor
alvo e o sufixo binario dos nomes de arquivo, invalidando datasets e modelos sem
levantar excecao. Estes testes fixam o formato atual."""
from __future__ import annotations

import pytest

from gefx.effects.catalog import (
    CANONICAL_EFFECT_CHAIN_ORDER,
    CHAIN_KEY_SEPARATOR,
    EFFECT_CHAINS,
    EFFECT_PARAMETER_RANGES,
    chain_key,
    chain_key_to_effects,
    chain_output_dim,
    effect_fixed_params,
    effect_predictable_params,
    parameter_names_for_chain,
)

# Tabela dourada: cadeia -> (dimensao da saida, nomes na ordem do vetor alvo).
GOLDEN = {
    "distortion": (1, ["distortion_drive_db"]),
    "chorus": (2, ["chorus_rate_hz", "chorus_depth"]),
    "vibrato": (2, ["vibrato_rate_hz", "vibrato_depth"]),
    "flanger": (
        4,
        ["flanger_rate_hz", "flanger_depth", "flanger_feedback", "flanger_centre_delay_ms"],
    ),
    "feedback_delay": (
        3,
        ["feedback_delay_delay_seconds", "feedback_delay_feedback", "feedback_delay_mix"],
    ),
    "slapback_delay": (2, ["slapback_delay_delay_seconds", "slapback_delay_mix"]),
    "phaser": (2, ["phaser_rate_hz", "phaser_depth"]),
    "reverb": (1, ["reverb_room_size"]),
    "distortion__chorus": (3, ["distortion_drive_db", "chorus_rate_hz", "chorus_depth"]),
    "chorus__slapback_delay": (
        4,
        ["chorus_rate_hz", "chorus_depth", "slapback_delay_delay_seconds", "slapback_delay_mix"],
    ),
    "distortion__chorus__slapback_delay": (
        5,
        [
            "distortion_drive_db",
            "chorus_rate_hz",
            "chorus_depth",
            "slapback_delay_delay_seconds",
            "slapback_delay_mix",
        ],
    ),
}


def test_effect_order_is_the_documented_one():
    assert list(EFFECT_PARAMETER_RANGES) == [
        "distortion",
        "chorus",
        "vibrato",
        "flanger",
        "feedback_delay",
        "slapback_delay",
        "phaser",
        "reverb",
    ]


def test_chain_set_is_every_single_effect_plus_three_stacked():
    keys = [chain_key(chain) for chain in EFFECT_CHAINS]
    assert keys == list(EFFECT_PARAMETER_RANGES) + [
        "distortion__chorus",
        "chorus__slapback_delay",
        "distortion__chorus__slapback_delay",
    ]
    assert len(keys) == 11
    assert len(set(keys)) == 11


def test_golden_table_covers_every_chain():
    assert set(GOLDEN) == {chain_key(chain) for chain in EFFECT_CHAINS}


@pytest.mark.parametrize("chain_key_value", sorted(GOLDEN))
def test_output_dim_and_parameter_names(chain_key_value):
    expected_dim, expected_names = GOLDEN[chain_key_value]
    assert chain_output_dim(chain_key_value) == expected_dim
    assert parameter_names_for_chain(chain_key_value) == expected_names


@pytest.mark.parametrize("chain_key_value", sorted(GOLDEN))
def test_names_and_dim_agree(chain_key_value):
    assert len(parameter_names_for_chain(chain_key_value)) == chain_output_dim(chain_key_value)


@pytest.mark.parametrize("chain", EFFECT_CHAINS, ids=lambda chain: chain_key(chain))
def test_chain_key_round_trip(chain):
    assert chain_key_to_effects(chain_key(chain)) == list(chain)


def test_empty_chain_key():
    assert chain_key_to_effects("") == []
    assert chain_output_dim("") == 0
    assert parameter_names_for_chain("") == []


def test_no_effect_name_contains_the_separator():
    # Um nome com `__` quebraria `chain_key_to_effects` sem sinal nenhum.
    for effect in EFFECT_PARAMETER_RANGES:
        assert CHAIN_KEY_SEPARATOR not in effect


@pytest.mark.parametrize("effect", sorted(EFFECT_PARAMETER_RANGES))
def test_ranges_are_well_formed(effect):
    for param in EFFECT_PARAMETER_RANGES[effect]:
        assert set(param) <= {"name", "min", "max", "predict"}
        assert param["min"] <= param["max"]


@pytest.mark.parametrize("effect", sorted(EFFECT_PARAMETER_RANGES))
def test_fixed_params_are_degenerate_ranges(effect):
    # `raw_params_for_effect` usa `min` como valor do parametro fixo; se algum
    # fixo tivesse min != max, o `max` seria silenciosamente ignorado.
    for param in effect_fixed_params(effect):
        assert param["min"] == param["max"]


@pytest.mark.parametrize("effect", sorted(EFFECT_PARAMETER_RANGES))
def test_predictable_and_fixed_partition_the_params(effect):
    predictable = effect_predictable_params(effect)
    fixed = effect_fixed_params(effect)
    assert len(predictable) + len(fixed) == len(EFFECT_PARAMETER_RANGES[effect])
    names = [param["name"] for param in predictable + fixed]
    assert len(names) == len(set(names))


def test_canonical_order_effects_exist():
    for effect in CANONICAL_EFFECT_CHAIN_ORDER:
        assert effect in EFFECT_PARAMETER_RANGES
