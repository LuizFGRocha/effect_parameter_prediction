"""Conversao normalizado <-> fisico, fatiamento do vetor alvo e sufixo binario."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.effects.catalog import EFFECT_CHAINS, chain_key
from gefx.effects.parameters import (
    binary_suffix,
    convert_normalized_to_raw,
    effect_presence,
    raw_params_for_effect,
    sample_normalized_matrix,
    split_vector_by_effect,
)

# Cadeia -> sufixo binario, na ordem de insercao do catalogo.
GOLDEN_SUFFIX = {
    "distortion": "10000000",
    "chorus": "01000000",
    "vibrato": "00100000",
    "flanger": "00010000",
    "feedback_delay": "00001000",
    "slapback_delay": "00000100",
    "phaser": "00000010",
    "reverb": "00000001",
    "distortion__chorus": "11000000",
    "chorus__slapback_delay": "01000100",
    "distortion__chorus__slapback_delay": "11000100",
}

PARAMS = [{"name": "a", "min": 5.0, "max": 40.0}, {"name": "b", "min": -1.0, "max": 1.0}]


@pytest.mark.parametrize(
    "value, expected_a, expected_b",
    [(0.0, 5.0, -1.0), (0.5, 22.5, 0.0), (1.0, 40.0, 1.0)],
)
def test_convert_normalized_to_raw_is_affine(value, expected_a, expected_b):
    raw = convert_normalized_to_raw(PARAMS, [value, value])
    assert raw == {"a": pytest.approx(expected_a), "b": pytest.approx(expected_b)}


def test_convert_normalized_to_raw_returns_plain_floats():
    raw = convert_normalized_to_raw(PARAMS, np.array([0.25, 0.75]))
    assert all(type(value) is float for value in raw.values())


@pytest.mark.parametrize("norm_values", [[], [0.0], [0.0, 0.0, 0.0]])
def test_convert_normalized_to_raw_rejects_a_mismatched_vector(norm_values):
    # Um vetor curto renderizaria o efeito com os defaults do plugin nos
    # parametros que faltassem, sem nenhum sinal.
    with pytest.raises(ValueError, match=f"{len(norm_values)} valores para 2 parametros"):
        convert_normalized_to_raw(PARAMS, norm_values)


def test_split_vector_by_effect_rejects_a_vector_that_is_too_short():
    # O fatiamento por offset produziria a ultima fatia curta.
    with pytest.raises(ValueError, match="valores para 2 parametros"):
        split_vector_by_effect(["distortion", "chorus"], [0.5, 0.5])


def test_raw_params_for_effect_adds_fixed_params():
    raw = raw_params_for_effect("chorus", [0.0, 1.0])
    assert raw == {
        "rate_hz": pytest.approx(0.1),
        "depth": pytest.approx(0.4),
        "mix": pytest.approx(0.5),  # fixo, fora do vetor alvo
    }


def test_raw_params_for_effect_puts_fixed_params_last():
    assert list(raw_params_for_effect("reverb", [0.5])) == ["room_size", "wet_level", "dry_level"]


def test_split_vector_by_effect_slices_in_chain_order():
    chain = ["distortion", "chorus", "slapback_delay"]
    per_effect = split_vector_by_effect(chain, [0.0, 0.0, 1.0, 1.0, 0.5])

    assert list(per_effect) == chain
    assert per_effect["distortion"]["drive_db"] == pytest.approx(5.0)
    assert per_effect["chorus"]["rate_hz"] == pytest.approx(0.1)
    assert per_effect["chorus"]["depth"] == pytest.approx(0.4)
    assert per_effect["slapback_delay"]["delay_seconds"] == pytest.approx(0.2)
    assert per_effect["slapback_delay"]["mix"] == pytest.approx(0.5)


@pytest.mark.parametrize("chain", EFFECT_CHAINS, ids=lambda chain: chain_key(chain))
def test_split_vector_covers_every_predictable_param(chain):
    from gefx.effects.catalog import chain_output_dim, effect_predictable_params

    dim = chain_output_dim(chain_key(chain))
    per_effect = split_vector_by_effect(chain, [0.5] * dim)
    for effect in chain:
        for param in effect_predictable_params(effect):
            assert param["name"] in per_effect[effect]


def test_effect_presence_uses_catalog_order():
    from gefx.effects.catalog import EFFECT_PARAMETER_RANGES

    presence = effect_presence(["chorus"])
    assert list(presence) == list(EFFECT_PARAMETER_RANGES)
    assert presence["chorus"] == 1
    assert sum(presence.values()) == 1


@pytest.mark.parametrize("chain_key_value", sorted(GOLDEN_SUFFIX))
def test_binary_suffix_golden(chain_key_value):
    chain = chain_key_value.split("__")
    assert binary_suffix(chain) == GOLDEN_SUFFIX[chain_key_value]


def test_binary_suffixes_are_unique_and_fixed_width():
    suffixes = [binary_suffix(chain) for chain in EFFECT_CHAINS]
    assert len(set(suffixes)) == len(suffixes)
    assert {len(suffix) for suffix in suffixes} == {8}


def test_binary_suffix_and_the_sidecar_column_share_one_order():
    # O sufixo do nome de arquivo e a coluna `effect_presence` do sidecar sao a
    # mesma informacao; desde a correcao os dois usam a ordem do catalogo.
    import json

    from gefx.data.metadata import RenderRecord

    chain = ["distortion"]
    record = RenderRecord(
        file_name="x.wav", chain_key="distortion", chain_length=1, effect_order=chain,
        effect_presence=effect_presence(chain), normalized_parameter_vector=[0.5],
        raw_parameter_dict={}, source_audio_id="s.wav", random_seed=0,
    )
    column = list(json.loads(record.as_row()["effect_presence"]).values())

    assert binary_suffix(chain) == "10000000"  # distortion e o bit 0
    assert "".join(str(bit) for bit in column) == binary_suffix(chain)


def test_sample_normalized_matrix_shape_and_range():
    matrix = sample_normalized_matrix(np.random.default_rng(0), 4, 3)
    assert matrix.shape == (4, 3)
    assert matrix.dtype == np.float64
    assert ((matrix >= 0.0) & (matrix <= 1.0)).all()


def test_sample_normalized_matrix_is_one_bulk_draw():
    # A ordem dos saques faz parte do contrato de reproducao do dataset: uma
    # matriz (n, p) e um saque so, nao n saques de p.
    expected = np.random.default_rng(7).random((4, 3), dtype=np.float64)
    assert np.array_equal(sample_normalized_matrix(np.random.default_rng(7), 4, 3), expected)
