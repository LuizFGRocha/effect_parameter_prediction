"""Amostrador de tuplas controladas.

O teste central deste arquivo e `test_swap_target_is_exactly_the_missing_corner`:
ele protege a costura que torna barata a extensao da fase 2 (reconstrucao com
troca de codigos). Se ele quebrar, a extensao deixa de ser um modulo a mais e
vira uma reescrita do pipeline de dados.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gefx.disent.sampler import (
    GridIndex,
    SwapTuple,
    as_arrays,
    build_index,
    swap_tuples,
)

CONTENTS = ["c0", "c1", "c2", "c3"]
CONFIGS = [0, 1, 2]
ARMS = ["m0", "m1", "m2"]


def _frame(contents=CONTENTS, configs=CONFIGS, arms=ARMS, split="train"):
    return pd.DataFrame(
        [
            {
                "file_name": f"{content}__{config}.wav",
                "content_id": content,
                "config_index": config,
                "arm": arm,
                "split": split,
            }
            for content, config, arm in itertools.product(contents, configs, arms)
        ]
    )


def test_index_shape_and_size():
    index = build_index(_frame())
    assert index.shape == (len(CONTENTS), len(CONFIGS), len(ARMS))
    assert len(index) == len(CONTENTS) * len(CONFIGS) * len(ARMS)


def test_lookup_round_trips_through_the_labels():
    index = build_index(_frame())
    for content, config, arm in itertools.product(*(range(size) for size in index.shape)):
        row = index.row(content, config, arm)
        labels = index.labels([row])
        assert (labels["content"][0], labels["config"][0], labels["arm"][0]) == (
            content, config, arm
        )


def test_a_hole_in_the_grid_is_refused_at_construction():
    # Um buraco tem de virar erro na montagem do indice, e nao uma tupla
    # silenciosamente errada no meio do treino.
    frame = _frame().drop(index=0)
    with pytest.raises(ValueError, match="nao esta cruzada"):
        GridIndex(frame)


def test_missing_factor_columns_are_refused():
    with pytest.raises(ValueError, match="faltam colunas"):
        GridIndex(_frame().drop(columns=["arm"]))


# --- o invariante da fase 2 ---------------------------------------------------
def test_swap_target_is_exactly_the_missing_corner():
    # decode(z_e(doador), z_c(ancora), arm(ancora)) deve reconstruir
    # x[conteudo(ancora), configuracao(doador), arm(ancora)]. Este teste cobra que
    # essa linha exista e tenha exatamente esses rotulos, para toda tupla sorteada.
    index = build_index(_frame())
    for item in swap_tuples(index, 200, seed=7):
        labels = index.labels([item.anchor, item.effect_donor, item.swap_target])
        anchor, donor, target = (
            {name: labels[name][position] for name in labels} for position in range(3)
        )
        assert target["content"] == anchor["content"]
        assert target["config"] == donor["config"]
        assert target["arm"] == anchor["arm"]


def test_effect_donor_never_shares_the_anchor_content():
    # Se compartilhasse, o alvo da troca seria o proprio doador e a reconstrucao
    # nao exigiria separar efeito de conteudo -- a supervisao seria vazia.
    index = build_index(_frame())
    for item in swap_tuples(index, 200, seed=3):
        labels = index.labels([item.anchor, item.effect_donor])
        assert labels["content"][0] != labels["content"][1]


def test_swap_target_differs_from_the_donor_whenever_the_configs_differ():
    index = build_index(_frame())
    for item in swap_tuples(index, 200, seed=11):
        if item.swap_target != item.effect_donor:
            continue
        # So pode coincidir se conteudo e arm tambem coincidirem, o que o teste
        # anterior ja proibe -- logo, nunca.
        pytest.fail("alvo da troca coincidiu com o doador")


# --- o par positivo do contrastivo -------------------------------------------
def test_positive_shares_the_config_but_not_content_or_arm():
    index = build_index(_frame())
    for item in swap_tuples(index, 200, seed=5):
        labels = index.labels([item.anchor, item.positive])
        assert labels["config"][0] == labels["config"][1]
        assert labels["content"][0] != labels["content"][1]
        assert labels["arm"][0] != labels["arm"][1]


def test_positive_falls_back_to_the_same_arm_when_there_is_only_one():
    # Condicao 1 do protocolo (treino com uma so implementacao): o positivo ainda
    # tem de existir, so que sem variar a implementacao.
    index = build_index(_frame(arms=["m0"]))
    for item in swap_tuples(index, 50, seed=5):
        labels = index.labels([item.anchor, item.positive])
        assert labels["config"][0] == labels["config"][1]
        assert labels["content"][0] != labels["content"][1]
        assert labels["arm"][0] == labels["arm"][1]


# --- recortes -----------------------------------------------------------------
def test_build_index_restricts_to_a_split_and_to_a_set_of_arms():
    frame = pd.concat([_frame(split="train"), _frame(split="query")], ignore_index=True)
    index = build_index(frame, split="train", arms=["m0", "m1"])
    assert index.arms == ["m0", "m1"]
    assert set(index.frame["split"]) == {"train"}


def test_leave_one_arm_out_keeps_the_grid_crossed():
    # O recorte do leave-one-out preserva o cruzamento, entao o alvo da troca
    # continua existindo dentro dele.
    frame = _frame()
    index = build_index(frame, arms=["m0", "m1"])
    for item in swap_tuples(index, 100, seed=13):
        labels = index.labels([item.anchor, item.effect_donor, item.swap_target])
        assert labels["content"][2] == labels["content"][0]
        assert labels["config"][2] == labels["config"][1]
        assert labels["arm"][2] == labels["arm"][0]


def test_empty_slice_is_refused():
    with pytest.raises(ValueError, match="recorte vazio"):
        build_index(_frame(), split="nao-existe")


# --- utilitarios ---------------------------------------------------------------
def test_sampling_is_deterministic_in_the_seed():
    index = build_index(_frame())
    assert swap_tuples(index, 20, seed=1) == swap_tuples(index, 20, seed=1)
    assert swap_tuples(index, 20, seed=1) != swap_tuples(index, 20, seed=2)


def test_as_arrays_gives_one_column_per_role():
    index = build_index(_frame())
    tuples = swap_tuples(index, 10, seed=1)
    arrays = as_arrays(tuples)
    assert set(arrays) == {"anchor", "positive", "effect_donor", "swap_target"}
    for name, values in arrays.items():
        assert values.shape == (10,)
        assert values.tolist() == [getattr(item, name) for item in tuples]


def test_a_single_content_cannot_produce_an_informative_swap():
    index = build_index(_frame(contents=["c0"]))
    with pytest.raises(ValueError, match="ao menos 2 conteudos"):
        swap_tuples(index, 5)


def test_zero_tuples_is_allowed_and_negative_is_not():
    index = build_index(_frame())
    assert swap_tuples(index, 0) == []
    with pytest.raises(ValueError, match="n deve ser"):
        swap_tuples(index, -1)


def test_file_names_follow_the_row_order():
    index = build_index(_frame())
    rows = [3, 1, 0]
    assert index.file_names(rows) == [index.frame["file_name"][row] for row in rows]
