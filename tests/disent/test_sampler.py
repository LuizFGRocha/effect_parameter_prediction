"""Indice da grade cruzada e batches balanceados por configuracao."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gefx.disent.sampler import GridIndex, build_index, class_balanced_batch

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


def test_build_index_restricts_to_a_split_and_to_a_set_of_arms():
    frame = pd.concat([_frame(split="train"), _frame(split="query")], ignore_index=True)
    index = build_index(frame, split="train", arms=["m0", "m1"])
    assert index.arms == ["m0", "m1"]
    assert set(index.frame["split"]) == {"train"}


def test_empty_slice_is_refused():
    with pytest.raises(ValueError, match="recorte vazio"):
        build_index(_frame(), split="nao-existe")


# --- utilitarios ---------------------------------------------------------------
def test_file_names_follow_the_row_order():
    index = build_index(_frame())
    rows = [3, 1, 0]
    assert index.file_names(rows) == [index.frame["file_name"][row] for row in rows]


# --- batches balanceados por classe ------------------------------------------
def test_the_batch_has_the_asked_shape_and_the_configs_are_distinct():
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(0), 2, 3)
    assert len(batch) == 6
    assert len(set(batch.config)) == 2
    assert list(np.sort(np.bincount(batch.config, minlength=3))[-2:]) == [3, 3]


def test_every_anchor_in_the_batch_has_at_least_one_positive():
    """E a condicao de o contrastivo supervisionado produzir termo. Sem ela a
    perda passaria a medir a sorte da amostragem, e nao o que a rede aprendeu."""
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(1), 3, 2)
    counts = np.bincount(batch.config)
    assert all(counts[label] >= 2 for label in batch.config)


def test_the_batch_labels_match_the_rows_it_carries():
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(2), 2, 2)
    labels = index.labels(batch.rows)
    assert np.array_equal(labels["config"], batch.config)
    assert np.array_equal(labels["content"], batch.content)
    assert np.array_equal(labels["arm"], batch.arm)


def test_a_batch_with_a_single_view_per_config_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="positivo"):
        class_balanced_batch(index, np.random.default_rng(0), 2, 1)


def test_asking_for_more_configs_than_the_grid_has_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="configuracoes"):
        class_balanced_batch(index, np.random.default_rng(0), len(CONFIGS) + 1, 2)


def test_the_views_of_one_config_vary_in_content():
    """Se as vistas repetissem conteudo, o batch ensinaria invariancia a
    implementacao sem nunca mostrar conteudo diferente sob a mesma configuracao."""
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(5), 2, 4)
    for label in set(batch.config):
        where = batch.config == label
        assert len(set(batch.content[where])) == int(where.sum())




def test_the_batch_is_deterministic_in_the_generator():
    index = build_index(_frame())
    first = class_balanced_batch(index, np.random.default_rng(7), 2, 2).rows
    again = class_balanced_batch(index, np.random.default_rng(7), 2, 2).rows
    other = class_balanced_batch(index, np.random.default_rng(8), 2, 2).rows
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
