"""Indice da grade cruzada e batches balanceados por classe."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gefx.training.sampler import GridIndex, build_index, class_balanced_batch

CONTENTS = ["c0", "c1", "c2", "c3"]
LEVELS = [0, 1, 2]
ARMS = ["m0", "m1", "m2"]


def _frame(contents=CONTENTS, levels=LEVELS, arms=ARMS, split="train"):
    return pd.DataFrame(
        [
            {
                "file_name": f"{content}__{level}.wav",
                "content_id": content,
                "drive_level": level,
                "arm": arm,
                "split": split,
            }
            for content, level, arm in itertools.product(contents, levels, arms)
        ]
    )


def _labels(index, rows):
    """Os rotulos (conteudo, nivel) das linhas, como o treino os le."""
    return index.content_label[rows], index.level_label[rows]


def test_the_lookup_and_the_labels_agree():
    index = build_index(_frame())
    assert index.lookup.shape == (len(CONTENTS), len(LEVELS), len(ARMS))
    assert len(index) == len(CONTENTS) * len(LEVELS) * len(ARMS)
    for content, level, arm in itertools.product(*(range(size) for size in index.lookup.shape)):
        row = index.lookup[content, level, arm]
        assert _labels(index, row) == (content, level)
        assert index.frame["arm"][row] == ARMS[arm]


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


# --- batches balanceados por classe ------------------------------------------
def _batch(index, seed, classes, views, by="level"):
    """As linhas do batch e os rotulos (conteudo, nivel) delas."""
    rows = class_balanced_batch(index, np.random.default_rng(seed), classes, views, by=by)
    content, level = _labels(index, rows)
    return rows, content, level


def test_the_batch_has_the_asked_shape_and_the_levels_are_distinct():
    index = build_index(_frame())
    rows, _, level = _batch(index, 0, 2, 3)
    assert len(rows) == 6
    assert len(set(level)) == 2
    assert list(np.sort(np.bincount(level, minlength=3))[-2:]) == [3, 3]


def test_every_anchor_in_the_batch_has_at_least_one_positive():
    """E a condicao de o contrastivo supervisionado produzir termo. Sem ela a
    perda passaria a medir a sorte da amostragem, e nao o que a rede aprendeu."""
    index = build_index(_frame())
    _, _, level = _batch(index, 1, 3, 2)
    counts = np.bincount(level)
    assert all(counts[label] >= 2 for label in level)


def test_a_batch_with_a_single_view_per_class_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="positivo"):
        class_balanced_batch(index, np.random.default_rng(0), 2, 1)


def test_asking_for_more_levels_than_the_grid_has_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="niveis"):
        class_balanced_batch(index, np.random.default_rng(0), len(LEVELS) + 1, 2)


def test_the_views_of_one_level_vary_in_content():
    """Se as vistas repetissem conteudo, o batch ensinaria invariancia a
    implementacao sem nunca mostrar conteudo diferente sob o mesmo nivel."""
    index = build_index(_frame())
    _, content, level = _batch(index, 5, 2, 4)
    for label in set(level):
        where = level == label
        assert len(set(content[where])) == int(where.sum())


def test_the_batch_is_deterministic_in_the_generator():
    index = build_index(_frame())
    first = class_balanced_batch(index, np.random.default_rng(7), 2, 2)
    again = class_balanced_batch(index, np.random.default_rng(7), 2, 2)
    other = class_balanced_batch(index, np.random.default_rng(8), 2, 2)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


# --- batch por trecho (cabeca de conteudo) ------------------------------------
def test_by_content_puts_each_chosen_content_in_every_level():
    """Com K igual ao numero de niveis, cada trecho aparece em todos: P trechos x
    K niveis, e cada nivel P vezes no batch."""
    index = build_index(_frame())
    rows, content, level = _batch(index, 0, 2, len(LEVELS), by="content")
    assert len(rows) == 2 * len(LEVELS) and len(set(content)) == 2
    assert len(set(zip(content, level))) == len(rows)
    assert list(np.bincount(level)) == [2] * len(LEVELS)


def test_every_anchor_of_a_by_content_batch_has_positives_of_both_factors():
    index = build_index(_frame())
    _, content, level = _batch(index, 1, 3, len(LEVELS), by="content")
    for labels in (level, content):
        counts = np.bincount(labels)
        assert all(counts[label] >= 2 for label in labels)


def test_the_by_level_batch_did_not_change_with_the_parameter():
    """As execucoes ja treinadas sairam deste sorteio: o padrao tem de dar as
    mesmas linhas de antes, na mesma ordem."""
    index = build_index(_frame())
    rng = np.random.default_rng(11)
    chosen = rng.choice(len(LEVELS), size=2, replace=False)
    expected = []
    for level in chosen:
        contents = rng.choice(len(CONTENTS), size=3, replace=False)
        arms = rng.integers(0, len(ARMS), size=3)
        expected += [index.lookup[c, level, a] for c, a in zip(contents, arms)]
    assert list(class_balanced_batch(index, np.random.default_rng(11), 2, 3)) == expected


def test_an_unknown_batch_class_and_too_many_contents_are_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="desconhecida"):
        class_balanced_batch(index, np.random.default_rng(0), 2, 2, by="arm")
    with pytest.raises(ValueError, match="trechos"):
        class_balanced_batch(index, np.random.default_rng(0), len(CONTENTS) + 1, 2, by="content")
