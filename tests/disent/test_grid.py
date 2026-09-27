"""Grade de configuracoes e particao de conteudo."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.grid import (
    DEFAULT_SPLIT_FRACTIONS,
    SPLITS,
    Config,
    all_configs,
    config_from_index,
    config_index,
    parse_config_key,
    render_name,
    split_contents,
)

DRIVE_LEVELS = 8


def test_grid_has_the_declared_size():
    configs = all_configs(DRIVE_LEVELS)
    assert len(configs) == DRIVE_LEVELS
    assert len(set(configs)) == len(configs)


def test_index_and_config_are_inverses():
    # A ordem canonica vira rotulo de classe do contrastivo e ordem das entradas
    # do catalogo; se ela mudar, modelos treinados param de fazer sentido.
    for index, config in enumerate(all_configs(DRIVE_LEVELS)):
        assert config_index(config) == index
        assert config_from_index(index) == config


def test_canonical_order_is_the_drive_level():
    assert all_configs(3) == [Config(0), Config(1), Config(2)]


def test_config_key_round_trips():
    for config in all_configs(DRIVE_LEVELS):
        assert parse_config_key(config.key) == config


@pytest.mark.parametrize("bad", ["", "d", "t1", "d0t0", "x0", "d-1"])
def test_parse_config_key_rejects_junk(bad):
    with pytest.raises(ValueError, match="invalida"):
        parse_config_key(bad)


# --- nome do render -----------------------------------------------------------
def test_render_name_is_identical_across_arms():
    # O nome nao carrega o arm: e o que permite ao oraculo parear <A>/<n> com
    # <B>/<n> sabendo que so a implementacao mudou.
    config = Config(3)
    assert render_name("src", config, 7) == "src__d3__00007.wav"
    assert parse_config_key(render_name("src", config, 7).split("__")[1]) == config


# --- particao ------------------------------------------------------------------
def _contents(n: int = 100):
    return [f"c{index:03d}" for index in range(n)]


def test_split_sizes_follow_the_fractions():
    split = split_contents(_contents())
    assert {name: len(items) for name, items in split.items()} == {
        "train": 60, "catalog": 20, "query": 20
    }


def test_splits_are_disjoint_and_cover_everything():
    split = split_contents(_contents())
    joined = [item for name in SPLITS for item in split[name]]
    assert sorted(joined) == sorted(_contents())
    assert len(joined) == len(set(joined))


def test_split_is_deterministic_and_seed_dependent():
    assert split_contents(_contents(), seed=1) == split_contents(_contents(), seed=1)
    assert split_contents(_contents(), seed=1) != split_contents(_contents(), seed=2)


def test_split_shuffles_instead_of_slicing_the_sorted_list():
    # Os nomes de Rossi vem agrupados por guitarra/captador/tecnica; fatiar a
    # lista ordenada poria uma guitarra inteira num split so e confundiria
    # "outra execucao" com "outro instrumento".
    split = split_contents(_contents())
    assert split["train"] != sorted(_contents())[:60]


def test_query_absorbs_the_rounding_so_nothing_is_dropped():
    for size in range(3, 40):
        split = split_contents(_contents(size))
        assert sum(len(items) for items in split.values()) == size


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError, match="somar 1.0"):
        split_contents(_contents(), fractions={"train": 0.5, "catalog": 0.2, "query": 0.2})
    with pytest.raises(ValueError, match="exatamente"):
        split_contents(_contents(), fractions={"train": 0.5, "catalog": 0.5})


def test_split_rejects_duplicates_and_tiny_inputs():
    with pytest.raises(ValueError, match="repetido"):
        split_contents(["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="ao menos"):
        split_contents(["a", "b"])


def test_default_fractions_sum_to_one():
    assert np.isclose(sum(DEFAULT_SPLIT_FRACTIONS.values()), 1.0)


def test_every_split_gets_at_least_one_content_at_the_minimum_size():
    # A pre-condicao anuncia `len(SPLITS)` conteudos; ela tem de ser suficiente,
    # e nao so necessaria.
    split = split_contents(["a", "b", "c"])
    assert {name: len(items) for name, items in split.items()} == {
        "train": 1, "catalog": 1, "query": 1
    }
