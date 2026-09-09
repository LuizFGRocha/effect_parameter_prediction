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
    batch_stream,
    build_index,
    class_balanced_batch,
    permute_configs,
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


def test_the_batch_carries_the_swap_target_of_the_phase_two_seam():
    """Cada alvo tem o conteudo e a implementacao da ancora e a configuracao do
    doador. E a mesma propriedade que `swap_tuples` garante -- e ela tem de
    sobreviver ao caminho que o treino da fase 1 realmente usa."""
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(3), 3, 3)
    anchors = index.labels(batch.rows)
    donors = index.labels(batch.effect_donor)
    targets = index.labels(batch.swap_target)
    assert np.array_equal(targets["content"], anchors["content"])
    assert np.array_equal(targets["arm"], anchors["arm"])
    assert np.array_equal(targets["config"], donors["config"])


def test_the_effect_donor_never_shares_the_anchor_content_in_a_batch():
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(4), 3, 3)
    assert not np.any(index.labels(batch.effect_donor)["content"] == batch.content)


def test_a_batch_with_a_single_view_per_config_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="positivo"):
        class_balanced_batch(index, np.random.default_rng(0), 2, 1)


def test_asking_for_more_configs_than_the_grid_has_is_refused():
    index = build_index(_frame())
    with pytest.raises(ValueError, match="configuracoes"):
        class_balanced_batch(index, np.random.default_rng(0), len(CONFIGS) + 1, 2)


def test_the_batch_stream_is_deterministic_in_the_seed():
    index = build_index(_frame())
    first = [b.rows for b in batch_stream(index, 3, 2, 2, seed=7)]
    second = [b.rows for b in batch_stream(index, 3, 2, 2, seed=7)]
    other = [b.rows for b in batch_stream(index, 3, 2, 2, seed=8)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_the_views_of_one_config_vary_in_content():
    """Se as vistas repetissem conteudo, o batch ensinaria invariancia a
    implementacao sem nunca mostrar conteudo diferente sob a mesma configuracao."""
    index = build_index(_frame())
    batch = class_balanced_batch(index, np.random.default_rng(5), 2, 4)
    for label in set(batch.config):
        where = batch.config == label
        assert len(set(batch.content[where])) == int(where.sum())


# --- controle de permutacao ---------------------------------------------------
def test_the_permutation_keeps_the_grid_crossed_and_the_batches_balanced():
    """A forma do embaralhamento e o ponto do controle: rotulo sorteado ao acaso
    deixaria o contrastivo sem positivos, e ai o controle mediria "treino sem
    gradiente" em vez de "treino com rotulo sem sentido"."""
    frame = _frame()
    permuted = permute_configs(frame, seed=0)
    index = build_index(permuted)
    assert index.shape == (len(CONTENTS), len(CONFIGS), len(ARMS))
    batch = class_balanced_batch(index, np.random.default_rng(0), 2, 2)
    counts = np.bincount(batch.config)
    assert all(counts[label] >= 2 for label in batch.config)


def test_the_permutation_moves_the_configuration_and_not_the_audio():
    frame = _frame()
    permuted = permute_configs(frame, seed=0)
    assert permuted["file_name"].equals(frame["file_name"])
    assert permuted["content_id"].equals(frame["content_id"])
    assert permuted["arm"].equals(frame["arm"])
    assert not permuted["config_index"].equals(frame["config_index"])


def test_the_permutation_preserves_the_configuration_marginal():
    frame = _frame()
    permuted = permute_configs(frame, seed=1)
    assert sorted(permuted["config_index"]) == sorted(frame["config_index"])


def test_the_permutation_keeps_each_row_internally_coherent():
    """Rotulo, nivel e valor tem de andar juntos: uma linha com o `config_index`
    de uma configuracao e o `drive_level` de outra nao e um controle, e um bug."""
    frame = _frame()
    frame["drive_level"] = frame["config_index"] * 10
    permuted = permute_configs(frame, seed=2)
    assert (permuted["drive_level"] == permuted["config_index"] * 10).all()


def test_the_permutation_is_deterministic_in_the_seed():
    frame = _frame()
    a = permute_configs(frame, seed=3)["config_index"].to_numpy()
    b = permute_configs(frame, seed=3)["config_index"].to_numpy()
    c = permute_configs(frame, seed=4)["config_index"].to_numpy()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_the_permutation_needs_the_grouping_columns():
    with pytest.raises(ValueError, match="agrupar"):
        permute_configs(pd.DataFrame({"config_index": [0, 1]}))
