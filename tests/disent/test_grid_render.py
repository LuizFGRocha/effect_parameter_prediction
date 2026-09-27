"""Selecao de conteudo do renderizador da grade.

Nao renderiza nada: `_render_chunk` carrega VST3 e fica fora da suite. O que se
testa aqui e a parte que decide *o que* renderizar, que e onde mora o
determinismo do dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.audio import export_audio
from gefx.disent.render import RenderOptions, _chunks, content_items

SR = 44100


@pytest.fixture
def recordings(tmp_path):
    """Gravacoes sinteticas de 11,5 s, como as de Rossi."""

    def build(n=20, seconds=11.5):
        folder = tmp_path / "wavs"
        folder.mkdir(exist_ok=True)
        rng = np.random.default_rng(0)
        for index in range(n):
            audio = (rng.standard_normal((1, int(SR * seconds))) * 0.05).astype(np.float32)
            export_audio(audio, SR, folder / f"rec{index:03d}.wav")
        return folder

    return build


def _options(folder, **overrides):
    defaults = dict(input_dir=folder, n_contents=8, workers=2)
    return RenderOptions(**{**defaults, **overrides})


def test_content_items_pick_the_requested_number(recordings):
    items = content_items(_options(recordings()))
    assert len(items) == 8 * 5
    assert len({item.content_id for item in items}) == 8 * 5
    assert len({item.source_path for item in items}) == 8


def test_segments_of_a_recording_are_consecutive_and_share_the_split(recordings):
    # Particao por gravacao: dois trechos da mesma execucao em splits diferentes
    # deixariam a busca acertar reconhecendo a execucao.
    frames = int(round(2.0 * SR))
    items = content_items(_options(recordings()))
    by_file = {}
    for item in items:
        by_file.setdefault(item.source_path, []).append(item)
    for segments in by_file.values():
        starts = [item.segment_start for item in segments]
        assert [b - a for a, b in zip(starts, starts[1:])] == [frames] * 4
        assert len({item.split for item in segments}) == 1


def test_asking_for_more_content_than_available_is_refused(recordings):
    with pytest.raises(ValueError, match="menos que os"):
        content_items(_options(recordings(n=6), n_contents=8))


def test_segment_start_is_deterministic_in_the_seed(recordings):
    folder = recordings()
    first = content_items(_options(folder, seed=1))
    assert [item.segment_start for item in first] == [
        item.segment_start for item in content_items(_options(folder, seed=1))
    ]
    assert [item.segment_start for item in first] != [
        item.segment_start for item in content_items(_options(folder, seed=2))
    ]


def test_segment_start_does_not_depend_on_the_arm_list(recordings):
    # E o que garante que o mesmo content_id de o mesmo trecho tocado em toda a
    # grade: o sorteio olha so o indice do conteudo.
    folder = recordings()
    one = content_items(_options(folder, arms=["a"]))
    other = content_items(_options(folder, arms=["a", "b", "c"]))
    assert [item.segment_start for item in one] == [item.segment_start for item in other]


def test_segment_stays_inside_the_recording_with_the_edge_margin(recordings):
    seconds, margin = 11.5, int(round(0.5 * SR))
    frames = int(round(2.0 * SR))
    for item in content_items(_options(recordings(seconds=seconds))):
        assert item.segment_start >= margin
        assert item.segment_start + frames <= int(SR * seconds) - margin


def test_a_recording_too_short_is_refused(recordings):
    with pytest.raises(ValueError, match="curta demais"):
        content_items(_options(recordings(seconds=10.5), n_contents=3))


def test_every_content_lands_in_exactly_one_split(recordings):
    items = content_items(_options(recordings()))
    assert {item.split for item in items} == {"train", "catalog", "query"}
    assert len({item.content_id for item in items}) == len(items)


def test_split_is_stable_under_a_different_render_seed(recordings):
    # A particao responde a `split_seed`, nao a `seed`: da para re-sortear os
    # trechos sem embaralhar treino, catalogo e consulta.
    folder = recordings()
    base = {item.content_id: item.split for item in content_items(_options(folder, seed=1))}
    same = {item.content_id: item.split for item in content_items(_options(folder, seed=2))}
    other = {
        item.content_id: item.split
        for item in content_items(_options(folder, split_seed=99))
    }
    assert base == same
    assert base != other


# --- particionamento do trabalho ---------------------------------------------
def test_chunks_cover_everything_without_repeating():
    items = list(range(10))
    for workers in range(1, 12):
        chunks = _chunks(items, workers)
        assert [item for chunk in chunks for item in chunk] == items


def test_chunks_never_come_back_empty():
    for workers in range(1, 12):
        assert all(_chunks(list(range(10)), workers))
