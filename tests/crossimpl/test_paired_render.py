"""Renderizacao pareada.

O invariante central do POC II: o mesmo vetor normalizado e o mesmo segmento de
audio vao para todos os bracos. Sem isso, uma diferenca entre implementacoes
podia ser so um conjunto de avaliacao diferente.

I/O de audio e plugins sao substituidos por dubles; o RNG e o de verdade.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gefx.crossimpl import render as render_module
from gefx.crossimpl.render import RenderOptions, render
from gefx.data.metadata import METADATA_COLUMNS
from gefx.effects.catalog import EFFECT_PARAMETER_RANGES
from gefx.effects.registry import REFERENCE_ARM

SR = 100
FRAMES = 1000


@pytest.fixture
def stub_io(monkeypatch, tmp_path):
    """Troca audio e plugins por dubles e captura o que iria para o sidecar."""
    written = {}

    monkeypatch.setattr(
        render_module, "iter_wav_files", lambda root: [tmp_path / "a.wav", tmp_path / "b.wav"]
    )
    monkeypatch.setattr(
        render_module, "load_audio_file",
        lambda path: (np.zeros((1, FRAMES), dtype=np.float32), SR),
    )
    monkeypatch.setattr(render_module, "normalize_loudness", lambda audio, sr: audio)
    monkeypatch.setattr(render_module, "build_chain", lambda chain, raw: (lambda a, sr: a))
    monkeypatch.setattr(render_module, "export_audio", lambda audio, sr, path: None)
    monkeypatch.setattr(render_module, "load_arm", lambda spec: (object(), False))
    monkeypatch.setattr(render_module, "process_mono", lambda p, s, segment, sr: segment)
    monkeypatch.setattr(render_module, "set_parameter", lambda plugin, name, value: None)
    monkeypatch.setattr(
        render_module, "append_metadata_csv",
        lambda root, rows: written.setdefault(root.name, []).extend(rows),
    )
    return written


def make_options(tmp_path, **overrides):
    values = dict(
        input_dir=str(tmp_path), output_root=str(tmp_path / "out"), n=3,
        segment_seconds=2.0, seed=7, effects=["distortion"],
    )
    values.update(overrides)
    return RenderOptions(**values)


def normalized(rows):
    return [json.loads(row["normalized_parameter_vector"]) for row in rows]


def test_every_arm_gets_the_same_normalized_vectors(stub_io, tmp_path):
    # E isto que torna a comparacao entre bracos legitima: o ground truth e
    # literalmente o mesmo, sorteado uma vez por efeito e reusado.
    render(make_options(tmp_path, arms=[REFERENCE_ARM, "lsp"]))

    assert set(stub_io) == {REFERENCE_ARM, "lsp"}
    assert normalized(stub_io[REFERENCE_ARM]) == normalized(stub_io["lsp"])


def test_every_arm_gets_the_same_source_audio(stub_io, tmp_path):
    render(make_options(tmp_path, arms=[REFERENCE_ARM, "lsp"]))

    for key in ("source_audio_id", "file_name"):
        assert [row[key] for row in stub_io[REFERENCE_ARM]] == [
            row[key] for row in stub_io["lsp"]
        ]


def test_rows_follow_the_sidecar_schema(stub_io, tmp_path):
    render(make_options(tmp_path))
    rows = stub_io[REFERENCE_ARM]

    assert len(rows) == 3
    for row in rows:
        assert list(row) == METADATA_COLUMNS
        assert row["chain_key"] == "distortion"
        assert row["chain_length"] == 1
        assert json.loads(row["effect_order"]) == ["distortion"]
        assert row["random_seed"] == 7


def test_file_names_encode_source_effect_and_index(stub_io, tmp_path):
    render(make_options(tmp_path))
    names = [row["file_name"] for row in stub_io[REFERENCE_ARM]]

    assert all(name.endswith(".wav") for name in names)
    assert [name.split("__")[1] for name in names] == ["distortion"] * 3
    assert [name.split("__")[2] for name in names] == ["00000.wav", "00001.wav", "00002.wav"]


def test_arms_without_a_registry_entry_are_skipped(stub_io, tmp_path):
    # `slapback_delay` ficou de fora do registro de proposito.
    render(make_options(tmp_path, effects=["slapback_delay"], arms=[REFERENCE_ARM, "lsp"]))
    assert set(stub_io) == {REFERENCE_ARM}


@pytest.mark.parametrize(
    "legacy, expected",
    [(False, list(EFFECT_PARAMETER_RANGES)), (True, sorted(EFFECT_PARAMETER_RANGES))],
)
def test_legacy_controls_the_sidecar_column_order(stub_io, tmp_path, legacy, expected):
    render(make_options(tmp_path, legacy=legacy))
    column = stub_io[REFERENCE_ARM][0]["effect_presence"]
    assert list(json.loads(column)) == expected


def test_requires_source_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(render_module, "iter_wav_files", lambda root: [])
    with pytest.raises(SystemExit, match="Nenhum wav em"):
        render(make_options(tmp_path))
