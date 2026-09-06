"""Renderizacao do dataset.

A ordem dos saques do RNG dentro de `render_source_file` e contrato declarado no
docstring do modulo: mudar a sequencia muda o dataset gerado com o mesmo seed.
O teste a fixa sem tocar em audio, substituindo I/O e o backend de efeitos.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from gefx.config import DatasetConfig
from gefx.data import render as render_module
from gefx.data.render import generate_dataset, render_source_file, source_id_to_prefix
from gefx.effects.catalog import EFFECT_CHAINS, chain_key, chain_output_dim
from gefx.effects.parameters import binary_suffix

SR = 100
FRAMES = 1000
SAMPLES_PER_FILE = 2

# Limites do saque do segmento para SR=100, FRAMES=1000, segmento 2 s e 0,5 s
# ignorados de cada lado.
SEGMENT_DRAW_BOUNDS = (50, 751)


@pytest.mark.parametrize(
    "source_id, expected",
    [
        ("les_bridge_fing01.wav", "les_bridge_fing01"),
        ("sub/dir/a.wav", "sub__dir__a"),
        ("sub\\dir\\a.wav", "sub__dir__a"),
        ("maiuscula.WAV", "maiuscula"),
        ("sem_extensao", "sem_extensao"),
        ("a.wav.wav", "a.wav"),  # so a ultima extensao sai
    ],
)
def test_source_id_to_prefix(source_id, expected):
    assert source_id_to_prefix(source_id) == expected


@pytest.fixture
def stub_render_io(monkeypatch):
    """Troca I/O de audio e o backend por dubles; o RNG continua o de verdade."""
    exported = []

    monkeypatch.setattr(
        render_module,
        "load_audio_file",
        lambda file: (np.zeros((1, FRAMES), dtype=np.float32), SR),
    )
    monkeypatch.setattr(render_module, "normalize_loudness", lambda audio, sr: audio)
    monkeypatch.setattr(
        render_module, "build_chain", lambda chain, raw: (lambda audio, sr: audio)
    )
    monkeypatch.setattr(
        render_module, "export_audio", lambda audio, sr, path: exported.append(path)
    )
    return exported


def make_config(**overrides):
    values = dict(
        samples_per_file=SAMPLES_PER_FILE,
        segment_seconds=2.0,
        ignore_start_seconds=0.5,
        ignore_end_seconds=0.5,
    )
    values.update(overrides)
    return DatasetConfig(**values)


def replay_rng_stream(seed_sequence, config):
    """Reproduz a sequencia de saques que `render_source_file` deve fazer."""
    rng = np.random.default_rng(seed_sequence)
    expected = []
    for chain_effects in EFFECT_CHAINS:
        chain_key_value = chain_key(chain_effects)
        matrix = rng.random(
            (config.samples_per_file, chain_output_dim(chain_key_value)), dtype=np.float64
        )
        for sample_index in range(config.samples_per_file):
            if not config.use_full_audio:
                rng.integers(*SEGMENT_DRAW_BOUNDS)
            render_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            expected.append(
                (
                    chain_key_value,
                    [float(value) for value in matrix[sample_index].tolist()],
                    render_seed,
                )
            )
    return expected


def run_render(tmp_path, config, seed=12345):
    seed_sequence = np.random.SeedSequence(seed)
    records = render_source_file(
        tmp_path / "in" / "les_bridge_fing01.wav",
        tmp_path / "in",
        tmp_path / "out",
        seed_sequence,
        config,
    )
    return records, seed_sequence


def test_renders_every_chain_for_every_sample(tmp_path, stub_render_io):
    records, _ = run_render(tmp_path, make_config())
    assert len(records) == len(EFFECT_CHAINS) * SAMPLES_PER_FILE


def test_rng_draw_order_is_the_contract(tmp_path, stub_render_io):
    # Uma matriz por cadeia, depois, por amostra: o saque do segmento e o saque
    # do `random_seed` registrado. Reordenar qualquer um muda o dataset.
    config = make_config()
    records, seed_sequence = run_render(tmp_path, config)

    observed = [
        (record.chain_key, record.normalized_parameter_vector, record.random_seed)
        for record in records
    ]
    assert observed == replay_rng_stream(seed_sequence, config)


def test_same_seed_sequence_gives_identical_records(tmp_path, stub_render_io):
    config = make_config()
    first, _ = run_render(tmp_path, config)
    second, _ = run_render(tmp_path, config)

    assert [record.as_row() for record in first] == [record.as_row() for record in second]


def test_use_full_audio_shifts_the_rng_stream(tmp_path, stub_render_io):
    # Sem o saque do segmento, tudo o que vem depois anda para tras. A primeira
    # matriz e sorteada antes do primeiro segmento, entao so ela coincide.
    segmented, _ = run_render(tmp_path, make_config())
    full, _ = run_render(tmp_path, make_config(use_full_audio=True))

    assert segmented[0].normalized_parameter_vector == full[0].normalized_parameter_vector
    assert segmented[-1].normalized_parameter_vector != full[-1].normalized_parameter_vector


def test_output_file_names_encode_prefix_bits_and_index(tmp_path, stub_render_io):
    records, _ = run_render(tmp_path, make_config())
    names = [record.file_name for record in records]

    for name in names:
        assert re.fullmatch(r"les_bridge_fing01__[01]{8}__s\d{4}\.wav", name)
    assert len(set(names)) == len(names)

    distortion = [record for record in records if record.chain_key == "distortion"]
    assert [record.file_name for record in distortion] == [
        "les_bridge_fing01__10000000__s0000.wav",
        "les_bridge_fing01__10000000__s0001.wav",
    ]


def test_records_carry_a_coherent_sidecar_row(tmp_path, stub_render_io):
    records, _ = run_render(tmp_path, make_config())
    record = next(item for item in records if item.chain_key == "distortion__chorus")

    assert record.effect_order == ["distortion", "chorus"]
    assert record.chain_length == 2
    assert len(record.normalized_parameter_vector) == chain_output_dim("distortion__chorus")
    assert set(record.raw_parameter_dict) == {"distortion", "chorus"}
    assert record.source_audio_id == "les_bridge_fing01.wav"
    assert binary_suffix(["distortion", "chorus"]) in record.file_name


def test_chain_directories_are_created(tmp_path, stub_render_io):
    run_render(tmp_path, make_config())
    for chain_effects in EFFECT_CHAINS:
        assert (tmp_path / "out" / chain_key(chain_effects)).is_dir()


def test_exports_one_wav_per_record(tmp_path, stub_render_io):
    records, _ = run_render(tmp_path, make_config())
    assert [path.name for path in stub_render_io] == [record.file_name for record in records]


def test_generate_dataset_requires_source_audio(tmp_path):
    empty = tmp_path / "vazio"
    empty.mkdir()
    config = make_config(input_dir=str(empty), output_dir=str(tmp_path / "out"))

    with pytest.raises(RuntimeError, match="No wav files found under"):
        generate_dataset(config)
