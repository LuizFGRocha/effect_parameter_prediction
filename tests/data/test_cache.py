"""Cache de features. `extract_feature` e substituido por um stub, entao nada
aqui carrega librosa nem le audio de verdade."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gefx.data import cache as cache_module
from gefx.data.cache import (
    build_cache,
    cache_exists,
    ensure_feature_cache,
    feature_cache_file,
    file_names_cache_file,
    load_cache,
    save_cache,
)


@pytest.fixture
def stub_extract(monkeypatch):
    """Feature deterministica derivada do nome: a ordem das linhas fica legivel."""
    calls = []

    def extract(path, feature_name):
        calls.append(path.name)
        return np.full((2, 3), float(ord(path.stem[-1])), dtype=np.float64)

    monkeypatch.setattr(cache_module, "extract_feature", extract)
    return calls


@pytest.fixture
def chain_folder(tmp_path):
    folder = tmp_path / "distortion"
    folder.mkdir()
    for name in ["c.wav", "a.wav", "b.wav"]:
        (folder / name).touch()
    return folder


def test_build_cache_orders_rows_by_sorted_wav_name(chain_folder, stub_extract):
    features, file_names = build_cache(chain_folder, "Spec")

    assert file_names.tolist() == ["a.wav", "b.wav", "c.wav"]
    assert stub_extract == ["a.wav", "b.wav", "c.wav"]
    assert features.shape == (3, 2, 3)
    assert [float(row[0, 0]) for row in features] == [ord("a"), ord("b"), ord("c")]


def test_build_cache_requires_wavs(tmp_path, stub_extract):
    empty = tmp_path / "vazia"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No wav files found in"):
        build_cache(empty, "Spec")


def test_build_cache_ignores_non_wav_files(chain_folder, stub_extract):
    (chain_folder / "Spec.npz").touch()
    (chain_folder / "nota.txt").touch()
    _, file_names = build_cache(chain_folder, "Spec")
    assert file_names.tolist() == ["a.wav", "b.wav", "c.wav"]


def test_save_load_round_trip(chain_folder):
    features = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    names = np.array(["a.wav", "b.wav"])

    save_cache(chain_folder, "Spec", features, names)
    loaded_features, loaded_names = load_cache(chain_folder, "Spec")

    assert np.allclose(loaded_features, features)
    assert loaded_features.dtype == np.float32  # a leitura rebaixa para float32
    assert loaded_names.tolist() == names.tolist()


def test_cache_exists_requires_both_artifacts(chain_folder):
    assert not cache_exists(chain_folder, "Spec")

    feature_cache_file(chain_folder, "Spec").touch()
    assert not cache_exists(chain_folder, "Spec")  # falta o file_names.json

    file_names_cache_file(chain_folder).touch()
    assert cache_exists(chain_folder, "Spec")


def test_cache_is_per_feature(chain_folder, stub_extract):
    ensure_feature_cache(chain_folder, "Spec")
    assert cache_exists(chain_folder, "Spec")
    assert not cache_exists(chain_folder, "MFCC40")


def test_ensure_feature_cache_builds_once_then_reads(chain_folder, stub_extract):
    first, names = ensure_feature_cache(chain_folder, "Spec")
    assert len(stub_extract) == 3

    second, names_again = ensure_feature_cache(chain_folder, "Spec")
    assert len(stub_extract) == 3  # segunda chamada nao extraiu nada
    assert np.allclose(first, second)
    assert names_again.tolist() == names.tolist()


def test_force_rebuild_ignores_an_existing_cache(chain_folder, stub_extract):
    ensure_feature_cache(chain_folder, "Spec")
    (chain_folder / "d.wav").touch()

    features, names = ensure_feature_cache(chain_folder, "Spec", force_rebuild=True)
    assert names.tolist() == ["a.wav", "b.wav", "c.wav", "d.wav"]
    assert len(features) == 4
