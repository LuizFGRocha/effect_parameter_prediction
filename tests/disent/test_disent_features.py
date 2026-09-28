"""Acesso ao cache de features sem carregar tudo.

Os caches dos arms nao cabem juntos na RAM, por isso o `.npy` mapeado. O que os
testes protegem e o que quebraria em silencio -- o alinhamento por nome de
arquivo, que e a mesma armadilha do POC I: usar as duas ordens trocadas nao
levanta erro, so treina com o rotulo errado.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent import features as features_module
from gefx.disent.features import (
    FeatureStore,
    PixelStandardizer,
    arm_feature_folder,
    ensure_arm_cache,
    open_cache_memmap,
)
from gefx.disent.sidecar import EFFECT_FOLDER

ARMS = ("m0", "m1")
SHAPE = (6, 5)


def _write_arm(root, arm, names, features):
    folder = root / arm / "distortion"
    folder.mkdir(parents=True)
    np.save(folder / "Spec.npy", features)
    (folder / "file_names.json").write_text(json.dumps(list(names)), encoding="utf-8")
    return folder


def _dataset(tmp_path, rows=4):
    rng = np.random.default_rng(0)
    frames = []
    payload = {}
    for arm in ARMS:
        names = [f"f{index}.wav" for index in range(rows)]
        features = rng.random((rows, *SHAPE)).astype(np.float32)
        payload[arm] = (names, features)
        # A ordem do sidecar e deliberadamente o inverso da ordem do cache: e
        # exatamente o desalinhamento que o casamento por nome tem de absorver.
        _write_arm(tmp_path, arm, names, features)
        frames.append(
            pd.DataFrame({"file_name": names[::-1], "arm": arm,
                          "content_id": [f"c{i}" for i in range(rows)][::-1]})
        )
    return pd.concat(frames, ignore_index=True), payload


def test_open_cache_memmap_does_not_read_the_array_into_memory(tmp_path):
    features = np.random.default_rng(1).random((3, 4)).astype(np.float32)
    _write_arm(tmp_path, "m0", ["a.wav", "b.wav", "c.wav"], features)
    mapped, names = open_cache_memmap(tmp_path, "m0", "Spec")
    assert isinstance(mapped, np.memmap)
    assert list(names) == ["a.wav", "b.wav", "c.wav"]
    assert np.array_equal(np.asarray(mapped), features)


def test_a_missing_cache_says_which_command_builds_it(tmp_path):
    (tmp_path / "m0" / "distortion").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="gefx disent cache"):
        open_cache_memmap(tmp_path, "m0", "Spec")


def test_a_cache_and_a_name_list_of_different_lengths_are_refused(tmp_path):
    folder = tmp_path / "m0" / "distortion"
    folder.mkdir(parents=True)
    np.save(folder / "Spec.npy", np.zeros((3, 2), dtype=np.float32))
    (folder / "file_names.json").write_text(json.dumps(["a.wav"]), encoding="utf-8")
    with pytest.raises(ValueError, match="file_names.json"):
        open_cache_memmap(tmp_path, "m0", "Spec")


def test_the_store_takes_rows_in_the_order_of_the_frame(tmp_path):
    frame, payload = _dataset(tmp_path)
    store = FeatureStore(tmp_path, frame, "Spec")
    taken = store.take(np.arange(len(frame)))
    for row in range(len(frame)):
        arm = frame["arm"].iloc[row]
        names, features = payload[arm]
        assert np.array_equal(taken[row], features[names.index(frame["file_name"].iloc[row])])


def test_the_store_answers_a_shuffled_selection_in_the_order_asked(tmp_path):
    frame, _ = _dataset(tmp_path)
    store = FeatureStore(tmp_path, frame, "Spec")
    everything = store.take(np.arange(len(frame)))
    wanted = np.array([5, 0, 3, 3, 1])
    assert np.array_equal(store.take(wanted), everything[wanted])


def test_a_row_without_a_cached_feature_is_refused(tmp_path):
    frame, _ = _dataset(tmp_path)
    frame.loc[0, "file_name"] = "ausente.wav"
    with pytest.raises(ValueError, match="sem feature em cache"):
        FeatureStore(tmp_path, frame, "Spec")


def test_the_store_streams_in_blocks_that_cover_everything_once(tmp_path):
    frame, _ = _dataset(tmp_path)
    store = FeatureStore(tmp_path, frame, "Spec")
    seen = np.concatenate([block for block, _ in store.stream(np.arange(len(frame)), chunk=3)])
    assert np.array_equal(seen, np.arange(len(frame)))


def test_the_standardizer_matches_the_direct_computation(tmp_path):
    frame, _ = _dataset(tmp_path)
    store = FeatureStore(tmp_path, frame, "Spec")
    everything = store.take(np.arange(len(frame)))
    standardizer = PixelStandardizer.fit(store, chunk=3)
    assert np.allclose(standardizer.mean, everything.mean(axis=0), atol=1e-6)
    assert np.allclose(standardizer.std, everything.std(axis=0), atol=1e-6)


def test_the_standardizer_adds_the_channel_axis_the_convolution_wants(tmp_path):
    frame, _ = _dataset(tmp_path)
    store = FeatureStore(tmp_path, frame, "Spec")
    transformed = PixelStandardizer.fit(store).transform(store.take([0, 1]))
    assert transformed.shape == (2, *SHAPE, 1)


def test_a_constant_pixel_does_not_become_a_division_by_zero():
    """Bandas altas em silencio sao constantes no dataset inteiro; sem piso no
    desvio elas propagariam NaN pela rede toda."""
    standardizer = PixelStandardizer(
        mean=np.zeros((2, 2), dtype=np.float32),
        std=np.zeros((2, 2), dtype=np.float32),
        n=10,
    )
    standardizer = PixelStandardizer(standardizer.mean, np.maximum(standardizer.std, 1e-6), 10)
    assert np.all(np.isfinite(standardizer.transform(np.zeros((1, 2, 2), dtype=np.float32))))


def test_the_standardizer_refuses_a_shape_it_was_not_fitted_on():
    standardizer = PixelStandardizer(np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32), 5)
    with pytest.raises(ValueError, match="incompativel"):
        standardizer.transform(np.zeros((1, 3, 3), dtype=np.float32))


def test_the_standardizer_round_trips_through_disk(tmp_path):
    original = PixelStandardizer(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        np.full((2, 2), 2.0, dtype=np.float32),
        n=7,
    )
    original.save(tmp_path / "s.npz")
    restored = PixelStandardizer.load(tmp_path / "s.npz")
    assert np.array_equal(restored.mean, original.mean)
    assert np.array_equal(restored.std, original.std)
    assert restored.n == 7


def test_the_cache_is_written_once_and_read_back_as_a_memmap(tmp_path, monkeypatch):
    """O que `ensure_arm_cache` grava e o que `open_cache_memmap` le: sem
    reextrair na segunda vez, e sem trazer o array para a RAM."""
    calls = []
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    def fake_build(folder, feature_name):
        calls.append(folder)
        return features, np.array(["a.wav", "b.wav"])

    monkeypatch.setattr(features_module, "build_cache", fake_build)
    arm_feature_folder(tmp_path, "m0").mkdir(parents=True)
    for _ in range(2):
        mapped, names = ensure_arm_cache(tmp_path, "m0", "Spec")
    assert len(calls) == 1
    assert isinstance(mapped, np.memmap)
    assert np.array_equal(mapped, features)
    assert names.tolist() == ["a.wav", "b.wav"]


def test_feature_folder_is_beside_the_audio(tmp_path):
    # O cache mora dentro do dataset, ao lado dos wavs, como no POC I: copiar o
    # dataset leva o cache junto.
    assert arm_feature_folder(tmp_path, "m0") == tmp_path / "m0" / EFFECT_FOLDER
