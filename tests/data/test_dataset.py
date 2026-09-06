"""Leitura de um dataset.

O contrato central: a ordem das linhas de `X` e `y` vem do `file_names.json` do
cache, e nao da ordem das linhas do `metadata.csv`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gefx.data import cache as cache_module
from gefx.data.cache import file_names_cache_file
from gefx.data.dataset import load_chain_dataset
from gefx.data.metadata import METADATA_FILENAME


@pytest.fixture(autouse=True)
def stub_extract(monkeypatch):
    monkeypatch.setattr(
        cache_module,
        "extract_feature",
        lambda path, feature_name: np.full((2, 3), float(ord(path.stem[-1])), dtype=np.float64),
    )


def test_load_chain_dataset_returns_features_targets_and_names(fake_dataset):
    root = fake_dataset(chains=[["chorus"]], samples=3)
    features, targets, file_names = load_chain_dataset(root, "chorus", feature_name="Spec")

    assert features.shape == (3, 2, 3)
    assert targets.shape == (3, 2)  # chorus preve 2 parametros
    assert targets.dtype == np.float32
    assert len(file_names) == 3


def test_targets_follow_file_names_json_not_csv_row_order(fake_dataset):
    # Este e o teste que protege o alinhamento entre X e y. Se `y` passar a ser
    # montado pela ordem do CSV, ele fica vermelho.
    root = fake_dataset(chains=[["distortion"]], samples=3)

    path = root / METADATA_FILENAME
    shuffled = pd.read_csv(path).iloc[::-1].reset_index(drop=True)
    shuffled.to_csv(path, index=False)

    _, targets, file_names = load_chain_dataset(root, "distortion", feature_name="Spec")

    lookup = dict(
        zip(shuffled["file_name"], shuffled["normalized_parameter_vector"].map(json.loads))
    )
    expected = [lookup[name] for name in file_names]
    assert np.allclose(targets, expected)
    assert list(file_names) == sorted(file_names)  # a ordem do cache e a ordenada


def test_rejects_unknown_feature_name(fake_dataset):
    root = fake_dataset(chains=[["distortion"]])
    with pytest.raises(ValueError, match="Unsupported feature_name=NaoExiste"):
        load_chain_dataset(root, "distortion", feature_name="NaoExiste")


def test_raises_when_a_cached_wav_has_no_metadata_row(fake_dataset):
    root = fake_dataset(chains=[["distortion"]], samples=3)

    path = root / METADATA_FILENAME
    frame = pd.read_csv(path)
    orphan = frame.loc[0, "file_name"]
    frame.iloc[1:].to_csv(path, index=False)

    with pytest.raises(RuntimeError, match=f"Example missing file: {orphan}"):
        load_chain_dataset(root, "distortion", feature_name="Spec")


def test_raises_when_the_cache_is_stale(fake_dataset):
    # npz com 3 linhas e file_names.json com 2: e o estado que sobra quando a
    # extracao muda e o cache nao e reconstruido.
    root = fake_dataset(chains=[["distortion"]], samples=3)
    load_chain_dataset(root, "distortion", feature_name="Spec")

    names_path = file_names_cache_file(root / "distortion")
    names = json.loads(names_path.read_text(encoding="utf-8"))
    names_path.write_text(json.dumps(names[:2]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Feature/target size mismatch: X=3 y=2"):
        load_chain_dataset(root, "distortion", feature_name="Spec")


def test_missing_chain_folder(fake_dataset):
    root = fake_dataset(chains=[["distortion"]])
    with pytest.raises(FileNotFoundError, match="Missing chain folder"):
        load_chain_dataset(root, "chorus", feature_name="Spec")
