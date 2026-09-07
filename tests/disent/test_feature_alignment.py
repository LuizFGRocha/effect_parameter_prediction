"""Alinhamento entre o cache de features e as linhas do sidecar.

E a armadilha herdada do POC I: `file_names.json` guarda a ordem alfabetica dos
wavs, que nao e a ordem das linhas do sidecar. Usar as duas trocadas nao levanta
erro -- so treina com os rotulos errados.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gefx.data.cache import save_cache
from gefx.disent.features import align_to_frame, arm_feature_folder, load_features
from gefx.disent.sidecar import EFFECT_FOLDER

NAMES = ["a.wav", "b.wav", "c.wav"]


def _frame(names, arm="m0"):
    return pd.DataFrame({"file_name": names, "arm": [arm] * len(names)})


def test_alignment_follows_the_frame_and_not_the_cache_order():
    features = np.array([[10.0], [20.0], [30.0]])
    # O frame pede outra ordem que a do cache; e esse o caso real.
    aligned = align_to_frame(features, NAMES, _frame(["c.wav", "a.wav", "b.wav"]))
    assert aligned.tolist() == [[30.0], [10.0], [20.0]]


def test_alignment_handles_a_frame_that_repeats_a_row():
    features = np.array([[10.0], [20.0], [30.0]])
    aligned = align_to_frame(features, NAMES, _frame(["b.wav", "b.wav"]))
    assert aligned.tolist() == [[20.0], [20.0]]


def test_alignment_refuses_a_row_without_its_feature():
    features = np.array([[10.0], [20.0], [30.0]])
    with pytest.raises(ValueError, match="sem feature em cache"):
        align_to_frame(features, NAMES, _frame(["a.wav", "z.wav"]))


def test_a_cache_larger_than_the_frame_is_fine():
    # O recorte por split pede menos linhas que o cache do arm inteiro.
    features = np.array([[10.0], [20.0], [30.0]])
    assert align_to_frame(features, NAMES, _frame(["b.wav"])).tolist() == [[20.0]]


def test_load_features_keeps_row_order_across_arms(tmp_path):
    # O `GridIndex` e montado sobre o mesmo frame, entao os indices das tuplas
    # indexam este array diretamente -- a ordem precisa bater linha a linha.
    root = tmp_path / "ds"
    for arm, base in (("m0", 0.0), ("m1", 100.0)):
        folder = arm_feature_folder(root, arm)
        folder.mkdir(parents=True)
        save_cache(
            folder, "Spec",
            np.array([[base + index] for index in range(len(NAMES))], dtype=np.float32),
            np.array(NAMES),
        )

    frame = pd.DataFrame(
        {
            "file_name": ["c.wav", "a.wav", "b.wav", "a.wav"],
            "arm": ["m1", "m0", "m1", "m1"],
        }
    )
    features = load_features(root, frame, "Spec")
    assert features.shape == (4, 1)
    assert features[:, 0].tolist() == [102.0, 0.0, 101.0, 100.0]


def test_load_features_refuses_an_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="frame vazio"):
        load_features(tmp_path, pd.DataFrame({"file_name": [], "arm": []}), "Spec")


def test_feature_folder_is_beside_the_audio(tmp_path):
    # O cache mora dentro do dataset, ao lado dos wavs, como no POC I: copiar o
    # dataset leva o cache junto.
    assert arm_feature_folder(tmp_path, "m0") == tmp_path / "m0" / EFFECT_FOLDER
