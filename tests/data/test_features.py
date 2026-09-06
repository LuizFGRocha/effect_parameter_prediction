"""Extracao de features.

`stack_features` e puro e barato; `extract_feature` carrega librosa/spafe e le um
wav de verdade, entao fica marcado `slow`.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.data.features import FEATURE_NAMES, GFCC_SAMPLE_RATE, extract_feature, stack_features


@pytest.mark.parametrize("feature_name", ["Spec", "MFCC40", "Chroma"])
def test_stack_features_keeps_the_axes_for_non_gfcc(feature_name):
    features = [np.zeros((4, 7)), np.ones((4, 7))]
    stacked = stack_features(features, feature_name)

    assert stacked.shape == (2, 4, 7)
    assert stacked.dtype == np.float32


def test_stack_features_swaps_axes_for_gfcc():
    # O spafe entrega (frames, coeficientes); as demais entregam (linhas, frames).
    # A troca de eixos e o que coloca as quatro features no mesmo formato.
    features = [np.zeros((7, 40)), np.ones((7, 40))]
    stacked = stack_features(features, "GFCC40")

    assert stacked.shape == (2, 40, 7)
    assert stacked.dtype == np.float32


def test_stack_features_preserves_values_through_the_swap():
    single = np.arange(6, dtype=np.float64).reshape(3, 2)
    stacked = stack_features([single], "GFCC40")
    assert np.array_equal(stacked[0], single.T)


def test_unknown_feature_name_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported feature_name=NaoExiste"):
        extract_feature(tmp_path / "x.wav", "NaoExiste")


def test_feature_names_is_the_documented_set():
    assert FEATURE_NAMES == ("Spec", "MFCC40", "Chroma", "GFCC40")
    assert GFCC_SAMPLE_RATE == 16000
