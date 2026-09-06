"""Carga do modelo e predicao: o caminho unico compartilhado por treino e crossimpl."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from gefx.training.inference import MODEL_FILENAME, TrainedChain, load_trained_chain
from gefx.training.scaling import fit_scalers


class StubModel:
    def __init__(self, output_dim=2):
        self.output_dim = output_dim
        self.seen = None
        self.verbose = None

    def predict(self, features, verbose=None):
        self.seen = features
        self.verbose = verbose
        return np.zeros((len(features), self.output_dim))


def test_predict_scales_before_calling_the_model():
    raw = np.random.default_rng(0).normal(loc=5.0, size=(6, 3, 4))
    scalers = fit_scalers(raw)
    model = StubModel()

    chain = TrainedChain(model=model, scalers=scalers, directory="qualquer")
    output = chain.predict(raw)

    # O modelo recebe o array ja escalado e com o eixo de canal.
    assert model.seen.shape == (6, 3, 4, 1)
    assert np.allclose(model.seen[..., 0].mean(axis=0), 0.0, atol=1e-6)
    assert model.verbose == 0
    assert output.shape == (6, 2)


def test_predict_does_not_mutate_the_caller_array():
    raw = np.random.default_rng(0).normal(size=(6, 3, 4))
    original = raw.copy()
    TrainedChain(model=StubModel(), scalers=fit_scalers(raw), directory="x").predict(raw)
    assert np.array_equal(raw, original)


def test_load_trained_chain_reports_a_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Modelo nao encontrado"):
        load_trained_chain(tmp_path)


def test_load_trained_chain_reports_missing_scalers(tmp_path):
    (tmp_path / MODEL_FILENAME).touch()
    with pytest.raises(FileNotFoundError, match="Scalers nao encontrados"):
        load_trained_chain(tmp_path)


def test_a_missing_model_does_not_pay_for_tensorflow(tmp_path):
    # O `import keras` fica depois das checagens: um caminho errado responde na
    # hora em vez de gastar ~1,4 s carregando o TensorFlow para so entao falhar.
    script = f"""
import sys
from gefx.training.inference import load_trained_chain
try:
    load_trained_chain({str(tmp_path)!r})
except FileNotFoundError:
    pass
print("tensorflow" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
