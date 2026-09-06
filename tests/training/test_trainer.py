"""Treino: a montagem do `predictions.csv`, o split e a orquestracao.

Nada aqui toca TensorFlow — `train_one_chain` e substituido por um duble, o que
deixa visivel exatamente o que `train` decide e o que ele grava.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gefx.config import load_config
from gefx.training import trainer as trainer_module
from gefx.training.trainer import _predictions_frame, train


def test_predictions_frame_interleaves_true_and_predicted_columns():
    frame = _predictions_frame(
        np.array(["a.wav", "b.wav"]),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.15, 0.25], [0.35, 0.45]]),
        ["chorus_rate_hz", "chorus_depth"],
    )

    assert list(frame.columns) == [
        "file_name",
        "y_true_chorus_rate_hz",
        "y_pred_chorus_rate_hz",
        "y_true_chorus_depth",
        "y_pred_chorus_depth",
    ]
    assert frame["file_name"].tolist() == ["a.wav", "b.wav"]
    assert frame["y_pred_chorus_depth"].tolist() == [0.25, 0.45]


def test_predictions_frame_stringifies_numpy_file_names():
    frame = _predictions_frame(
        np.array(["a.wav"], dtype=object), np.zeros((1, 1)), np.zeros((1, 1)), ["p"]
    )
    assert isinstance(frame["file_name"][0], str)


@pytest.fixture
def stub_training(monkeypatch, tmp_path):
    """Substitui tudo o que custa caro; devolve o registro do que foi chamado."""
    calls = {"chains": [], "clear_session": 0, "seeds": []}

    def fake_train_one_chain(chain_key_value, config):
        calls["chains"].append(chain_key_value)
        return {
            "chain_key": chain_key_value,
            "feature": config.train.feature,
            "mae": 0.1,
            "mse": 0.01,
            "n_train": 8,
        }

    monkeypatch.setattr(trainer_module, "train_one_chain", fake_train_one_chain)
    monkeypatch.setattr(
        trainer_module, "validate_sidecar_integrity", lambda root: {"distortion": 3}
    )
    monkeypatch.setattr(
        trainer_module,
        "set_global_seeds",
        lambda seed, deterministic=False: calls["seeds"].append((seed, deterministic)),
    )
    monkeypatch.setattr(
        trainer_module, "list_chain_keys", lambda root: ["chorus", "distortion"]
    )
    monkeypatch.setattr(
        trainer_module,
        "clear_session",
        lambda: calls.__setitem__("clear_session", calls["clear_session"] + 1),
    )
    return calls


def make_config(tmp_path, **overrides):
    return load_config(dataset_root=str(tmp_path / "ds"), results_root=str(tmp_path / "res"), **overrides)


def test_train_runs_every_chain_in_the_dataset(tmp_path, stub_training):
    metrics = train(make_config(tmp_path))

    assert stub_training["chains"] == ["chorus", "distortion"]
    assert [item["chain_key"] for item in metrics] == ["chorus", "distortion"]


def test_train_honours_chain_key(tmp_path, stub_training):
    train(make_config(tmp_path, chain_key="distortion"))
    assert stub_training["chains"] == ["distortion"]


def test_train_seeds_once_before_any_chain(tmp_path, stub_training):
    train(make_config(tmp_path, seed=7, deterministic=True))
    assert stub_training["seeds"] == [(7, True)]


def test_train_clears_the_session_between_chains(tmp_path, stub_training):
    train(make_config(tmp_path))
    assert stub_training["clear_session"] == 2


def test_no_clear_session_leaves_the_session_alone(tmp_path, stub_training):
    train(make_config(tmp_path, clear_session=False))
    assert stub_training["clear_session"] == 0


def test_train_rejects_an_empty_chain_list(tmp_path, stub_training, monkeypatch):
    monkeypatch.setattr(trainer_module, "list_chain_keys", lambda root: [])
    with pytest.raises(RuntimeError, match="No matching chains found"):
        train(make_config(tmp_path))


def test_train_writes_all_metrics_and_the_manifest(tmp_path, stub_training):
    config = make_config(tmp_path, feature="Spec")
    train(config)

    results_root = tmp_path / "res"
    all_metrics = json.loads((results_root / "all_metrics.json").read_text(encoding="utf-8"))
    assert [item["chain_key"] for item in all_metrics] == ["chorus", "distortion"]

    manifest = json.loads((results_root / "run.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"timestamp", "git_revision", "config", "metrics"}
    assert manifest["config"]["train"]["feature"] == "Spec"
    # O resumo do manifesto e estreitado a quatro chaves.
    assert all(set(item) == {"chain_key", "feature", "mae", "mse"} for item in manifest["metrics"])
