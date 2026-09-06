"""Avaliacao de um run: quais arquivos `gefx evaluate` deixa em disco e com que
conteudo. `evaluate_run` roda uma vez so, no escopo do modulo."""
from __future__ import annotations

import pandas as pd
import pytest

from gefx.evaluation.runner import evaluate_run

Y_TRUE = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
Y_PRED = [[0.15, 0.1], [0.2, 0.5], [0.6, 0.55]]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, chain_results_builder):
    root = tmp_path_factory.mktemp("res")
    chain_results_builder("chorus", Y_TRUE, Y_PRED, root)
    chain_results_builder("distortion", [[0.1], [0.4], [0.9]], [[0.2], [0.3], [0.8]], root)
    return evaluate_run(root)


def test_produces_the_five_named_outputs(outputs):
    assert set(outputs) == {
        "chain_metrics",
        "parameter_metrics",
        "chain_plot",
        "parity_plots",
        "best_worst_plot",
    }
    assert outputs["chain_metrics"].name == "chain_metrics.csv"
    assert outputs["parameter_metrics"].name == "parameter_metrics.csv"
    assert outputs["chain_plot"].name == "chain_baseline_mae_mse.png"
    assert outputs["parity_plots"].name == "estimated_vs_real"
    assert outputs["best_worst_plot"].name == "estimated_vs_real_best_worst.png"
    for path in outputs.values():
        assert path.exists()


def test_chain_metrics_csv_has_one_row_per_chain(outputs):
    frame = pd.read_csv(outputs["chain_metrics"])
    assert frame["chain_key"].tolist() == ["chorus", "distortion"]
    assert {"mae", "mse", "mae_sem", "mse_sem", "feature"} <= set(frame.columns)


def test_parameter_metrics_csv_follows_the_target_vector_order(outputs):
    frame = pd.read_csv(outputs["parameter_metrics"])
    assert frame["parameter"].tolist() == [
        "chorus_rate_hz",
        "chorus_depth",
        "distortion_drive_db",
    ]


def test_legacy_reproduces_the_alphabetical_tables(tmp_path, chain_results_builder):
    # E o que o `--legacy` compra: reproduzir as tabelas ja commitadas.
    chain_results_builder("chorus", Y_TRUE, Y_PRED, tmp_path)
    legacy = evaluate_run(tmp_path, legacy=True)

    frame = pd.read_csv(legacy["parameter_metrics"])
    assert frame["parameter"].tolist() == ["chorus_depth", "chorus_rate_hz"]


def test_parity_plots_are_one_file_per_parameter(outputs):
    names = sorted(path.name for path in outputs["parity_plots"].iterdir())
    assert names == ["chorus_depth.png", "chorus_rate_hz.png", "distortion_drive_db.png"]


def test_requires_chain_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="No chain outputs found under"):
        evaluate_run(tmp_path)
