"""Metricas recomputadas a partir dos arquivos que o treino deixou em disco."""
from __future__ import annotations

import pytest

from gefx.effects.catalog import parameter_names_for_chain
from gefx.evaluation.metrics import (
    build_prediction_frame,
    evaluate_chain,
    find_chain_dirs,
    load_metrics,
    load_predictions,
    parameter_columns,
    standard_error,
)

# chorus: erros de 0.1/0.3/0.5 em rate_hz e 0.2/0.4/0.6 em depth.
Y_TRUE = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
Y_PRED = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


@pytest.mark.parametrize("values", [[], [1.0]])
def test_standard_error_is_zero_for_degenerate_samples(values):
    assert standard_error(values) == 0.0


def test_standard_error_uses_the_sample_std():
    # ddof=1: desvio amostral, nao populacional. Com [1,2,3,4] da
    # sqrt(5/3)/sqrt(4); com ddof=0 daria 0.559.
    assert standard_error([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.6454972, abs=1e-6)


def test_parameter_columns_follows_the_target_vector_order(fake_chain_results):
    # A ordem das colunas gravadas e a do vetor alvo; parameter_metrics.csv,
    # metrics.json e predictions.csv passam a concordar entre si.
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    columns = parameter_columns(load_predictions(chain_dir))

    assert columns == parameter_names_for_chain("chorus") == ["chorus_rate_hz", "chorus_depth"]


def test_parameter_columns_legacy_sorts_alphabetically(fake_chain_results):
    # Reproduz as tabelas ja geradas.
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    columns = parameter_columns(load_predictions(chain_dir), legacy=True)

    assert columns == ["chorus_depth", "chorus_rate_hz"]
    assert sorted(columns) == sorted(parameter_names_for_chain("chorus"))


def test_load_predictions_and_metrics_report_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing predictions file"):
        load_predictions(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing metrics file"):
        load_metrics(tmp_path)


def test_find_chain_dirs_requires_both_artifacts(fake_chain_results, tmp_path):
    root = tmp_path / "res"
    fake_chain_results("chorus", Y_TRUE, Y_PRED, root=root)
    fake_chain_results("distortion", [[0.0]], [[0.1]], root=root)

    (root / "so_metrics").mkdir()
    (root / "so_metrics" / "metrics.json").write_text("{}", encoding="utf-8")
    (root / "estimated_vs_real").mkdir()  # diretorio de graficos, deve ser ignorado

    assert [path.name for path in find_chain_dirs(root)] == ["chorus", "distortion"]


def test_evaluate_chain_recomputes_per_parameter_errors(fake_chain_results):
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    chain_row, param_rows = evaluate_chain(chain_dir)

    assert param_rows["parameter"].tolist() == ["chorus_rate_hz", "chorus_depth"]
    assert param_rows.loc[0, "mae"] == pytest.approx(0.3)
    assert param_rows.loc[1, "mae"] == pytest.approx(0.4)
    assert param_rows.loc[1, "mse"] == pytest.approx((0.04 + 0.16 + 0.36) / 3)
    assert param_rows["mae_sem"].gt(0).all()

    # O agregado vem do metrics.json, nao e recomputado.
    assert param_rows["global_mae"].unique().tolist() == [pytest.approx(0.35)]
    assert chain_row.loc[0, "mae"] == pytest.approx(0.35)


def test_evaluate_chain_row_carries_the_run_metadata(fake_chain_results):
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    chain_row, _ = evaluate_chain(chain_dir)

    assert chain_row.loc[0, "chain_key"] == "chorus"
    assert chain_row.loc[0, "chain_length"] == 1
    assert chain_row.loc[0, "feature"] == "Spec"
    assert chain_row.loc[0, "n_test"] == 3
    assert chain_row.loc[0, "output_dim"] == 2


def test_chain_sem_is_computed_over_all_parameters_pooled(fake_chain_results):
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    chain_row, _ = evaluate_chain(chain_dir)

    pooled = [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]  # rate_hz primeiro: ordem do vetor alvo
    assert chain_row.loc[0, "mae_sem"] == pytest.approx(standard_error(pooled))


def test_evaluate_chain_legacy_keeps_the_old_row_order(fake_chain_results):
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED)
    _, param_rows = evaluate_chain(chain_dir, legacy=True)

    assert param_rows["parameter"].tolist() == ["chorus_depth", "chorus_rate_hz"]
    assert param_rows.loc[0, "mae"] == pytest.approx(0.4)


def test_evaluate_chain_falls_back_to_the_directory_name(fake_chain_results):
    chain_dir = fake_chain_results("chorus", Y_TRUE, Y_PRED, chain_key=None)
    import json

    metrics = json.loads((chain_dir / "metrics.json").read_text(encoding="utf-8"))
    del metrics["chain_key"]
    (chain_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    chain_row, _ = evaluate_chain(chain_dir)
    assert chain_row.loc[0, "chain_key"] == "chorus"


def test_build_prediction_frame_is_long_format(fake_chain_results, tmp_path):
    root = tmp_path / "res"
    fake_chain_results("chorus", Y_TRUE, Y_PRED, root=root)
    fake_chain_results("distortion", [[0.0], [0.0]], [[0.1], [0.2]], root=root)

    frame = build_prediction_frame(find_chain_dirs(root))

    assert list(frame.columns) == ["chain_key", "parameter", "real", "estimated"]
    assert len(frame) == 3 * 2 + 2 * 1
    assert set(frame["chain_key"]) == {"chorus", "distortion"}
    assert set(frame["parameter"]) == {"chorus_depth", "chorus_rate_hz", "distortion_drive_db"}


def test_build_prediction_frame_rejects_an_empty_run():
    with pytest.raises(ValueError, match="Nenhuma predicao encontrada"):
        build_prediction_frame([])
