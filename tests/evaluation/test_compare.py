"""Comparacao entre features: `<base>/<FEATURE>/chain_metrics.csv`."""
from __future__ import annotations

import pandas as pd
import pytest

from gefx.evaluation.compare import build_feature_comparison, compare_features


@pytest.fixture
def results_base(tmp_path):
    def build(feature_rows):
        base = tmp_path / "feature_runs"
        for feature, rows in feature_rows.items():
            folder = base / feature
            folder.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(folder / "chain_metrics.csv", index=False)
        return base

    return build


def test_one_row_per_feature_sorted_by_mean_mae(results_base):
    base = results_base(
        {
            "MFCC40": [{"chain_key": "a", "mae": 0.4}, {"chain_key": "b", "mae": 0.6}],
            "Spec": [{"chain_key": "a", "mae": 0.1}, {"chain_key": "b", "mae": 0.3}],
        }
    )
    frame = build_feature_comparison(base)

    assert frame["feature"].tolist() == ["Spec", "MFCC40"]
    assert frame.loc[0, "mean_mae"] == pytest.approx(0.2)
    assert frame.loc[0, "min_mae"] == pytest.approx(0.1)
    assert frame.loc[0, "n_chains"] == 2
    assert frame.loc[0, "best_chain_key"] == "a"


def test_best_chain_key_is_the_argmin_not_the_first_row(results_base):
    base = results_base({"Spec": [{"chain_key": "ruim", "mae": 0.9}, {"chain_key": "boa", "mae": 0.1}]})
    assert build_feature_comparison(base).loc[0, "best_chain_key"] == "boa"


def test_ignores_directories_without_chain_metrics(results_base, tmp_path):
    base = results_base({"Spec": [{"chain_key": "a", "mae": 0.1}]})
    (base / "estimated_vs_real").mkdir()

    assert build_feature_comparison(base)["feature"].tolist() == ["Spec"]


def test_requires_at_least_one_feature_run(tmp_path):
    with pytest.raises(RuntimeError, match="No feature runs found under"):
        build_feature_comparison(tmp_path)


def test_rejects_an_empty_metrics_file(tmp_path):
    folder = tmp_path / "Spec"
    folder.mkdir()
    pd.DataFrame(columns=["chain_key", "mae"]).to_csv(folder / "chain_metrics.csv", index=False)

    with pytest.raises(ValueError, match="Empty metrics file"):
        build_feature_comparison(tmp_path)


@pytest.mark.parametrize("missing", ["mae", "chain_key"])
def test_rejects_metrics_without_the_required_columns(tmp_path, missing):
    folder = tmp_path / "Spec"
    folder.mkdir()
    rows = pd.DataFrame([{"chain_key": "a", "mae": 0.1}]).drop(columns=[missing])
    rows.to_csv(folder / "chain_metrics.csv", index=False)

    with pytest.raises(ValueError, match=f"Missing '{missing}' column"):
        build_feature_comparison(tmp_path)


def test_compare_features_default_output_paths(results_base):
    base = results_base({"Spec": [{"chain_key": "a", "mae": 0.1}]})
    csv_path, plot_path = compare_features(base)

    assert csv_path == base.resolve() / "feature_comparison.csv"
    assert plot_path == base.resolve() / "feature_comparison_mae.png"
    assert csv_path.exists() and plot_path.exists()


def test_compare_features_honours_explicit_output_paths(results_base, tmp_path):
    base = results_base({"Spec": [{"chain_key": "a", "mae": 0.1}]})
    csv_path, plot_path = compare_features(base, tmp_path / "t.csv", tmp_path / "g.png")

    assert csv_path.exists() and plot_path.exists()
    assert not (base / "feature_comparison.csv").exists()
