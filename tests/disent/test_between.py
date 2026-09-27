"""Configuracoes entre os niveis: os knobs dos pontos medios e as medidas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefx.disent.arms import load_levels, write_levels
from gefx.disent.between import between_predictions, summarize
from gefx.disent.calibrate import between_levels


def test_between_levels_fall_between_the_grid_knobs(tmp_path):
    knobs = np.linspace(0.0, 40.0, 41)
    curve = 1.0 - 0.004 * knobs  # Rnonlin cai reto com o knob
    pd.DataFrame({"arm": "a", "knob": knobs, "rnonlin": curve}).to_csv(
        tmp_path / "curvas.csv", index=False)
    write_levels(tmp_path / "niveis.yaml", {
        "descriptor": "rnonlin", "targets": [0.96, 0.92, 0.88],
        "arms": {"a": {"levels": [10.0, 20.0, 30.0], "unmatched": []}}})
    levels = between_levels(tmp_path / "niveis.yaml", tmp_path / "curvas.csv",
                            tmp_path / "entre.yaml")
    assert levels["a"] == pytest.approx([15.0, 25.0])
    assert load_levels(tmp_path / "entre.yaml")["targets"] == pytest.approx([0.94, 0.90])


def _frames():
    # Catalogo: nivel 0 em [1, 0], nivel 1 em [0, 1], em dois arms.
    catalog = pd.DataFrame({
        "arm": ["a", "a", "b", "b"], "drive_level": [0, 1, 0, 1],
        "drive_db_equivalente": [10.0, 20.0, 10.0, 20.0]})
    queries = pd.DataFrame({
        "file_name": ["q0", "q1"], "arm": ["a", "b"], "source_audio_id": ["r", "r"],
        "drive_level": [0, 0], "drive_db_equivalente": [15.0, 15.0]})
    z_catalog = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    return queries, catalog, z_catalog


def test_the_search_skips_the_query_arm_and_averages_the_k_nearest():
    queries, catalog, z_catalog = _frames()
    z_query = np.array([[1.0, 0.2], [0.2, 1.0]])
    out = between_predictions(queries, catalog, z_query, z_catalog, k=2)
    assert list(out["pred_drive_level"]) == [0, 1]
    assert list(out["knn_drive_db"]) == pytest.approx([15.0, 15.0])
    numbers = summarize(out)
    assert numbers["neighbor"] == 1.0
    assert numbers["err_nn_db"] == pytest.approx(5.0)
    assert numbers["err_knn_db"] == pytest.approx(0.0)
