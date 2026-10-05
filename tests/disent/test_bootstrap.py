"""A diferenca pareada por gravacao entre execucoes."""
from __future__ import annotations

import pandas as pd
import pytest

from gefx.disent.bootstrap import compare_runs, recording_accuracy


def _run(path, hits):
    """Uma execucao com duas consultas por gravacao; `hits[r]` acertos na gravacao r."""
    rows = []
    for recording, n_hits in enumerate(hits):
        for segment in range(2):
            rows.append({"query_content": f"r{recording}_s{segment}", "true_drive_level": 3,
                         "pred_drive_level": 3 if segment < n_hits else 0})
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "predictions_validacao.csv", index=False)
    return path


def test_the_segments_of_a_recording_are_averaged_together(tmp_path):
    accuracy = recording_accuracy(_run(tmp_path / "a", [2, 1, 0]))
    assert accuracy.to_dict() == {"r0": 1.0, "r1": 0.5, "r2": 0.0}


def test_the_difference_is_in_points_and_the_interval_contains_it(tmp_path):
    a = _run(tmp_path / "a", [2, 2, 1, 2, 1, 2])
    b = _run(tmp_path / "b", [1, 2, 0, 1, 1, 1])
    row = compare_runs([a], [b]).iloc[0]
    assert row["a_menos_b"] == pytest.approx(100 * (10 - 6) / 12, abs=0.01)
    assert row["ic95_inf"] <= row["a_menos_b"] <= row["ic95_sup"]


def test_several_seeds_add_a_joint_row(tmp_path):
    a = [_run(tmp_path / f"a{i}", [2, 1, 2, 0]) for i in range(2)]
    b = [_run(tmp_path / f"b{i}", [1, 1, 0, 0]) for i in range(2)]
    assert list(compare_runs(a, b)["par"]) == ["a0 - b0", "a1 - b1", "juntas"]


def test_runs_on_different_recordings_are_refused(tmp_path):
    with pytest.raises(ValueError, match="mesmas gravacoes"):
        compare_runs([_run(tmp_path / "a", [1, 1])], [_run(tmp_path / "b", [1, 1, 1])])
