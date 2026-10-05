"""Invariantes da recuperacao em catalogo e dos baselines sem aprendizado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefx.disent.retrieval import (
    LEVEL_AXES,
    alphabet,
    nearest,
    retrieve,
    retrieve_by_arm,
    score,
)


# --- vizinho mais proximo -----------------------------------------------------
def test_nearest_finds_the_exact_row():
    rng = np.random.default_rng(0)
    catalog = rng.standard_normal((20, 8)).astype(np.float32)
    picks, dists = nearest(catalog[[3, 7, 11]], catalog)
    assert picks[:, 0].tolist() == [3, 7, 11]
    assert dists[:, 0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_the_k_nearest_come_closest_first():
    catalog = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    picks, dists = nearest(np.array([[1.0, 0.1]], dtype=np.float32), catalog, k=3)
    assert picks.tolist() == [[0, 2, 1]]
    assert np.all(np.diff(dists[0]) > 0)


def test_nearest_is_cosine_and_ignores_the_norm():
    """O codigo aprendido vive na esfera, onde o contrastivo otimiza; o B0 e
    buscado do mesmo jeito para a comparacao ser so aprender ou nao."""
    queries = np.array([[2.0, 0.0]], dtype=np.float32)
    catalog = np.array([[1.0, 0.0], [2.0, 0.6]], dtype=np.float32)
    picks, dists = nearest(queries, catalog)
    assert picks[0, 0] == 0
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_nearest_is_euclidean_for_a_scalar_code():
    """Numa dimensao o cosseno so ve o sinal: 0.2 e 0.9 seriam o mesmo ponto."""
    catalog = np.array([[0.2], [0.9]], dtype=np.float32)
    picks, dists = nearest(np.array([[0.85]], dtype=np.float32), catalog)
    assert picks[0, 0] == 1
    assert dists[0, 0] == pytest.approx(0.05, abs=1e-6)


def test_nearest_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        nearest(np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32))


# --- a tarefa -----------------------------------------------------------------
def _frame(arm, contents, split, levels=(0, 1, 2, 3)):
    rows = []
    for content in contents:
        for drive in levels:
            rows.append(
                {
                    "file_name": f"{content}__d{drive}.wav",
                    "arm": arm,
                    "stratum": "S1",
                    "content_id": content,
                    "split": split,
                    "config_index": drive,
                    "drive_level": drive,
                    "drive_db_equivalente": 5.0 + 5.0 * drive,
                }
            )
    return pd.DataFrame(rows)


def _onehot(frame):
    """Codigo que codifica o nivel: a recuperacao tem de ser perfeita."""
    return np.eye(8, dtype=np.float32)[frame["drive_level"].to_numpy()]


def test_retrieve_refuses_shared_content_between_query_and_catalog():
    # E a guarda que sustenta a leitura do resultado inteiro: com conteudo em
    # comum, acertar por reconhecer a execucao passa por acertar o ajuste.
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c1"], "catalog")
    with pytest.raises(ValueError, match="compartilham conteudo"):
        retrieve(q, c, np.zeros((len(q), 3), np.float32), np.zeros((len(c), 3), np.float32))


def test_retrieve_refuses_two_segments_of_the_same_recording():
    q = _frame("a", ["r1_s0"], "query").assign(source_audio_id="r1.wav")
    c = _frame("b", ["r1_s1"], "catalog").assign(source_audio_id="r1.wav")
    with pytest.raises(ValueError, match="compartilham conteudo"):
        retrieve(q, c, np.zeros((len(q), 3), np.float32), np.zeros((len(c), 3), np.float32))


def test_retrieve_reports_what_was_asked_and_what_came_back():
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c2"], "catalog")
    out = retrieve(q, c, _onehot(q), _onehot(c))
    assert list(out["true_drive_level"]) == list(out["pred_drive_level"])
    assert set(out["retrieved_arm"]) == {"b"}
    assert out["distance"].to_numpy() == pytest.approx(0.0, abs=1e-6)


def test_score_reports_chance_next_to_every_accuracy():
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c2"], "catalog")
    out = retrieve(q, c, _onehot(q), _onehot(c))
    metrics = score(out, alphabet(q))
    assert metrics["drive_level"]["exact"] == 1.0
    assert metrics["drive_level"]["chance"] == pytest.approx(0.25)
    assert metrics["drive_level"]["mae_levels"] == 0.0
    assert metrics["mae_db"] == 0.0
    assert metrics["cross_implementation"] is True


def test_score_flags_a_catalog_that_leaked_the_query_arm():
    q = _frame("a", ["c1"], "query")
    c = _frame("a", ["c2"], "catalog")
    out = retrieve(q, c, _onehot(q), _onehot(c))
    assert score(out, alphabet(q))["cross_implementation"] is False


def test_score_counts_a_one_level_miss_as_within_one_but_not_exact():
    q = _frame("a", ["c1"], "query", levels=(0, 1))
    c = _frame("b", ["c2"], "catalog", levels=(0, 1))
    # Rotulos do catalogo deslocados de um nivel em relacao ao descritor: cada
    # consulta encontra o vizinho certo e recebe o rotulo errado por um.
    c["drive_level"] = [1, 2]
    c["drive_db_equivalente"] = [10.0, 15.0]
    codigos = np.eye(2, dtype=np.float32)
    out = retrieve(q, c, codigos, codigos)
    metrics = score(out, alphabet(q))
    assert metrics["drive_level"]["exact"] == 0.0
    assert metrics["drive_level"]["within_one"] == 1.0
    assert metrics["drive_level"]["mae_levels"] == 1.0
    assert metrics["mae_db"] == pytest.approx(5.0)


def test_alphabet_reads_the_slice_and_not_the_full_grid():
    # Um recorte com menos niveis tem outro acaso; reportar o da grade cheia
    # inflaria o resultado.
    frame = _frame("a", ["c1"], "query", levels=(0, 1))
    assert alphabet(frame) == {"drive_level": 2}
    assert set(alphabet(frame)) == set(LEVEL_AXES)


def _multi_arm(arms, contents, split):
    return pd.concat([_frame(arm, contents, split) for arm in arms], ignore_index=True)


def test_retrieve_by_arm_never_lets_a_query_reach_its_own_arm():
    arms = ["m0", "m1", "m2"]
    queries = _multi_arm(arms, ["q0", "q1"], "query")
    catalog = _multi_arm(arms, ["k0", "k1"], "catalog")
    rng = np.random.default_rng(0)
    result = retrieve_by_arm(
        queries, catalog,
        rng.random((len(queries), 3)).astype(np.float32),
        rng.random((len(catalog), 3)).astype(np.float32),
        same_arm=False,
    )
    assert not (result.predictions["query_arm"] == result.predictions["retrieved_arm"]).any()
    # O catalogo alcancavel exclui o proprio arm: nao e `len(catalog)`.
    assert result.metrics["catalog_size"] == len(catalog) - len(catalog) // len(arms)


def test_the_metrics_carry_what_the_figures_need_from_the_dataset():
    arms = ["m0", "m1"]
    queries = _multi_arm(arms, ["q0"], "query")
    catalog = _multi_arm(arms, ["k0"], "catalog")
    result = retrieve_by_arm(queries, catalog,
                             np.eye(len(queries), 4, dtype=np.float32),
                             np.eye(len(catalog), 4, dtype=np.float32))
    assert result.metrics["drive_db_ladder"] == [5.0, 10.0, 15.0, 20.0]
    assert result.metrics["strata"] == {"m0": "S1", "m1": "S1"}


def test_retrieve_by_arm_refuses_vectors_that_do_not_match_the_tables():
    queries = _multi_arm(["m0", "m1"], ["q0"], "query")
    catalog = _multi_arm(["m0", "m1"], ["k0"], "catalog")
    with pytest.raises(ValueError, match="desalinhados"):
        retrieve_by_arm(queries, catalog, np.zeros((1, 3), np.float32),
                        np.zeros((len(catalog), 3), np.float32))


def test_center_by_arm_removes_a_constant_offset_per_arm():
    from gefx.disent.retrieval import center_by_arm

    frame = pd.DataFrame({"arm": ["a", "a", "b", "b"]})
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    shifted = vectors + np.array([[5.0, 5.0], [5.0, 5.0], [-3.0, 2.0], [-3.0, 2.0]])
    assert np.allclose(center_by_arm(frame, shifted), center_by_arm(frame, vectors))
    assert np.allclose(center_by_arm(frame, shifted)[:2].mean(axis=0), 0.0)
