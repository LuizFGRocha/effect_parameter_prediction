"""Invariantes da recuperacao em catalogo (baseline B0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefx.disent.oracle import FFT_SIZES, N_MELS, log_mel_stack, spectral_distance
from gefx.disent.retrieval import (
    LEVEL_AXES,
    MEL_TIME_POOL,
    alphabet,
    cross_arm_table,
    descriptor,
    nearest,
    pool_time,
    retrieve,
    score,
)


# --- descritor ----------------------------------------------------------------
def test_pool_time_averages_within_each_band():
    frame = np.arange(12, dtype=float).reshape(2, 6)
    pooled = pool_time(frame, bands=3)
    assert pooled.shape == (2, 3)
    assert pooled[0] == pytest.approx([0.5, 2.5, 4.5])


def test_pool_time_rejects_more_bands_than_frames():
    with pytest.raises(ValueError, match="faixas de tempo"):
        pool_time(np.zeros((4, 2)), bands=3)


def test_pool_time_with_one_band_is_the_mean_spectrum():
    rng = np.random.default_rng(0)
    frame = rng.standard_normal((8, 20))
    assert pool_time(frame, bands=1)[:, 0] == pytest.approx(frame.mean(axis=1))


def test_descriptor_has_one_block_per_resolution():
    # A ponderacao do oraculo (media sobre resolucoes) so sobrevive a media
    # simples sobre o vetor achatado se todas contribuirem igual.
    sr = 22050
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((1, sr)) * 0.05).astype(np.float32)
    vector = descriptor(audio, sr)
    assert vector.shape == (len(FFT_SIZES) * N_MELS * MEL_TIME_POOL,)
    assert vector.dtype == np.float32


def test_descriptor_separates_a_distorted_signal_from_the_clean_one():
    sr = 22050
    t = np.arange(sr) / sr
    clean = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None, :]
    dirty = np.tanh(12.0 * clean).astype(np.float32)
    d_self = float(np.abs(descriptor(clean, sr) - descriptor(clean, sr)).mean())
    d_cross = float(np.abs(descriptor(clean, sr) - descriptor(dirty, sr)).mean())
    assert d_self == 0.0
    assert d_cross > 0.1


def test_pooled_distance_tracks_the_full_oracle_distance():
    # O descritor e uma REDUCAO da distancia do oraculo. Se a reducao invertesse
    # a ordem de um par obviamente ordenado, ela nao serviria de baseline.
    sr = 22050
    t = np.arange(sr) / sr
    clean = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None, :]
    pouco = np.tanh(2.0 * clean).astype(np.float32)
    muito = np.tanh(30.0 * clean).astype(np.float32)

    full_perto = spectral_distance(log_mel_stack(clean, sr), log_mel_stack(pouco, sr))
    full_longe = spectral_distance(log_mel_stack(clean, sr), log_mel_stack(muito, sr))
    red_perto = float(np.abs(descriptor(clean, sr) - descriptor(pouco, sr)).mean())
    red_longe = float(np.abs(descriptor(clean, sr) - descriptor(muito, sr)).mean())

    assert full_perto < full_longe
    assert red_perto < red_longe


# --- vizinho mais proximo -----------------------------------------------------
def test_nearest_finds_the_exact_row():
    rng = np.random.default_rng(0)
    catalog = rng.standard_normal((20, 8)).astype(np.float32)
    picks, dists = nearest(catalog[[3, 7, 11]], catalog)
    assert picks.tolist() == [3, 7, 11]
    assert dists == pytest.approx([0.0, 0.0, 0.0])


def test_nearest_is_independent_of_the_chunk_size():
    rng = np.random.default_rng(1)
    queries = rng.standard_normal((23, 5)).astype(np.float32)
    catalog = rng.standard_normal((17, 5)).astype(np.float32)
    a = nearest(queries, catalog, chunk=4)
    b = nearest(queries, catalog, chunk=100)
    assert a[0].tolist() == b[0].tolist()
    assert a[1] == pytest.approx(b[1])


def test_nearest_returns_the_mean_l1_distance():
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    catalog = np.array([[1.0, 3.0]], dtype=np.float32)
    _, dists = nearest(queries, catalog)
    assert dists[0] == pytest.approx(2.0)  # media, nao soma


def test_nearest_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimensoes incompativeis"):
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
                    "content_id": content,
                    "split": split,
                    "config_index": drive,
                    "drive_level": drive,
                    "tone_level": 0,
                    "drive_db_equivalente": 5.0 + 5.0 * drive,
                }
            )
    return pd.DataFrame(rows)


def test_retrieve_refuses_shared_content_between_query_and_catalog():
    # E a guarda que sustenta a leitura do resultado inteiro: com conteudo em
    # comum, acertar por reconhecer a execucao passa por acertar o ajuste.
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c1"], "catalog")
    with pytest.raises(ValueError, match="compartilham conteudo"):
        retrieve(q, c, np.zeros((len(q), 3), np.float32), np.zeros((len(c), 3), np.float32))


def test_retrieve_reports_what_was_asked_and_what_came_back():
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c2"], "catalog")
    # Descritor que codifica o nivel: a recuperacao tem de ser perfeita.
    qd = q[["drive_level"]].to_numpy(np.float32)
    cd = c[["drive_level"]].to_numpy(np.float32)
    out = retrieve(q, c, qd, cd)
    assert list(out["true_drive_level"]) == list(out["pred_drive_level"])
    assert set(out["retrieved_arm"]) == {"b"}
    assert out["distance"].to_numpy() == pytest.approx(0.0)


def test_score_reports_chance_next_to_every_accuracy():
    q = _frame("a", ["c1"], "query")
    c = _frame("b", ["c2"], "catalog")
    qd = q[["drive_level"]].to_numpy(np.float32)
    out = retrieve(q, c, qd, c[["drive_level"]].to_numpy(np.float32))
    metrics = score(out, alphabet(q))
    assert metrics["drive_level"]["exact"] == 1.0
    assert metrics["drive_level"]["chance"] == pytest.approx(0.25)
    assert metrics["drive_level"]["mae_levels"] == 0.0
    assert metrics["config_exact"] == 1.0
    assert metrics["mae_db"] == 0.0
    assert metrics["cross_implementation"] is True


def test_score_flags_a_catalog_that_leaked_the_query_arm():
    q = _frame("a", ["c1"], "query")
    c = _frame("a", ["c2"], "catalog")
    qd = q[["drive_level"]].to_numpy(np.float32)
    out = retrieve(q, c, qd, c[["drive_level"]].to_numpy(np.float32))
    assert score(out, alphabet(q))["cross_implementation"] is False


def test_score_counts_a_one_level_miss_as_within_one_but_not_exact():
    q = _frame("a", ["c1"], "query", levels=(0, 1))
    c = _frame("b", ["c2"], "catalog", levels=(0, 1))
    # Rotulos do catalogo deslocados de um nivel em relacao ao descritor: cada
    # consulta encontra o vizinho certo e recebe o rotulo errado por um.
    c["drive_level"] = [1, 2]
    c["drive_db_equivalente"] = [10.0, 15.0]
    qd = np.array([[0.0], [1.0]], dtype=np.float32)
    cd = np.array([[0.0], [1.0]], dtype=np.float32)
    out = retrieve(q, c, qd, cd)
    metrics = score(out, alphabet(q))
    assert metrics["drive_level"]["exact"] == 0.0
    assert metrics["drive_level"]["within_one"] == 1.0
    assert metrics["drive_level"]["mae_levels"] == 1.0
    assert metrics["mae_db"] == pytest.approx(5.0)


def test_alphabet_reads_the_slice_and_not_the_full_grid():
    # Um recorte com menos niveis tem outro acaso; reportar o da grade cheia
    # inflaria o resultado.
    frame = _frame("a", ["c1"], "query", levels=(0, 1))
    assert alphabet(frame) == {"drive_level": 2, "tone_level": 1}
    assert set(alphabet(frame)) == set(LEVEL_AXES)


def test_cross_arm_table_shows_where_the_hits_came_from():
    q = pd.concat([_frame("a", ["c1"], "query"), _frame("b", ["c1"], "query")],
                  ignore_index=True)
    c = pd.concat([_frame("x", ["c2"], "catalog"), _frame("y", ["c2"], "catalog")],
                  ignore_index=True)
    qd = q[["drive_level"]].to_numpy(np.float32)
    cd = c[["drive_level"]].to_numpy(np.float32)
    table = cross_arm_table(retrieve(q, c, qd, cd))
    assert set(table.index) == {"a", "b"}
    assert table["size"].to_numpy().sum() == len(q)


def test_block_size_respects_the_memory_budget():
    from gefx.disent.retrieval import block_size

    # 1.000 itens x 4.096 dimensoes x 4 bytes = 16,4 MB por consulta.
    assert block_size(1000, 4096, budget=64 * 1024 * 1024) == 4
    # Nunca zero, mesmo quando uma unica consulta ja estoura o orcamento.
    assert block_size(10**6, 4096, budget=1024) == 1


def test_nearest_sizes_its_own_block_when_none_is_given():
    rng = np.random.default_rng(2)
    queries = rng.standard_normal((9, 6)).astype(np.float32)
    catalog = rng.standard_normal((11, 6)).astype(np.float32)
    auto = nearest(queries, catalog)
    fixed = nearest(queries, catalog, chunk=1)
    assert auto[0].tolist() == fixed[0].tolist()


def test_paired_ceiling_keeps_content_fixed_and_crosses_implementations(tmp_path):
    # O teto so significa alguma coisa se cada consulta olhar para o MESMO
    # conteudo (senao nao e teto) e para OUTRA implementacao (senao nao mede
    # travessia). As duas coisas juntas, e nenhuma sozinha.
    import pandas as pd
    from gefx.disent.retrieval import paired_content_b0

    def _fake_root(root):
        for arm in ("a", "b"):
            pasta = root / arm
            pasta.mkdir(parents=True)
            linhas = []
            for content in ("c1", "c2"):
                for nivel in range(3):
                    linhas.append({
                        "file_name": f"{content}__d{nivel}.wav", "arm": arm,
                        "content_id": content, "split": "query",
                        "config_index": nivel, "drive_level": nivel,
                        "tone_level": 0, "drive_db_equivalente": 5.0 + 5.0 * nivel,
                    })
            pd.DataFrame(linhas).to_csv(pasta / "metadata.csv", index=False)

    _fake_root(tmp_path)
    # Descritor que codifica o nivel: a recuperacao pareada tem de ser exata.
    def fake_load(root, frame, bands=16, rebuild=False):
        return frame[["drive_level"]].to_numpy(dtype="float32")

    import gefx.disent.retrieval as mod
    original = mod.load_descriptors
    mod.load_descriptors = fake_load
    try:
        res = paired_content_b0(tmp_path)
    finally:
        mod.load_descriptors = original

    assert res.metrics["overall"]["drive_level"]["exact"] == 1.0
    # Cada consulta foi respondida pela outra implementacao, nunca pela propria.
    assert (res.predictions["query_arm"] != res.predictions["retrieved_arm"]).all()
    # E sempre dentro do proprio conteudo.
    assert res.metrics["overall"]["cross_implementation"] is True
    assert res.metrics["note"].startswith("teto")
