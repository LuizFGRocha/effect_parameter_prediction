"""Sondas lineares sobre o `z_e`: a evidencia de que o codigo guarda a
configuracao e perde conteudo e implementacao."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gefx.disent.probes import PROBE_FACTORS, linear_probes, probe_study


def _frame(n_arms=2, n_contents=4, n_drive=2, repeats=6):
    rows = []
    for arm, content, drive, _ in itertools.product(
        range(n_arms), range(n_contents), range(n_drive), range(repeats)
    ):
        rows.append({"arm": f"a{arm}", "content_id": f"c{content}", "drive_level": drive})
    return pd.DataFrame(rows)


def test_a_code_that_encodes_the_factor_is_read_at_the_ceiling():
    frame = _frame()
    codes = np.stack(
        [frame["drive_level"].to_numpy(dtype=float) * 10.0,
         np.zeros(len(frame))], axis=1
    )
    probes = linear_probes(codes, frame, factors=["drive_level"])
    assert probes["drive_level"]["accuracy"] == pytest.approx(1.0)
    assert probes["drive_level"]["above_chance"] == pytest.approx(1.0)


def test_a_code_that_carries_nothing_lands_at_chance():
    # Amostras de sobra: com poucas linhas por classe a sonda decora o ruido e
    # sai do acaso sozinha, que e um artefato do teste e nao do codigo.
    frame = _frame(repeats=40)
    codes = np.random.default_rng(0).normal(size=(len(frame), 3))
    probes = linear_probes(codes, frame, factors=["arm"])
    assert probes["arm"]["chance"] == pytest.approx(0.5)
    assert probes["arm"]["above_chance"] < 0.3


def test_above_chance_makes_factors_with_different_class_counts_comparable():
    """25% em 7 classes e 15% em 20 nao sao comparaveis crus; e a fracao do
    caminho entre acaso e acerto total que torna as duas linhas legiveis."""
    frame = _frame()
    codes = np.stack([frame["drive_level"].to_numpy(dtype=float), np.zeros(len(frame))], axis=1)
    probes = linear_probes(codes, frame, factors=["drive_level", "arm"])
    for numbers in probes.values():
        esperado = (numbers["accuracy"] - numbers["chance"]) / (1.0 - numbers["chance"])
        assert numbers["above_chance"] == pytest.approx(esperado)


def test_the_factors_cover_both_halves_of_the_claim_and_both_config_axes():
    """Sondar so o que deve sair mediria metade da afirmacao: um codigo
    constante zera implementacao e conteudo e nao serve para nada."""
    assert set(PROBE_FACTORS) == {"arm", "content_id", "drive_level"}


def test_a_factor_missing_from_the_sidecar_is_refused():
    with pytest.raises(KeyError, match="tempo"):
        linear_probes(np.zeros((len(_frame()), 2)), _frame(), factors=["tempo"])


def test_codes_and_rows_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="codigos"):
        linear_probes(np.zeros((3, 2)), _frame())


def test_the_probe_is_deterministic_in_the_seed():
    frame = _frame()
    codes = np.random.default_rng(1).normal(size=(len(frame), 4))
    first = linear_probes(codes, frame, factors=["arm"], seed=5)
    second = linear_probes(codes, frame, factors=["arm"], seed=5)
    assert first == second


def test_probe_study_refuses_a_directory_without_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.json"):
        probe_study(tmp_path, tmp_path)


def test_probe_study_gives_one_row_per_run_and_factor(tmp_path, monkeypatch):
    from gefx.disent import probes

    for nome in ("contrastive_aux", "random_encoder"):
        (tmp_path / nome).mkdir()
        (tmp_path / nome / "run.json").write_text("{}", encoding="utf-8")
    frame = _frame()
    codes = np.stack([frame["drive_level"].to_numpy(dtype=float),
                      np.zeros(len(frame))], axis=1)
    monkeypatch.setattr(
        probes, "embed_run",
        lambda run_dir, *args, **kwargs: (
            {"config": {"technique": run_dir.name}}, frame, codes),
    )
    table = probe_study(tmp_path, tmp_path)
    assert list(table["run"].unique()) == ["contrastive_aux", "random_encoder"]
    assert len(table) == 2 * len(PROBE_FACTORS)
    assert set(table["factor"]) == set(PROBE_FACTORS)


def test_the_probe_is_invariant_to_the_scale_of_the_code():
    """Sem padronizar, a `LogisticRegression` com C=1 da muito menos
    regularizacao efetiva a um codigo de norma grande, e a sonda passa a medir
    escala."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"drive_level": np.repeat([0, 1, 2], 30)})
    codigo = rng.normal(scale=0.3, size=(90, 5))
    codigo[:, 1] += frame["drive_level"].to_numpy()
    pequeno = linear_probes(codigo, frame, factors=["drive_level"])
    grande = linear_probes(codigo * 200.0, frame, factors=["drive_level"])
    assert pequeno["drive_level"]["accuracy"] == pytest.approx(
        grande["drive_level"]["accuracy"], abs=1e-9
    )
