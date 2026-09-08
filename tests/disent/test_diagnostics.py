"""Sondas lineares sobre os codigos.

O modulo existe por causa de um resultado da etapa 5: o adversario de
implementacao ficou no acaso do primeiro passo ao ultimo e mesmo assim uma sonda
le a implementacao bem acima do acaso no mesmo `z_e`. Perda de adversario nao e
evidencia de remocao -- e a sonda que decide.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent.diagnostics import PROBE_FACTORS, linear_probes, probe_study


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


def test_the_three_factors_cover_both_halves_of_the_claim():
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


def test_probe_study_skips_techniques_that_were_not_run(tmp_path, monkeypatch):
    from gefx.disent import diagnostics

    (tmp_path / "contrastive").mkdir()
    (tmp_path / "contrastive" / "run.json").write_text(
        json.dumps({"config": {"technique": "contrastive"}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        diagnostics, "probe_run",
        lambda *args, **kwargs: {"probes": {"arm": {"accuracy": 0.3, "chance": 0.14,
                                                    "classes": 7, "above_chance": 0.2}}},
    )
    table = probe_study(tmp_path, tmp_path)
    assert list(table["technique"]) == ["contrastive"]
    assert list(table["factor"]) == ["arm"]
