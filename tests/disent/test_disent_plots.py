"""Contrato das figuras do POC II.

Nao verifica pixel: verifica que cada figura sai de dados reais e que as
convencoes que a tornam legivel -- ordem dos arms, ordem da escada -- nao mudam
sem alguem perceber, e que nada depende do roster: o dataset pode vir de outra
maquina com outros plugins.
"""
from __future__ import annotations

import json

import pandas as pd

from gefx.disent.plots import (
    TECHNIQUE_LABEL,
    build_all,
    ordered_arms,
    plot_baselines_by_arm,
    plot_diversity_curve,
)
from gefx.disent.train import TECHNIQUES

STRATA = {"ref": "S1", "gemeo": "S1", "clip": "S2", "fuzz": "S3"}
ARMS = list(STRATA)


def test_ordered_arms_goes_by_stratum_then_name_whatever_the_input_order():
    # Ordem estavel entre figuras e o que permite ler uma ao lado da outra.
    assert ordered_arms(list(reversed(ARMS)), STRATA) == ["gemeo", "ref", "clip", "fuzz"]


def _metrics():
    return {
        "per_query_arm": {
            arm: {
                "drive_level": {"exact": 0.2, "chance": 0.125, "mae_levels": 1.9,
                                "within_one": 0.47},
                "mae_db": 8.0,
            }
            for arm in ARMS
        },
        "overall": {"n": 5600, "drive_level": {"exact": 0.2}, "mae_db": 8.0},
        "catalog_size": 4800,
        "alphabet": {"drive_level": 8},
        "drive_db_ladder": [5.0 + 4.0 * nivel for nivel in range(8)],
        "strata": STRATA,
    }


def test_baselines_figure_is_written(tmp_path):
    destino = tmp_path / "baselines.png"
    plot_baselines_by_arm(_metrics(), _metrics(), destino)
    assert destino.exists() and destino.stat().st_size > 0


def test_every_technique_has_a_label():
    assert set(TECHNIQUE_LABEL) == set(TECHNIQUES)


def _run(root, nome, technique, drive_exact, mae_db, seed=1):
    pasta = root / "encoder" / nome
    pasta.mkdir(parents=True)
    (pasta / "run.json").write_text(json.dumps({
        "config": {"technique": technique, "seed": seed},
        "steps_executed": 0,
        "summary": {"drive_exact": drive_exact, "mae_db": mae_db,
                    "same_arm_drive_exact": drive_exact + 0.01},
    }), encoding="utf-8")
    (pasta / "metrics.json").write_text(json.dumps(_metrics()), encoding="utf-8")


def test_build_all_writes_what_there_is_data_for(tmp_path):
    _run(tmp_path, "random_encoder", "random_encoder", 0.31, 5.6)
    _run(tmp_path, "bn_only", "bn_only", 0.36, 4.8)
    _run(tmp_path, "rnc", "rnc", 0.44, 3.8)
    _run(tmp_path, "rnc_s2", "rnc", 0.43, 3.9, seed=2)
    nomes = {caminho.name for caminho in build_all(tmp_path)}
    assert nomes == {"escada.png", "por_arm.png"}


def test_build_all_adds_the_baselines_when_retrieve_wrote_them(tmp_path):
    for nome in ("b0.json", "b1.json"):
        (tmp_path / nome).write_text(json.dumps(_metrics()), encoding="utf-8")
    assert {c.name for c in build_all(tmp_path)} == {"baselines_por_arm.png"}


def test_build_all_of_an_empty_directory_writes_nothing(tmp_path):
    assert build_all(tmp_path) == []


def _curva():
    return pd.DataFrame([
        {"arm_retirado": "byod-mxr", "condicao": condicao, "k": k, "estratos": estratos,
         "drive_exact": acerto, "mae_db": 4.0, "n": 800}
        for k, estratos, acerto in [(1, 1, 0.20), (3, 2, 0.28), (6, 3, 0.33)]
        for condicao, acerto in (("transferencia", acerto), ("vistos", acerto + 0.1))
    ])


def test_the_diversity_curve_is_drawn_even_without_the_seen_reference(tmp_path):
    from gefx.disent.plots import plot_diversity_curve

    alvo = tmp_path / "curva.png"
    plot_diversity_curve(_curva(), 0.125, alvo)
    assert alvo.stat().st_size > 0


def test_the_curve_averages_the_seeds_and_still_shows_them(tmp_path):
    """A dispersao entre sementes e da ordem da excursao da curva. Uma linha
    media sozinha faria a curva parecer ter forma."""
    _run(tmp_path, "rnc", "rnc", 0.44, 3.8)
    for pasta, deslocamento in (("diversidade", 0.0), ("diversidade_s2", 0.05)):
        (tmp_path / "encoder" / pasta).mkdir()
        dados = _curva()
        dados["drive_exact"] = dados["drive_exact"] + deslocamento
        dados.to_csv(tmp_path / "encoder" / pasta / "resumo.csv", index=False)

    nomes = {caminho.name for caminho in build_all(tmp_path)}
    assert "curva_de_diversidade.png" in nomes


def test_the_ladder_takes_the_baselines_from_what_retrieve_wrote(tmp_path):
    for nome in ("b0.json", "b1.json"):
        (tmp_path / nome).write_text(json.dumps(_metrics()), encoding="utf-8")
    _run(tmp_path, "rnc", "rnc", 0.44, 3.8)
    nomes = {caminho.name for caminho in build_all(tmp_path)}
    assert {"baselines_por_arm.png", "escada.png", "por_arm.png"} <= nomes
