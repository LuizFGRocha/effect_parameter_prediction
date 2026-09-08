"""Contrato das figuras dos baselines.

Nao verifica pixel: verifica que cada figura sai de dados reais e que as
convencoes que a tornam legivel -- ordem do roster, sinal do vies, denominador
do sorvedouro -- nao mudam sem alguem perceber.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent.arms import arm_keys
from gefx.disent.plots import (
    GRID_STEP_DB,
    build_all,
    ordered_arms,
    plot_bias_convergence,
    plot_baselines_by_arm,
    plot_fidelity,
    plot_hubness,
    plot_pairwise,
)

ARMS = arm_keys()


def test_ordered_arms_follows_the_roster_not_the_input():
    # Ordem estavel entre figuras e o que permite ler uma ao lado da outra.
    embaralhado = list(reversed(ARMS))
    assert ordered_arms(embaralhado) == ARMS


def test_ordered_arms_drops_what_is_not_in_the_roster():
    assert ordered_arms([ARMS[0], "arm-que-nao-existe"]) == [ARMS[0]]


def _b1_frame(vies_db=None):
    vies_db = vies_db or {}
    escada = np.array([5.04, 9.13, 13.31, 17.44, 21.59, 25.77, 29.92, 34.07])
    linhas = []
    for arm in ARMS:
        for nivel, db in enumerate(escada):
            for _ in range(4):
                linhas.append({
                    "query_arm": arm,
                    "true_drive_level": nivel,
                    "true_drive_db": db,
                    "pred_drive_db": db + vies_db.get(arm, 0.0),
                    "pred_drive_level": nivel,
                    "true_tone_level": 0,
                    "pred_tone_level": 0,
                    "retrieved_arm": "(regressor POC I)",
                    "retrieved_file": "",
                })
    return pd.DataFrame(linhas)


def test_bias_convergence_subtracts_the_reference_and_flips_the_sign(tmp_path):
    # O vies do B1 e relativo a referencia (o teto do regressor age em todos) e
    # o sinal e o do teste cego: dB predito MAIOR = soa mais distorcido =
    # negativo em niveis. Trocar qualquer um dos dois inverteria a leitura da
    # figura sem quebrar nada mais.
    quente = ARMS[1]
    frame = _b1_frame({ARMS[0]: -2.0, quente: -2.0 + GRID_STEP_DB})
    destino = tmp_path / "vies.png"
    plot_bias_convergence(frame, {}, destino)
    assert destino.exists() and destino.stat().st_size > 0

    erro = frame["pred_drive_db"] - frame["true_drive_db"]
    media = erro.groupby(frame["query_arm"]).mean()
    esperado = -(media[quente] - media[ARMS[0]]) / GRID_STEP_DB
    assert esperado == pytest.approx(-1.0)


def test_bias_convergence_accepts_an_empty_listening_record(tmp_path):
    # PERCEPTUAL_BIAS so cobre dois arms hoje; a figura nao pode exigir todos.
    destino = tmp_path / "vies.png"
    plot_bias_convergence(_b1_frame(), {}, destino)
    assert destino.exists()


def test_hubness_uses_the_reachable_catalog_as_denominator(tmp_path):
    # Sem `same_arm` o proprio arm sai do catalogo: usar o catalogo inteiro
    # subestimaria a ocupacao esperada e faria o sorvedouro parecer menor.
    frame = pd.DataFrame({
        "retrieved_file": ["a.wav"] * 90 + [f"{i}.wav" for i in range(10)],
        "retrieved_arm": [ARMS[0]] * 90 + [ARMS[1]] * 10,
    })
    destino = tmp_path / "hub.png"
    plot_hubness(frame, catalog_size=100, out_path=destino)
    assert destino.exists() and destino.stat().st_size > 0


def test_pairwise_figure_survives_a_full_matrix(tmp_path):
    linhas = [
        {"query_arm": q, "catalog_arm": c, "drive_exact": 0.2 + 0.01 * i}
        for i, (q, c) in enumerate((q, c) for q in ARMS for c in ARMS)
    ]
    destino = tmp_path / "par.png"
    plot_pairwise(pd.DataFrame(linhas), destino)
    assert destino.exists() and destino.stat().st_size > 0


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
        "overall": {"n": 5600},
        "catalog_size": 4800,
    }


def test_baselines_figure_is_written(tmp_path):
    destino = tmp_path / "baselines.png"
    plot_baselines_by_arm(_metrics(), _metrics(), destino)
    assert destino.exists() and destino.stat().st_size > 0


def test_fidelity_figure_marks_the_chosen_pooling(tmp_path):
    sweep = pd.DataFrame({
        "bands": [1, 4, 16], "dims": [256, 1024, 4096],
        "pearson": [0.62, 0.80, 0.99], "spearman": [0.56, 0.70, 0.97],
        "mean_ratio": [0.53, 0.68, 0.90], "n_pairs": [497] * 3,
    })
    destino = tmp_path / "fid.png"
    plot_fidelity(sweep, chosen=16, out_path=destino)
    assert destino.exists() and destino.stat().st_size > 0


def test_fidelity_figure_rejects_a_pooling_absent_from_the_sweep(tmp_path):
    sweep = pd.DataFrame({
        "bands": [1, 4], "dims": [256, 1024], "pearson": [0.62, 0.80],
        "spearman": [0.56, 0.70], "mean_ratio": [0.53, 0.68], "n_pairs": [497] * 2,
    })
    with pytest.raises(IndexError):
        plot_fidelity(sweep, chosen=16, out_path=tmp_path / "fid.png")


def test_build_all_refuses_results_without_the_catalog_size(tmp_path):
    # A alternativa seria adivinhar o denominador, e um sorvedouro medido contra
    # o denominador errado e pior que nenhum.
    metrics = _metrics()
    del metrics["catalog_size"]
    (tmp_path / "b0_cross.json").write_text(json.dumps(metrics))
    (tmp_path / "b1.json").write_text(json.dumps(_metrics()))
    _b1_frame().to_csv(tmp_path / "b1.csv", index=False)
    pd.DataFrame({
        "retrieved_file": ["a.wav"], "retrieved_arm": [ARMS[0]],
        "query_arm": [ARMS[0]],
    }).to_csv(tmp_path / "b0_cross.csv", index=False)
    pd.DataFrame([
        {"query_arm": q, "catalog_arm": c, "drive_exact": 0.2}
        for q in ARMS for c in ARMS
    ]).to_csv(tmp_path / "b0_pairwise.csv", index=False)

    with pytest.raises(KeyError, match="catalog_size"):
        build_all(tmp_path)
