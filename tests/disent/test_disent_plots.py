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


# --- figuras da etapa 5 -------------------------------------------------------
def _etapa5_run(root, nome, drive_exact, mae_db, passos=3, checkpoints=None):
    import json

    pasta = root / nome
    pasta.mkdir(parents=True)
    (pasta / "run.json").write_text(
        json.dumps(
            {
                "decision": {"measured": {"drive_exact": drive_exact, "mae_db": mae_db,
                                          "top_arm_share": 0.2}},
                "checkpoints": checkpoints or [],
                "config": {"steps": passos},
                "steps_executed": passos,
            }
        ),
        encoding="utf-8",
    )
    (pasta / "metrics.json").write_text(
        json.dumps(
            {
                "alphabet": {"drive_level": 8, "tone_level": 5},
                "per_query_arm": {
                    arm: {"drive_level": {"exact": 0.5 if arm != "byod-bigmuff" else 0.2}}
                    for arm in ("pedalboard-tanh", "lsp-tanh", "byod-bigmuff")
                },
                "hubness": {"by_retrieved_arm": {"pedalboard-tanh": 0.4, "lsp-tanh": 0.35,
                                                 "byod-bigmuff": 0.25}},
            }
        ),
        encoding="utf-8",
    )
    (pasta / "history.json").write_text(
        json.dumps(
            [
                {"step": passo, "lambda": passo / passos, "total": 3.0 - passo * 0.1,
                 "contrastive": 3.0 - passo * 0.1, "adversary_arm": 1.9}
                for passo in range(1, passos + 1)
            ]
        ),
        encoding="utf-8",
    )
    return pasta


def test_the_technique_order_is_the_ladder_and_not_the_ranking():
    """Ordenar por resultado esconderia o que a escada mostra: que cada linha
    acrescenta uma ideia a anterior."""
    from gefx.disent.plots import TECHNIQUE_ORDER, ordered_techniques

    assert ordered_techniques({"full", "contrastive", "random_encoder"}) == [
        "random_encoder", "contrastive", "full"
    ]
    assert list(TECHNIQUE_ORDER).index("contrastive") < list(TECHNIQUE_ORDER).index("full")


def test_every_technique_of_the_study_has_a_label():
    from gefx.disent.plots import TECHNIQUE_LABEL, TECHNIQUE_ORDER
    from gefx.disent.train import TECHNIQUES

    assert set(TECHNIQUE_ORDER) == set(TECHNIQUES)
    assert set(TECHNIQUE_LABEL) == set(TECHNIQUES)


def test_build_etapa5_writes_every_figure_from_the_runs_on_disk(tmp_path):
    from gefx.disent.plots import build_etapa5

    _etapa5_run(tmp_path, "random_encoder", 0.30, 5.6, passos=0)
    _etapa5_run(tmp_path, "full", 0.45, 3.8,
                checkpoints=[{"step": 1, "drive_exact": 0.40, "mae_db": 4.2}])
    escritos = build_etapa5(tmp_path, tmp_path / "figuras")
    nomes = {caminho.name for caminho in escritos}
    assert nomes == {
        "escada_de_tecnicas.png", "curvas_de_treino.png",
        "sorvedouro_por_tecnica.png", "por_arm_melhor_tecnica.png",
    }
    assert all(caminho.stat().st_size > 0 for caminho in escritos)


def test_build_etapa5_refuses_a_directory_without_runs(tmp_path):
    from gefx.disent.plots import build_etapa5

    (tmp_path / "vazio").mkdir()
    with pytest.raises(FileNotFoundError, match="run.json"):
        build_etapa5(tmp_path)


def test_the_untrained_control_has_no_curve_and_does_not_break_the_figure(tmp_path):
    """O controle nao treina, entao nao tem historico. A figura tem de sair
    assim mesmo -- e ele que ancora a comparacao."""
    from gefx.disent.plots import build_etapa5

    _etapa5_run(tmp_path, "random_encoder", 0.30, 5.6, passos=0)
    escritos = build_etapa5(tmp_path, tmp_path / "figuras")
    assert not any(caminho.name == "curvas_de_treino.png" for caminho in escritos)
