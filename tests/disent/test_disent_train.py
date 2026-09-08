"""O laco de treino da etapa 5 e o estudo comparativo.

Dois testes carregam o peso deste arquivo: `test_a_run_writes_every_artifact...`,
porque um treino que nao grava o que o produziu nao e reproduzivel, e
`test_the_untrained_control_really_skips_optimization`, porque o controle de
arquitetura so vale como controle se ele de fato nao treinar -- e foi ele que
mostrou que boa parte do ganho sobre o B0 vinha da reducao de dimensao, e nao do
aprendizado.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent import train as train_module
from gefx.disent.train import (
    BASELINES,
    CRITERIA,
    TECHNIQUES,
    TrainConfig,
    aux_targets,
    compare,
    decide,
    split_frames,
)

ARMS = ("a0", "a1", "a2")
CONTENTS = {"train": ["c0", "c1", "c2"], "catalog": ["c3", "c4"], "query": ["c5", "c6"]}
DRIVES = (0, 1)
TONES = (0, 1)
SHAPE = (32, 24)


def _dataset(tmp_path, seed=0):
    """Dataset minusculo mas com a grade cruzada de verdade.

    Idempotente: um teste que treina duas tecnicas tem de reusar o mesmo
    dataset, senao a comparacao entre elas mediria dois datasets.
    """
    if (tmp_path / ARMS[0] / "metadata.csv").exists():
        return tmp_path
    rng = np.random.default_rng(seed)
    for arm_index, arm in enumerate(ARMS):
        rows = []
        for split, contents in CONTENTS.items():
            for content, (drive, tone) in itertools.product(
                contents, itertools.product(DRIVES, TONES)
            ):
                rows.append(
                    {
                        "file_name": f"{content}__d{drive}t{tone}.wav",
                        "arm": arm,
                        "content_id": content,
                        "config_index": drive * len(TONES) + tone,
                        "config_key": f"d{drive}t{tone}",
                        "drive_level": drive,
                        "tone_level": tone,
                        "drive_db_equivalente": 10.0 + 5.0 * drive,
                        "split": split,
                    }
                )
        frame = pd.DataFrame(rows)
        folder = tmp_path / arm / "distortion"
        folder.mkdir(parents=True)
        frame.to_csv(tmp_path / arm / "metadata.csv", index=False)
        # Sinal com estrutura: o nivel de drive move a media, o arm move o
        # offset. Sem estrutura nenhuma o treino nao teria o que aprender e os
        # testes de fim a fim nao distinguiriam funcionar de nao quebrar.
        features = (
            rng.normal(scale=0.1, size=(len(frame), *SHAPE)).astype(np.float32)
            + frame["drive_level"].to_numpy(dtype=np.float32)[:, None, None]
            + 0.05 * arm_index
        )
        np.savez(folder / "Spec.npz", features)
        (folder / "file_names.json").write_text(
            json.dumps(list(frame["file_name"])), encoding="utf-8"
        )
    return tmp_path


def _config(tmp_path, technique="contrastive", **kwargs):
    from gefx.disent.model import EncoderConfig

    defaults = dict(
        dataset_root=_dataset(tmp_path / "data"),
        technique=technique,
        steps=2,
        configs_per_batch=2,
        views_per_config=2,
        eval_every=0,
        embed_batch=8,
        seed=3,
        output_dir=tmp_path / "out",
        encoder=EncoderConfig(
            input_shape=(*SHAPE, 1), filters=(8, 16), trunk_units=16,
            effect_dim=4, content_dim=6, adversary_units=8,
        ),
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


# --- configuracao e criterios -------------------------------------------------
def test_every_technique_only_asks_for_registered_losses():
    from gefx.disent.losses import LOSS_REGISTRY

    for name, weights in TECHNIQUES.items():
        assert set(weights) <= set(LOSS_REGISTRY), name


def test_the_techniques_form_a_ladder_where_each_step_adds_one_idea():
    """A comparacao so atribui um ganho a uma ideia se as linhas vizinhas
    diferirem por uma ideia so."""
    ladder = ["contrastive", "contrastive_aux", "grl", "full"]
    for earlier, later in zip(ladder, ladder[1:]):
        assert set(TECHNIQUES[earlier]) < set(TECHNIQUES[later])


def test_an_unknown_technique_is_refused_before_anything_is_loaded():
    with pytest.raises(KeyError, match="inventada"):
        TrainConfig(technique="inventada").resolved_weights()


def test_explicit_weights_win_over_the_technique_preset():
    config = TrainConfig(technique="full", weights={"contrastive": 2.0})
    assert config.resolved_weights() == {"contrastive": 2.0}


def test_the_default_output_directory_is_named_after_the_technique():
    assert TrainConfig(technique="grl").resolved_output().name == "grl"


def test_the_criteria_are_declared_with_numbers_from_the_measured_baselines():
    assert set(CRITERIA) >= {
        "supera_b1_acerto", "supera_b0_acerto", "reduz_sorvedouro",
        "abaixo_do_teto", "supera_encoder_aleatorio",
    }
    assert BASELINES["B1_poc1_regressor"]["drive_exact"] == pytest.approx(0.317)
    assert BASELINES["paired_content_ceiling"]["drive_exact"] == pytest.approx(0.696)


def _metrics(drive_exact, mae_db, top_share):
    return {
        "overall": {"drive_level": {"exact": drive_exact}, "mae_db": mae_db},
        # `top_share` tem de ser o maximo: e a fracao do arm mais atrator.
        "hubness": {
            "by_retrieved_arm": {
                "a": top_share,
                "b": (1.0 - top_share) / 2,
                "c": (1.0 - top_share) / 2,
            }
        },
    }


def test_decide_reads_each_criterion_off_the_declared_thresholds():
    verdicts = decide(_metrics(0.40, 3.0, 0.30))["verdicts"]
    assert verdicts["supera_b1_acerto"] and verdicts["supera_b1_erro"]
    assert verdicts["supera_b0_acerto"] and verdicts["reduz_sorvedouro"]
    assert verdicts["abaixo_do_teto"]


def test_beating_the_paired_content_ceiling_is_a_failure_not_a_triumph():
    """O teto e o cenario do oraculo, com conteudo pareado. Passar dele nao e
    superar o oraculo, e vazamento."""
    assert not decide(_metrics(0.80, 1.0, 0.2))["verdicts"]["abaixo_do_teto"]


def test_criteria_that_need_another_run_are_left_out_and_not_passed_by_default():
    without = decide(_metrics(0.40, 3.0, 0.30))["verdicts"]
    assert "supera_encoder_aleatorio" not in without
    with_reference = decide(
        _metrics(0.40, 3.0, 0.30), {"random_encoder": {"drive_exact": 0.45}}
    )["verdicts"]
    assert with_reference["supera_encoder_aleatorio"] is False


def test_the_beta_vae_verdict_compares_against_the_contrastive_run():
    verdicts = decide(
        _metrics(0.50, 3.0, 0.3), {"contrastive": {"drive_exact": 0.40}},
        technique="beta_vae",
    )["verdicts"]
    assert verdicts["beta_vae_nao_ganha"] is False


def test_the_beta_vae_criterion_is_not_applied_to_the_supervised_techniques():
    """Aplicado a uma linha da escada, ele compararia duas tecnicas
    supervisionadas e reportaria falha por uma ser melhor que a outra -- o
    contrario do que a escada mostra."""
    verdicts = decide(
        _metrics(0.50, 3.0, 0.3), {"contrastive": {"drive_exact": 0.40}},
        technique="contrastive_aux",
    )["verdicts"]
    assert "beta_vae_nao_ganha" not in verdicts


def test_the_untrained_control_is_not_asked_to_beat_itself():
    verdicts = decide(
        _metrics(0.30, 5.6, 0.2), {"random_encoder": {"drive_exact": 0.30}},
        technique="random_encoder",
    )["verdicts"]
    assert "supera_encoder_aleatorio" not in verdicts


def test_the_study_order_makes_every_reference_available_before_it_is_needed():
    from gefx.disent.train import STUDY_ORDER

    assert set(STUDY_ORDER) == set(TECHNIQUES)
    posicao = {name: index for index, name in enumerate(STUDY_ORDER)}
    assert posicao["random_encoder"] == 0
    assert posicao["contrastive"] < posicao["beta_vae"]


def test_the_alternative_aggregate_drops_the_noisy_label_arm():
    from gefx.disent.train import EXCLUDED_FROM_AGGREGATE, aggregate_without

    metrics = {
        "per_query_arm": {
            "pedalboard-tanh": {"n": 10, "drive_level": {"exact": 0.6}, "mae_db": 3.0},
            "lsp-tanh": {"n": 10, "drive_level": {"exact": 0.5}, "mae_db": 3.5},
            "byod-bigmuff": {"n": 10, "drive_level": {"exact": 0.2}, "mae_db": 6.0},
        }
    }
    assert EXCLUDED_FROM_AGGREGATE == ("byod-bigmuff",)
    resumo = aggregate_without(metrics)
    assert resumo["arms"] == 2
    assert resumo["drive_exact"] == pytest.approx(0.55)
    assert resumo["mae_db"] == pytest.approx(3.25)


def test_the_alternative_aggregate_refuses_unbalanced_arms():
    """A media das taxas por arm so e o agregado das linhas se os arms tiverem o
    mesmo numero de consultas. Se deixarem de ter, isto tem de gritar."""
    from gefx.disent.train import aggregate_without

    metrics = {
        "per_query_arm": {
            "pedalboard-tanh": {"n": 10, "drive_level": {"exact": 0.6}, "mae_db": 3.0},
            "lsp-tanh": {"n": 7, "drive_level": {"exact": 0.5}, "mae_db": 3.5},
        }
    }
    with pytest.raises(ValueError, match="numeros de consultas diferentes"):
        aggregate_without(metrics)


def test_every_measured_summary_carries_the_aggregate_without_bigmuff(tmp_path):
    manifest = train_module.train(_config(tmp_path), verbose=False)
    assert "sem_bigmuff" in manifest["decision"]["measured"]


def test_rescore_reapplies_the_criteria_without_retraining(tmp_path):
    """Corrigir um criterio nao pode custar horas de GPU: o veredito sai de
    `metrics.json`, que ja esta em disco."""
    from gefx.disent.train import rescore

    estudo = tmp_path / "study"
    train_module.train(_config(tmp_path, technique="random_encoder",
                               output_dir=estudo / "random_encoder"), verbose=False)
    train_module.train(_config(tmp_path, technique="contrastive",
                               output_dir=estudo / "contrastive"), verbose=False)
    antes = json.loads((estudo / "contrastive" / "run.json").read_text())
    assert "supera_encoder_aleatorio" not in antes["decision"]["verdicts"]

    decisoes = rescore(estudo)
    assert "supera_encoder_aleatorio" in decisoes["contrastive"]["verdicts"]
    depois = json.loads((estudo / "contrastive" / "run.json").read_text())
    assert depois["decision"] == decisoes["contrastive"]
    assert depois["decision"]["measured"] == antes["decision"]["measured"]


# --- dados --------------------------------------------------------------------
def test_split_frames_gives_the_three_partitions_with_disjoint_contents(tmp_path):
    frames = split_frames(_dataset(tmp_path))
    assert set(frames) == {"train", "catalog", "query"}
    for first, second in itertools.combinations(frames.values(), 2):
        assert not set(first["content_id"]) & set(second["content_id"])


def test_split_frames_refuses_a_slice_that_empties_a_partition(tmp_path):
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match="vazias"):
        split_frames(root, arms=["nao-existe"])


def test_aux_targets_normalize_each_axis_by_the_levels_present(tmp_path):
    frame = split_frames(_dataset(tmp_path))["train"]
    targets = aux_targets(frame)
    assert targets.shape == (len(frame), 2)
    assert targets.min() == pytest.approx(0.0) and targets.max() == pytest.approx(1.0)


# --- fim a fim ----------------------------------------------------------------
def test_a_run_writes_every_artifact_that_makes_it_reproducible(tmp_path):
    manifest = train_module.train(_config(tmp_path), verbose=False)
    out = tmp_path / "out"
    for name in ("run.json", "metrics.json", "history.json", "predictions.csv",
                 "standardizer.npz"):
        assert (out / name).exists(), name
    assert (out / "weights" / "encoder.weights.h5").exists()
    assert manifest["config"]["weights"] == TECHNIQUES["contrastive"]
    assert manifest["splits"]["train"] == len(ARMS) * 3 * 4


def test_the_history_records_one_row_per_step_with_the_lambda_that_was_used(tmp_path):
    train_module.train(_config(tmp_path, steps=3), verbose=False)
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert [row["step"] for row in history] == [1, 2, 3]
    assert all(0.0 <= row["lambda"] <= 1.0 for row in history)


def test_only_the_losses_of_the_chosen_technique_show_up_in_the_history(tmp_path):
    train_module.train(_config(tmp_path, technique="contrastive"), verbose=False)
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert "adversary_arm" not in history[0]
    assert "contrastive" in history[0]


def test_the_untrained_control_really_skips_optimization(tmp_path):
    manifest = train_module.train(
        _config(tmp_path, technique="random_encoder", steps=50), verbose=False
    )
    assert manifest["steps_executed"] == 0
    assert json.loads((tmp_path / "out" / "history.json").read_text()) == []


def test_the_retrieval_never_answers_with_the_arm_that_asked(tmp_path):
    """O protocolo da etapa 4, inteiro: sem isso a rede poderia acertar sem
    atravessar implementacao nenhuma, que e a pergunta do trabalho."""
    train_module.train(_config(tmp_path), verbose=False)
    predictions = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert not (predictions["query_arm"] == predictions["retrieved_arm"]).any()
    assert not set(predictions["query_content"]) & set(CONTENTS["catalog"])


def test_the_full_technique_runs_every_registered_term(tmp_path):
    train_module.train(_config(tmp_path, technique="full"), verbose=False)
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert set(TECHNIQUES["full"]) <= set(history[0])


def test_the_beta_vae_control_trains_through_the_same_loop(tmp_path):
    manifest = train_module.train(
        _config(tmp_path, technique="beta_vae"), verbose=False
    )
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert set(history[0]) >= {"reconstruction", "kl", "total"}
    assert manifest["decision"]["measured"]["drive_exact"] >= 0.0


def test_a_misaligned_index_and_cache_is_refused_instead_of_trained_wrong(tmp_path, monkeypatch):
    """A armadilha silenciosa do POC I: rotulo e feature em ordens diferentes
    treina sem erro nenhum e produz um modelo errado."""
    config = _config(tmp_path)
    original = train_module.build_index

    def shuffled(frame, split=None, arms=None):
        return original(frame.iloc[::-1].reset_index(drop=True), split=split, arms=arms)

    monkeypatch.setattr(train_module, "build_index", shuffled)
    with pytest.raises(ValueError, match="ordens diferentes"):
        train_module.train(config, verbose=False)


def test_compare_reads_the_runs_from_disk_without_retraining(tmp_path):
    train_module.train(_config(tmp_path, output_dir=tmp_path / "study" / "contrastive"),
                       verbose=False)
    table = compare(tmp_path / "study")
    assert set(table["technique"]) >= set(BASELINES) | {"contrastive"}
    assert table.loc[table["technique"] == "contrastive", "kind"].iloc[0] == "learned"


def test_compare_skips_techniques_that_were_not_run(tmp_path):
    table = compare(tmp_path / "vazio")
    assert set(table["technique"]) == set(BASELINES)
