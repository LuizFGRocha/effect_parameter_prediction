"""O laco de treino e os dois controles que separam aprendizado de arquitetura.

Os testes que carregam o peso deste arquivo sao os dos controles: um controle
so vale como controle se ele de fato nao treinar -- e foram eles que mostraram
que boa parte do ganho sobre o B0 vinha da arquitetura, e nao do aprendizado.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent import train as train_module
from gefx.disent.sidecar import split_frames
from gefx.disent.train import (
    RESULTS_ROOT,
    TECHNIQUES,
    TrainConfig,
    compare,
)

ARMS = ("a0", "a1", "a2")
# 2 x 20 gravacoes de treino vao para a validacao (`grid.VALIDATION_RECORDINGS`).
CONTENTS = {"train": [f"t{i:02d}" for i in range(44)], "catalog": ["c3", "c4"],
            "query": ["c5", "c6"]}
DRIVES = (0, 1, 2, 3)
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
            for content, drive in itertools.product(contents, DRIVES):
                rows.append(
                    {
                        "file_name": f"{content}__d{drive}.wav",
                        "arm": arm,
                        "stratum": "S1",
                        "content_id": content,
                        "config_index": drive,
                        "config_key": f"d{drive}",
                        "drive_level": drive,
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
        np.save(folder / "Spec.npy", features)
        (folder / "file_names.json").write_text(
            json.dumps(list(frame["file_name"])), encoding="utf-8"
        )
    return tmp_path


def _config(tmp_path, technique="supcon", **kwargs):
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
            input_shape=(*SHAPE, 1), filters=(8, 16), trunk_units=16, effect_dim=4,
        ),
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


# --- configuracao -------------------------------------------------------------
def test_an_unknown_technique_is_refused_before_anything_is_loaded(tmp_path):
    with pytest.raises(KeyError, match="desconhecida"):
        train_module.train(TrainConfig(technique="full", dataset_root=tmp_path),
                           verbose=False)


def test_the_default_output_directory_is_named_after_the_technique():
    assert TrainConfig(technique="bn_only").resolved_output() == RESULTS_ROOT / "bn_only"


# --- dados --------------------------------------------------------------------
def test_split_frames_gives_the_five_partitions_with_disjoint_contents(tmp_path):
    frames = split_frames(_dataset(tmp_path))
    assert set(frames) == {"train", "val_query", "val_catalog", "catalog", "query"}
    for first, second in itertools.combinations(frames.values(), 2):
        assert not set(first["content_id"]) & set(second["content_id"])


def test_the_validation_comes_out_of_the_train_and_never_out_of_the_test(tmp_path):
    frames = split_frames(_dataset(tmp_path), validation=3)
    validation = set(frames["val_query"]["content_id"]) | set(frames["val_catalog"]["content_id"])
    assert len(set(frames["val_query"]["content_id"])) == 3
    assert validation <= set(CONTENTS["train"])
    assert set(frames["train"]["content_id"]) | validation == set(CONTENTS["train"])


def test_the_validation_is_the_same_in_every_run(tmp_path):
    first, second = split_frames(_dataset(tmp_path)), split_frames(_dataset(tmp_path))
    assert first["val_query"].equals(second["val_query"])


def test_split_frames_refuses_a_slice_that_empties_a_partition(tmp_path):
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match="vazias"):
        split_frames(root, arms=["nao-existe"])


# --- fim a fim ----------------------------------------------------------------
def test_a_run_writes_every_artifact_that_makes_it_reproducible(tmp_path):
    manifest = train_module.train(_config(tmp_path), verbose=False)
    out = tmp_path / "out"
    for name in ("run.json", "metrics_validacao.json", "history.json",
                 "predictions_validacao.csv", "standardizer.npz"):
        assert (out / name).exists(), name
    assert (out / "weights" / "encoder.weights.h5").exists()
    assert manifest["config"]["technique"] == "supcon"
    assert manifest["splits"]["train"] == len(ARMS) * (len(CONTENTS["train"]) - 40) * 4
    assert manifest["splits"]["val_query"] == len(ARMS) * 20 * 4
    assert set(manifest["summary"]) == {"drive_exact", "mae_db",
                                        "same_arm_drive_exact"}


def test_the_history_records_the_loss_at_every_step(tmp_path):
    train_module.train(_config(tmp_path, steps=3), verbose=False)
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert [row["step"] for row in history] == [1, 2, 3]
    assert all(np.isfinite(row["loss"]) for row in history)


def test_the_untrained_control_really_skips_optimization(tmp_path):
    manifest = train_module.train(
        _config(tmp_path, technique="random_encoder", steps=50), verbose=False
    )
    assert manifest["steps_executed"] == 0
    assert json.loads((tmp_path / "out" / "history.json").read_text()) == []


def test_the_batchnorm_control_moves_only_the_moving_statistics(tmp_path, monkeypatch):
    """`bn_only` so vale como controle se nenhum peso treinavel mudar. Se mudar,
    ele passa a medir aprendizado e a diferenca para o encoder treinado some."""
    vistos = []
    original = train_module.build_model

    def espiao(config, technique):
        modelo = original(config, technique)
        vistos.append((modelo, {v.path: np.array(v) for v in modelo.weights}))
        return modelo

    monkeypatch.setattr(train_module, "build_model", espiao)
    manifest = train_module.train(_config(tmp_path, technique="bn_only", steps=5),
                                  verbose=False)
    modelo, inicial = vistos[0]
    treinaveis = {v.path for v in modelo.trainable_weights}
    mudaram = {v.path for v in modelo.weights
               if not np.array_equal(np.array(v), inicial[v.path])}
    assert mudaram, "a BatchNorm nao foi calibrada"
    assert not mudaram & treinaveis
    assert all("moving" in path for path in mudaram)
    assert manifest["steps_executed"] == 5
    assert json.loads((tmp_path / "out" / "history.json").read_text()) == []


def test_the_retrieval_never_answers_with_the_arm_that_asked(tmp_path):
    """O protocolo inteiro: sem isso a rede poderia acertar sem atravessar
    implementacao nenhuma, que e a pergunta do trabalho."""
    train_module.train(_config(tmp_path), verbose=False)
    predictions = pd.read_csv(tmp_path / "out" / "predictions_validacao.csv")
    assert not (predictions["query_arm"] == predictions["retrieved_arm"]).any()
    assert set(predictions["query_content"]) <= set(CONTENTS["train"])


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


def test_the_code_comes_out_of_the_encoder_in_the_row_order_of_the_frame(tmp_path):
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import build_encoder
    from gefx.disent.train import embed

    config = _config(tmp_path)
    frame = split_frames(config.dataset_root)["catalog"]
    store = FeatureStore(config.dataset_root, frame, "Spec")
    standardizer = PixelStandardizer.fit(store)
    model = build_encoder(config.encoder)

    codes = embed(model, store, standardizer, batch=8)
    assert codes.shape == (len(frame), config.encoder.effect_dim)
    direto = model(standardizer.transform(store.take(np.arange(3))), training=False)
    assert np.allclose(codes[:3], np.asarray(direto), atol=1e-6)


# --- o controle escalar -------------------------------------------------------
def test_the_scalar_control_trains_reloads_and_is_read_without_a_catalog(tmp_path):
    from gefx.disent.probes import embed_run

    manifest = train_module.train(_config(tmp_path, technique="regressao"), verbose=False)
    metrics = json.loads((tmp_path / "out" / "metrics_validacao.json").read_text())
    assert metrics["direct_readout"]["drive_level"]["chance"] == pytest.approx(0.25)
    assert {"direct_drive_exact", "direct_mae_db"} <= set(manifest["summary"])
    _, frame, codes = embed_run(tmp_path / "out", manifest["config"]["dataset_root"])
    assert codes.shape == (len(frame), 1)


def test_the_direct_readout_maps_the_unit_interval_back_onto_the_ladder(tmp_path):
    """0 e 1 sao os extremos da escada do treino; um valor exato acerta tudo."""
    frames = split_frames(_dataset(tmp_path))
    queries = frames["query"]
    exact = queries["drive_level"].to_numpy() / (len(DRIVES) - 1)
    readout = train_module.direct_readout(queries, frames["train"], exact)
    assert readout["drive_level"]["exact"] == pytest.approx(1.0)
    assert readout["mae_db"] == pytest.approx(0.0, abs=1e-9)


# --- a escada -----------------------------------------------------------------
def test_compare_reads_the_runs_from_disk_without_retraining(tmp_path):
    for technique in TECHNIQUES:
        train_module.train(
            _config(tmp_path, technique=technique,
                    output_dir=tmp_path / "study" / technique), verbose=False)
    table = compare(tmp_path / "study")
    assert set(table["run"]) == set(TECHNIQUES)


def test_compare_puts_the_baselines_that_retrieve_wrote_on_top(tmp_path):
    metricas = {"overall": {"drive_level": {"exact": 0.21}, "mae_db": 8.1},
                "alphabet": {"drive_level": 8}}
    (tmp_path / "b0_validacao.json").write_text(json.dumps(metricas), encoding="utf-8")
    table = compare(tmp_path / "encoder")
    assert list(table["run"]) == ["chance", "B0"]
    assert table.set_index("run").loc["chance", "drive_exact"] == pytest.approx(0.125)


def test_compare_of_an_empty_directory_is_empty(tmp_path):
    assert compare(tmp_path / "vazio").empty


def test_compare_skips_a_run_that_was_not_evaluated(tmp_path):
    pasta = tmp_path / "sem_avaliacao"
    pasta.mkdir(parents=True)
    (pasta / "run.json").write_text(json.dumps({"summary": None, "config": {}}))
    assert "sem_avaliacao" not in set(compare(tmp_path)["run"])


# --- o recorte de uma implementacao so (B2) -----------------------------------
def test_a_single_arm_slice_refuses_the_cross_implementation_evaluation(tmp_path):
    """Excluir o proprio arm de um catalogo de um arm so deixa catalogo vazio, e
    a busca morre com `argmax of an empty sequence`. O erro tem de dizer o que
    aconteceu e apontar a saida -- foi assim que o ponto B2 da curva de
    diversidade quebrou depois de treinar 13 minutos."""
    with pytest.raises(ValueError, match="ao menos 2 arms"):
        train_module.train(_config(tmp_path, arms=(ARMS[0],)), verbose=False)


def test_a_run_without_the_final_evaluation_still_saves_what_reloads_it(tmp_path):
    """Sem avaliacao interna a execucao continua reproduzivel: pesos,
    padronizador e o `run.json` com a configuracao. E disso que a
    curva de diversidade precisa -- quem pontua e o arm retirado, por fora."""
    config = _config(tmp_path, arms=(ARMS[0],), evaluate_at_end=False)
    manifest = train_module.train(config, verbose=False)
    saida = config.resolved_output()
    assert manifest["summary"] is None
    assert not (saida / "metrics.json").exists()
    assert not (saida / "predictions.csv").exists()
    assert (saida / "standardizer.npz").exists()
    assert json.loads((saida / "run.json").read_text())["config"]["arms"] == [ARMS[0]]


# --- determinismo -------------------------------------------------------------
def test_the_deterministic_flag_turns_on_the_tensorflow_kernels(tmp_path, monkeypatch):
    """Sem isto a semente fixa so a inicializacao: a perda do passo 1 e bit a bit
    igual entre duas execucoes e no passo 60 ja divergem, na GPU e na CPU. Com a
    bandeira ligada, 34 de 34 tensores saem identicos -- medido fora da suite,
    porque exige duas execucoes de verdade. Aqui se cobra que a bandeira chegue
    ao TensorFlow e ao manifesto, que e o que a suite pode garantir."""
    import tensorflow as tf

    chamadas = []
    monkeypatch.setattr(tf.config.experimental, "enable_op_determinism",
                        lambda: chamadas.append(True))
    config = _config(tmp_path, deterministic=True, evaluate_at_end=False)
    manifest = train_module.train(config, verbose=False)
    assert chamadas == [True]
    assert manifest["config"]["deterministic"] is True


def test_without_the_flag_nothing_is_turned_on(tmp_path, monkeypatch):
    import tensorflow as tf

    chamadas = []
    monkeypatch.setattr(tf.config.experimental, "enable_op_determinism",
                        lambda: chamadas.append(True))
    train_module.train(_config(tmp_path, evaluate_at_end=False), verbose=False)
    assert chamadas == []


# --- B0 -----------------------------------------------------------------------
def test_b0_searches_the_standardized_encoder_input_across_implementations(tmp_path):
    """O B0 le a mesma feature e a mesma padronizacao do encoder: no dataset
    sintetico o nivel de drive move a media, entao sem aprender nada ele ja tem
    de passar do acaso -- e nunca responder com o proprio arm."""
    from gefx.disent.retrieval import baseline_b0

    result = baseline_b0(_dataset(tmp_path))
    assert not (result.predictions["query_arm"] == result.predictions["retrieved_arm"]).any()
    overall = result.metrics["overall"]
    assert overall["drive_level"]["exact"] > overall["drive_level"]["chance"]


# --- parada pela validacao ----------------------------------------------------
def _scripted_validation(monkeypatch, errors):
    """Troca o erro de validacao por uma sequencia fixa e guarda os pesos de cada
    avaliacao, para conferir quais o treino devolve."""
    seen = []

    def fake(model, *args, **kwargs):
        seen.append([w.copy() for w in model.get_weights()])
        return {"mae_db": errors[len(seen) - 1], "drive_exact": 0.0, "same_arm": False}

    monkeypatch.setattr(train_module, "validation_error", fake)
    return seen


def test_training_stops_after_patience_evaluations_without_improvement(tmp_path, monkeypatch):
    _scripted_validation(monkeypatch, [3.0, 2.0, 1.0, 1.5, 1.2, 0.5, 0.4])
    manifest = train_module.train(
        _config(tmp_path, steps=7, eval_every=1, patience=2), verbose=False)
    assert manifest["steps_executed"] == 5
    assert manifest["best_step"] == 3
    assert manifest["best_val_mae_db"] == 1.0


def test_the_weights_that_stay_are_those_of_the_validation_minimum(tmp_path, monkeypatch):
    from gefx.disent.model import WEIGHTS_FILE, build_encoder

    seen = _scripted_validation(monkeypatch, [3.0, 1.0, 2.0, 2.5])
    config = _config(tmp_path, steps=4, eval_every=1, patience=2)
    train_module.train(config, verbose=False)
    model = build_encoder(config.encoder)
    model.load_weights(tmp_path / "out" / WEIGHTS_FILE)
    for saved, at_minimum in zip(model.get_weights(), seen[1]):
        assert np.array_equal(saved, at_minimum)


def test_patience_zero_runs_every_step(tmp_path, monkeypatch):
    _scripted_validation(monkeypatch, [1.0, 2.0, 3.0, 4.0])
    manifest = train_module.train(
        _config(tmp_path, steps=4, eval_every=1, patience=0), verbose=False)
    assert manifest["steps_executed"] == 4
    assert manifest["best_step"] == 1


def test_the_validation_error_is_the_test_search_on_the_validation_split(tmp_path):
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import build_encoder

    config = _config(tmp_path)
    frames = split_frames(config.dataset_root)
    stores = {name: FeatureStore(config.dataset_root, frames[name], "Spec")
              for name in ("train", "val_query", "val_catalog")}
    result = train_module.validation_error(
        build_encoder(config.encoder), frames, stores,
        PixelStandardizer.fit(stores["train"]), batch=8)
    assert result["same_arm"] is False
    assert 0.0 <= result["drive_exact"] <= 1.0 and result["mae_db"] >= 0.0


# --- validacao no desenvolvimento, teste so no fim ---------------------------
def test_training_reads_only_the_validation(tmp_path):
    train_module.train(_config(tmp_path), verbose=False)
    out = tmp_path / "out"
    frames = split_frames(_dataset(tmp_path / "data"))
    predictions = pd.read_csv(out / "predictions_validacao.csv")
    assert set(predictions["query_content"]) == set(frames["val_query"]["content_id"])
    assert not list(out.glob("*teste*"))


def test_the_final_pass_scores_the_saved_model_on_the_test(tmp_path):
    train_module.train(_config(tmp_path, output_dir=tmp_path / "study" / "supcon"),
                       verbose=False)
    table = train_module.evaluate_runs(tmp_path / "study", split="teste")
    run_dir = tmp_path / "study" / "supcon"
    predictions = pd.read_csv(run_dir / "predictions_teste.csv")
    assert set(predictions["query_content"]) == set(CONTENTS["query"])
    assert list(table["run"]) == ["supcon"]
    assert json.loads((run_dir / "metrics_teste.json").read_text())["split"] == "teste"


# --- a taxa de aprendizado ----------------------------------------------------
def test_the_cosine_goes_from_the_rate_to_the_supcon_floor_over_the_budget():
    """O `--cosine` do SupCon oficial: de `lr` a `lr * 0.1 ** 3` em `steps`."""
    config = TrainConfig(steps=1000, learning_rate=1e-3)
    schedule = train_module.learning_rate(config)
    assert float(schedule(0)) == pytest.approx(1e-3)
    assert float(schedule(500)) == pytest.approx(1e-3 * (1 + 1e-3) / 2, rel=1e-4)
    assert float(schedule(1000)) == pytest.approx(1e-6, rel=1e-3)


def test_the_constant_schedule_is_the_poc1_fixed_rate():
    assert train_module.learning_rate(TrainConfig(lr_schedule="constant")) == 1e-3


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="lr_schedule"):
        train_module.learning_rate(TrainConfig(lr_schedule="step"))
