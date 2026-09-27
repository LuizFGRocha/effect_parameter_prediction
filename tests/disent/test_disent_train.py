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
CONTENTS = {"train": ["c0", "c1", "c2"], "catalog": ["c3", "c4"], "query": ["c5", "c6"]}
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
        np.savez(folder / "Spec.npz", features)
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


def _encoder_weights(model):
    return [np.array(v) for v in model.encoder.weights]


# --- configuracao -------------------------------------------------------------
def test_an_unknown_technique_is_refused_before_anything_is_loaded(tmp_path):
    with pytest.raises(KeyError, match="desconhecida"):
        train_module.train(TrainConfig(technique="full", dataset_root=tmp_path),
                           verbose=False)


def test_the_default_output_directory_is_named_after_the_technique():
    assert TrainConfig(technique="bn_only").resolved_output() == RESULTS_ROOT / "bn_only"


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


# --- fim a fim ----------------------------------------------------------------
def test_a_run_writes_every_artifact_that_makes_it_reproducible(tmp_path):
    manifest = train_module.train(_config(tmp_path), verbose=False)
    out = tmp_path / "out"
    for name in ("run.json", "metrics.json", "history.json", "predictions.csv",
                 "standardizer.npz"):
        assert (out / name).exists(), name
    assert (out / "weights" / "encoder.weights.h5").exists()
    assert manifest["config"]["technique"] == "supcon"
    assert manifest["splits"]["train"] == len(ARMS) * 3 * 4
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
    from gefx.disent import model as model_module

    vistos = []
    original = model_module.EffectModel

    class Espiao(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            vistos.append(self)
            self.inicial = {v.path: np.array(v) for v in self.encoder.weights}

    monkeypatch.setattr(train_module, "EffectModel", Espiao)
    manifest = train_module.train(_config(tmp_path, technique="bn_only", steps=5),
                                  verbose=False)
    modelo = vistos[0]
    treinaveis = {v.path for v in modelo.encoder.trainable_weights}
    mudaram = {v.path for v in modelo.encoder.weights
               if not np.array_equal(np.array(v), modelo.inicial[v.path])}
    assert mudaram, "a BatchNorm nao foi calibrada"
    assert not mudaram & treinaveis
    assert all("moving" in path for path in mudaram)
    assert manifest["steps_executed"] == 5
    assert json.loads((tmp_path / "out" / "history.json").read_text()) == []


def test_the_retrieval_never_answers_with_the_arm_that_asked(tmp_path):
    """O protocolo inteiro: sem isso a rede poderia acertar sem atravessar
    implementacao nenhuma, que e a pergunta do trabalho."""
    train_module.train(_config(tmp_path), verbose=False)
    predictions = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert not (predictions["query_arm"] == predictions["retrieved_arm"]).any()
    assert not set(predictions["query_content"]) & set(CONTENTS["catalog"])


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
    from gefx.disent.model import EffectModel
    from gefx.disent.train import embed

    config = _config(tmp_path)
    frame = split_frames(config.dataset_root)["catalog"]
    store = FeatureStore(config.dataset_root, frame, "Spec")
    standardizer = PixelStandardizer.fit(store)
    model = EffectModel(config.encoder)

    codes = embed(model, store, standardizer, batch=8)
    assert codes.shape == (len(frame), config.encoder.effect_dim)
    direto = model.encode(standardizer.transform(store.take(np.arange(3))))
    assert np.allclose(codes[:3], np.asarray(direto), atol=1e-6)


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
    (tmp_path / "b0.json").write_text(json.dumps(metricas), encoding="utf-8")
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
