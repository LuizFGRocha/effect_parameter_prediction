"""CLI: parsing e a traducao de flags em overrides de config.

Os handlers importam pesado por dentro, entao os testes substituem a funcao no
modulo de origem e olham so o que a CLI passou adiante.
"""
from __future__ import annotations

import pytest

from gefx.cli import main


@pytest.fixture
def captured_train(monkeypatch):
    """Intercepta `load_config` e `train`, devolvendo os overrides recebidos."""
    import gefx.config
    import gefx.training.trainer

    seen = {}

    def fake_load_config(path=None, **overrides):
        seen["path"] = path
        seen["overrides"] = overrides
        return "config-falsa"

    monkeypatch.setattr(gefx.config, "load_config", fake_load_config)
    monkeypatch.setattr(gefx.training.trainer, "train", lambda config: seen.setdefault("config", config))
    return seen


@pytest.fixture
def captured_render(monkeypatch):
    import gefx.config
    import gefx.data.render

    seen = {}

    def fake_load_config(path=None, **overrides):
        seen["overrides"] = overrides

        class Fake:
            dataset = "dataset-falso"

        return Fake()

    monkeypatch.setattr(gefx.config, "load_config", fake_load_config)
    monkeypatch.setattr(
        gefx.data.render, "generate_dataset", lambda config: seen.setdefault("dataset", config)
    )
    return seen


def test_unset_flags_arrive_as_none(captured_train):
    main(["train"])
    overrides = captured_train["overrides"]

    assert captured_train["path"] is None
    # `None` significa "nao informado" e nao sobrescreve o YAML.
    assert overrides["epochs"] is None
    assert overrides["feature"] is None
    assert overrides["chain_key"] is None


def test_store_true_flags_are_none_when_absent(captured_train):
    # O idioma `args.flag or None`: uma flag booleana ausente vira None em vez de
    # False, senao ela apagaria um `true` vindo do YAML.
    main(["train"])
    overrides = captured_train["overrides"]

    assert overrides["rebuild_cache"] is None
    assert overrides["deterministic"] is None
    assert overrides["clear_session"] is None


def test_store_true_flags_are_true_when_present(captured_train):
    main(["train", "--rebuild-cache", "--deterministic"])
    overrides = captured_train["overrides"]

    assert overrides["rebuild_cache"] is True
    assert overrides["deterministic"] is True


def test_no_clear_session_is_the_only_way_to_turn_a_boolean_off(captured_train):
    # CARACTERIZACAO: `--rebuild-cache` e `--deterministic` sao de mao unica — nao
    # ha flag que force `false` sobre um `true` do YAML. `clear_session` tem a
    # sua, por isso ela e `False` explicito e nao `None`.
    main(["train", "--no-clear-session"])
    assert captured_train["overrides"]["clear_session"] is False


def test_train_forwards_the_architecture_flags(captured_train):
    main(["train", "--n-nodes", "128", "--dropout", "0.5"])
    overrides = captured_train["overrides"]

    assert overrides["n_nodes"] == 128
    assert overrides["dropout"] == 0.5
    assert overrides["n_conv"] is None


def test_render_dataset_forwards_use_full_audio(captured_render):
    main(["render-dataset"])
    assert captured_render["overrides"]["use_full_audio"] is None

    main(["render-dataset", "--use-full-audio"])
    assert captured_render["overrides"]["use_full_audio"] is True


def test_every_forwarded_flag_is_a_real_config_field(monkeypatch):
    """Roda `load_config` de verdade nos dois handlers que passam overrides.

    Como `load_config` agora rejeita chave desconhecida, este teste falha se um
    handler mandar um nome que nao existe em nenhuma dataclass — que era
    exatamente o erro que sumia em silencio antes.
    """
    import gefx.data.render
    import gefx.training.trainer

    seen = []
    monkeypatch.setattr(gefx.training.trainer, "train", lambda config: seen.append(config))
    monkeypatch.setattr(gefx.data.render, "generate_dataset", lambda config: seen.append(config))

    main(
        ["train", "--dataset-root", "ds", "--results-root", "res", "--feature", "Spec",
         "--epochs", "1", "--batch-size", "8", "--test-size", "0.1", "--split-seed", "1",
         "--seed", "2", "--chain-key", "distortion", "--rebuild-cache", "--no-clear-session",
         "--deterministic", "--n-conv", "1", "--n-full", "2", "--n-nodes", "8",
         "--n-filters", "3", "--dropout", "0.1", "--learning-rate", "0.01"]
    )
    main(
        ["render-dataset", "--input-dir", "in", "--output-dir", "out",
         "--samples-per-file", "2", "--seed", "3", "--use-full-audio",
         "--segment-seconds", "1.0", "--ignore-start-seconds", "0.1",
         "--ignore-end-seconds", "0.1"]
    )

    train_config, dataset_config = seen
    assert train_config.train.epochs == 1
    assert train_config.architecture.n_nodes == 8
    assert dataset_config.samples_per_file == 2
    assert dataset_config.render_seed == 3  # `--seed` do render-dataset e o da renderizacao


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], None),                        # nao informado: nao sobrescreve o YAML
        (["--legacy"], True),
        (["--no-legacy"], False),          # unica forma de desligar um `legacy: true` do YAML
    ],
)
def test_render_dataset_legacy_is_two_way(captured_render, argv, expected):
    main(["render-dataset", *argv])
    assert captured_render["overrides"]["legacy"] is expected


def test_evaluate_forwards_legacy(monkeypatch):
    import gefx.evaluation.runner

    seen = {}
    monkeypatch.setattr(
        gefx.evaluation.runner,
        "evaluate_run",
        lambda root, legacy=False: seen.update(root=root, legacy=legacy),
    )
    main(["evaluate", "--results-root", "r"])
    assert seen == {"root": "r", "legacy": False}

    main(["evaluate", "--results-root", "r", "--legacy"])
    assert seen == {"root": "r", "legacy": True}


def test_cross_impl_render_forwards_legacy(monkeypatch):
    import gefx.crossimpl.render

    seen = {}
    monkeypatch.setattr(gefx.crossimpl.render, "render", lambda options: seen.update(o=options))
    main(["cross-impl", "render", "--legacy"])
    assert seen["o"].legacy is True

