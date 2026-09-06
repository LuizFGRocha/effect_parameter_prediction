"""Configuracao declarativa: precedencia default -> YAML -> flag da CLI."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gefx.config import TrainConfig, git_revision, load_config, write_run_manifest

BASE_YAML = Path(__file__).resolve().parents[1] / "experiments" / "base.yaml"


def write_yaml(tmp_path, text):
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_the_repository_base_experiment():
    config = load_config(BASE_YAML)
    assert config.train.feature == "Spec"
    assert config.train.epochs == 70
    assert config.architecture.kernel_size == (3, 3)
    assert config.dataset.samples_per_file == 56
    assert config.dataset.render_seed == 42
    # O experimento do POC I fica na convencao antiga do sidecar, para o dataset
    # sair igual ao que ja esta em disco.
    assert config.dataset.legacy is True


def test_yaml_overrides_dataclass_defaults(tmp_path):
    path = write_yaml(tmp_path, "train:\n  epochs: 5\n  feature: Chroma\n")
    config = load_config(path)
    assert config.train.epochs == 5
    assert config.train.feature == "Chroma"
    assert config.train.batch_size == 32  # nao mencionado: fica no default


def test_explicit_override_beats_yaml(tmp_path):
    path = write_yaml(tmp_path, "train:\n  epochs: 5\n")
    assert load_config(path, epochs=99).train.epochs == 99


def test_none_override_does_not_clobber_yaml(tmp_path):
    # E o que faz uma flag nao informada na CLI ser inofensiva.
    path = write_yaml(tmp_path, "train:\n  epochs: 5\n")
    assert load_config(path, epochs=None).train.epochs == 5


def test_empty_yaml_falls_back_to_defaults(tmp_path):
    assert load_config(write_yaml(tmp_path, "")).train == TrainConfig()


def test_rejects_a_non_mapping_yaml(tmp_path):
    with pytest.raises(ValueError, match="deve conter um mapeamento no topo"):
        load_config(write_yaml(tmp_path, "- um\n- dois\n"))


def test_rejects_unknown_sections(tmp_path):
    path = write_yaml(tmp_path, "treino:\n  epochs: 5\n")
    with pytest.raises(ValueError, match=r"Secoes desconhecidas .*\['treino'\]"):
        load_config(path)


def test_rejects_unknown_keys_within_a_section(tmp_path):
    path = write_yaml(tmp_path, "train:\n  epocas: 5\n")
    with pytest.raises(ValueError, match=r"Chaves desconhecidas em 'train': \['epocas'\]"):
        load_config(path)


def test_kernel_size_list_from_yaml_becomes_a_tuple(tmp_path):
    path = write_yaml(tmp_path, "architecture:\n  kernel_size: [5, 5]\n")
    assert load_config(path).architecture.kernel_size == (5, 5)


def test_rejects_an_unknown_feature():
    with pytest.raises(ValueError, match="feature='Mel' invalida"):
        load_config(feature="Mel")


def test_the_three_seeds_are_independent():
    # Os overrides da CLI sao achatados, entao um campo homonimo em duas secoes
    # seria escrito nas duas. `render_seed` tem nome proprio justamente por isso.
    config = load_config(seed=5, render_seed=6, split_seed=7)

    assert config.train.seed == 5           # init de pesos, dropout, ordem dos batches
    assert config.dataset.render_seed == 6  # segmentos e vetores de parametro
    assert config.train.split_seed == 7     # particao treino/teste


def test_train_seed_does_not_touch_the_render_seed():
    config = load_config(seed=5)
    assert config.train.seed == 5
    assert config.dataset.render_seed == 42  # intocado


def test_render_seed_does_not_touch_the_train_seed():
    config = load_config(render_seed=6)
    assert config.dataset.render_seed == 6
    assert config.train.seed == 42  # intocado


def test_no_field_name_is_shared_between_two_sections():
    # A colisao de nomes e o que fazia o override vazar; um campo novo homonimo
    # reintroduziria o problema.
    import dataclasses
    from collections import Counter

    from gefx.config import _SECTIONS

    counts = Counter(
        field.name for cls in _SECTIONS.values() for field in dataclasses.fields(cls)
    )
    assert [name for name, count in counts.items() if count > 1] == []


def test_rejects_unknown_override_keys():
    # Simetrico com o YAML, que ja levantava. Antes, um `--flag` novo na CLI que
    # ninguem ligou na dataclass sumia sem aviso.
    with pytest.raises(ValueError, match=r"Overrides desconhecidos: \['chave_que_nao_existe'\]"):
        load_config(chave_que_nao_existe=123)


def test_rejects_an_unknown_override_key_even_when_it_is_none():
    # A flag nao informada tambem passa por aqui; o nome errado precisa aparecer
    # mesmo quando o valor e None.
    with pytest.raises(ValueError, match="Overrides desconhecidos"):
        load_config(chave_que_nao_existe=None)


def test_write_run_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("gefx.config.git_revision", lambda: "deadbeef")
    config = load_config(feature="Spec")

    path = write_run_manifest(tmp_path / "res", config, [{"chain_key": "distortion", "mae": 0.1}])

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "run.json"
    assert manifest["git_revision"] == "deadbeef"
    assert manifest["config"]["train"]["feature"] == "Spec"
    assert manifest["metrics"] == [{"chain_key": "distortion", "mae": 0.1}]
    assert manifest["timestamp"].endswith("+00:00")


def test_write_run_manifest_creates_the_results_root(tmp_path):
    write_run_manifest(tmp_path / "fundo" / "do" / "poco", load_config())
    assert (tmp_path / "fundo" / "do" / "poco" / "run.json").exists()


def test_git_revision_returns_none_outside_a_repository(monkeypatch):
    def explode(*args, **kwargs):
        raise subprocess.SubprocessError("nao e um repositorio")

    monkeypatch.setattr(subprocess, "run", explode)
    assert git_revision() is None
