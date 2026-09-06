"""Configuracao declarativa dos experimentos.

Precedencia: default da dataclass -> arquivo YAML (`--config`) -> flag de CLI.
Sem `--config` e sem flags, os defaults reproduzem o comportamento historico do
projeto, para o `second_main_run` seguir comparavel.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FEATURE_CHOICES = ("Spec", "MFCC40", "Chroma", "GFCC40")


@dataclass
class DatasetConfig:
    """Renderizacao do dataset a partir das gravacoes limpas."""

    input_dir: str = "datasets/unprocessed_samples"
    output_dir: str = "datasets/default"
    samples_per_file: int = 56
    # Nome proprio, e nao `seed`: os overrides da CLI sao achatados, entao um
    # campo homonimo em duas secoes seria escrito nas duas de uma vez.
    render_seed: int = 42
    use_full_audio: bool = False
    segment_seconds: float = 2.0
    ignore_start_seconds: float = 0.5
    ignore_end_seconds: float = 0.5
    # Serializa `effect_presence` e `raw_parameter_dict` em ordem alfabetica, como
    # antes, em vez da ordem do catalogo. Ligado em `experiments/base.yaml` para o
    # sidecar do POC I continuar byte a byte igual.
    legacy: bool = False


@dataclass
class ArchitectureConfig:
    """Hiperparametros da CNN. Os defaults sao os usados no POC I."""

    kernel_size: Tuple[int, int] = (3, 3)
    n_conv: int = 2
    n_full: int = 3
    n_nodes: int = 64
    n_filters: int = 6
    dropout: float = 0.2
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        # YAML entrega listas; a Keras espera tupla.
        if isinstance(self.kernel_size, list):
            self.kernel_size = tuple(self.kernel_size)


@dataclass
class TrainConfig:
    """Treino por cadeia."""

    dataset_root: str = "datasets/default"
    results_root: str = "results/default_results"
    feature: str = "MFCC40"
    epochs: int = 60
    batch_size: int = 32
    test_size: float = 0.20
    split_seed: int = 42
    # Fixa numpy/random/tensorflow: sem isso so o split e reproduzivel, o treino nao.
    seed: int = 42
    # Kernels deterministicos na GPU. Sem isso, duas runs com o mesmo seed ainda
    # divergem na ordem de 1e-4 (reducoes atomicas do cuDNN). Custa velocidade,
    # entao fica desligado por padrao e se liga quando o experimento exige
    # comparar runs bit a bit.
    deterministic: bool = False
    chain_key: Optional[str] = None
    rebuild_cache: bool = False
    clear_session: bool = True

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_CHOICES:
            raise ValueError(
                f"feature={self.feature!r} invalida. Esperado um de {list(FEATURE_CHOICES)}"
            )


@dataclass
class ExperimentConfig:
    """Um experimento completo: dados, arquitetura e treino."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "dataset": DatasetConfig,
    "architecture": ArchitectureConfig,
    "train": TrainConfig,
}


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> ExperimentConfig:
    """Monta a config a partir dos defaults, do YAML e dos overrides da CLI.

    `overrides` sao pares chave-valor achatados (ex.: `epochs=70`), aplicados na
    secao a que pertencem; valores `None` sao ignorados, para uma flag nao
    informada nao sobrescrever o que veio do YAML.
    """
    raw: Dict[str, Any] = {}
    if path is not None:
        import yaml

        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {path} deve conter um mapeamento no topo.")
        unknown = sorted(set(loaded) - set(_SECTIONS))
        if unknown:
            raise ValueError(
                f"Secoes desconhecidas em {path}: {unknown}. Esperado {sorted(_SECTIONS)}."
            )
        raw = loaded

    sections = {}
    for name, cls in _SECTIONS.items():
        values = dict(raw.get(name) or {})
        unknown = sorted(set(values) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"Chaves desconhecidas em '{name}': {unknown}")
        sections[name] = values

    known_fields = {f.name for cls in _SECTIONS.values() for f in dataclasses.fields(cls)}
    unknown = sorted(set(overrides) - known_fields)
    if unknown:
        raise ValueError(f"Overrides desconhecidos: {unknown}")

    for key, value in overrides.items():
        if value is None:
            continue
        for name, cls in _SECTIONS.items():
            if key in {f.name for f in dataclasses.fields(cls)}:
                sections[name][key] = value

    return ExperimentConfig(
        dataset=DatasetConfig(**sections["dataset"]),
        architecture=ArchitectureConfig(**sections["architecture"]),
        train=TrainConfig(**sections["train"]),
    )


def git_revision() -> Optional[str]:
    """SHA do HEAD, ou None fora de um repositorio git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def write_run_manifest(
    results_root: Path,
    config: ExperimentConfig,
    metrics: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Grava `run.json`: config resolvida, git SHA, timestamp e resumo das metricas."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "config": config.to_dict(),
        "metrics": metrics or [],
    }
    path = results_root / "run.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def set_global_seeds(seed: int, deterministic: bool = False) -> None:
    """Fixa random, numpy e tensorflow. Chamar antes de construir o modelo.

    `deterministic=True` liga tambem os kernels deterministicos do TensorFlow;
    sem isso o seed sozinho nao garante duas runs identicas na GPU.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
