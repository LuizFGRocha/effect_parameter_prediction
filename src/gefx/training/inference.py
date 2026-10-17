"""Carga de um modelo treinado e predicao.

Existe para que o treino e o estudo cross-implementation passem pelo mesmo
caminho de escala e predicao. Antes o `cross_impl` reimplementava o laco de
`transform`, e uma divergencia silenciosa ali invalidaria a comparacao.

`load_trained_chain` carrega um regressor do POC I; `open_run`, uma execucao do
POC II (`training/encoder.py`), com as particoes que ela viu.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from gefx.training.scaling import apply_scalers

MODEL_FILENAME = "model.keras"
SCALERS_FILENAME = "feature_scalers.pkl"


@dataclass
class TrainedChain:
    """Modelo de uma cadeia e os scalers com que ele foi treinado."""

    model: object
    scalers: Dict[int, StandardScaler]
    directory: Path

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Escala e prediz. `features` vem cru, no formato (n, altura, largura)."""
        return self.model.predict(apply_scalers(features, self.scalers), verbose=0)


def load_trained_chain(model_dir: str | Path) -> TrainedChain:
    directory = Path(model_dir)
    model_path = directory / MODEL_FILENAME
    scalers_path = directory / SCALERS_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers nao encontrados: {scalers_path}")

    # Depois das checagens: um caminho errado nao precisa pagar o TensorFlow.
    import keras

    return TrainedChain(
        model=keras.models.load_model(model_path),
        scalers=joblib.load(scalers_path),
        directory=directory,
    )


def clear_session() -> None:
    """Limpa a sessao Keras entre cadeias, para limitar o pico de memoria."""
    import gc

    import keras

    keras.backend.clear_session()
    gc.collect()


def list_runs(results_dir: str | Path) -> List[str]:
    """As execucoes do POC II gravadas em `results_dir`: toda subpasta com `run.json`."""
    return sorted(path.parent.name for path in Path(results_dir).glob("*/run.json"))


def load_run(run_dir: str | Path, code: str = "z_e"):
    """`(modelo, manifesto)` de uma execucao do POC II; o modelo da so a cabeca
    `code` (`z_e`, o que a busca le, ou `z_c`)."""
    import json

    from gefx.config import ArchitectureConfig
    from gefx.training.architecture import build_network, head
    from gefx.training.encoder import WEIGHTS_FILE

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    model = build_network(manifest["input_shape"], ArchitectureConfig(**config["architecture"]),
                          config["technique"])
    model.load_weights(run_dir / WEIGHTS_FILE)
    return head(model, code), manifest


def open_run(run_dir: str | Path, dataset_root: Optional[Path] = None, all_arms: bool = False,
             code: str = "z_e"):
    """`(modelo, manifesto, padronizador, particoes)` de uma execucao do POC II.

    As particoes sao as que ela viu: os arms (todos, com `all_arms`) e a validacao
    do manifesto, sobre `dataset_root` (por padrao, o do treino). O modelo da so a
    cabeca `code`.
    """
    from gefx.data.grid import split_frames
    from gefx.training.encoder import STANDARDIZER_FILE
    from gefx.training.scaling import PixelStandardizer

    model, manifest = load_run(run_dir, code)
    config = manifest["config"]
    frames = split_frames(Path(dataset_root or config["dataset_root"]),
                          None if all_arms else config["arms"],
                          config["validation_recordings"])
    standardizer = PixelStandardizer.load(Path(run_dir) / STANDARDIZER_FILE)
    return model, manifest, standardizer, frames
