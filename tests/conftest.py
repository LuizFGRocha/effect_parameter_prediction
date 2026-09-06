"""Fixtures compartilhadas.

As fixtures montam datasets e diretorios de resultado sinteticos usando o proprio
codigo de producao (`RenderRecord`, `write_metadata_csv`, `_predictions_frame`),
para que um teste que le tambem exercite quem escreve. Nada aqui depende de
`datasets/` nem de `results/`, que sao grandes e ficam fora do git.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

# Precisa vir antes de qualquer import de `evaluation.plots`.
matplotlib.use("Agg")

from gefx.data.metadata import METADATA_FILENAME, RenderRecord, write_metadata_csv
from gefx.effects.catalog import chain_key, chain_output_dim, parameter_names_for_chain
from gefx.effects.parameters import binary_suffix, effect_presence, split_vector_by_effect


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Zera o estado de processo que vazaria de um teste para o outro."""
    from gefx.audio import get_meter
    from gefx.effects import vst_adapter

    get_meter.cache_clear()
    vst_adapter._WARNED.clear()
    yield
    get_meter.cache_clear()
    vst_adapter._WARNED.clear()


@pytest.fixture
def make_record():
    """Expoe `_make_record` como fixture: importar de `tests.conftest` colidiria
    com um pacote `tests` que existe em site-packages."""
    return _make_record


def _make_norm_vector(dim: int, row: int) -> list[float]:
    """Vetor normalizado distinto por linha, para ordenacao errada ficar visivel."""
    return [round(0.05 * (row + 1) + 0.01 * index, 4) for index in range(dim)]


def _make_record(chain_effects, file_name: str, norm_vector, source="src.wav", seed=0):
    return RenderRecord(
        file_name=file_name,
        chain_key=chain_key(chain_effects),
        chain_length=len(chain_effects),
        effect_order=list(chain_effects),
        effect_presence=effect_presence(chain_effects),
        normalized_parameter_vector=list(norm_vector),
        raw_parameter_dict=split_vector_by_effect(chain_effects, norm_vector),
        source_audio_id=source,
        random_seed=seed,
    )


@pytest.fixture
def fake_dataset(tmp_path):
    """Factory de uma raiz de dataset: pastas de cadeia, wavs vazios e sidecar.

    Os wavs sao criados com `touch`: nem o validador nem os testes de cache leem
    audio de verdade. `mutate` recebe a lista de `RenderRecord` antes da escrita,
    e e por onde os testes produzem sidecars quebrados.
    """

    def build(chains=None, samples=3, write_wavs=True, mutate=None, root=None):
        chains = chains if chains is not None else [["distortion"], ["chorus"]]
        root = Path(root) if root is not None else tmp_path / "dataset"

        records = []
        for chain_effects in chains:
            chain_key_value = chain_key(chain_effects)
            folder = root / chain_key_value
            folder.mkdir(parents=True, exist_ok=True)
            dim = chain_output_dim(chain_key_value)
            bits = binary_suffix(chain_effects)
            for index in range(samples):
                file_name = f"src__{bits}__s{index:04d}.wav"
                if write_wavs:
                    (folder / file_name).touch()
                records.append(
                    _make_record(
                        chain_effects,
                        file_name,
                        _make_norm_vector(dim, index),
                        seed=index,
                    )
                )

        if mutate is not None:
            records = mutate(records)
        write_metadata_csv(root / METADATA_FILENAME, records)
        return root

    return build


def _build_chain_results(chain_key_value, y_true, y_pred, root, **overrides):
    """Escreve um diretorio de cadeia no formato de `trainer.train_one_chain`,
    inclusive as colunas `y_true_<p>`/`y_pred_<p>` intercaladas."""
    import numpy as np

    from gefx.training.trainer import _predictions_frame

    out_dir = Path(root) / chain_key_value
    out_dir.mkdir(parents=True, exist_ok=True)

    names = parameter_names_for_chain(chain_key_value)
    y_true = np.asarray(y_true if y_true is not None else [[0.2] * len(names)], dtype=float)
    y_pred = np.asarray(y_pred if y_pred is not None else [[0.3] * len(names)], dtype=float)
    file_names = np.array([f"f{index:03d}.wav" for index in range(len(y_true))])

    _predictions_frame(file_names, y_true, y_pred, names).to_csv(
        out_dir / "predictions.csv", index=False
    )

    metrics = {
        "chain_key": chain_key_value,
        "chain_length": len(chain_key_value.split("__")),
        "effect_order": chain_key_value.split("__"),
        "feature": "Spec",
        "n_train": 8,
        "n_test": len(y_true),
        "output_dim": len(names),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "parameter_names": names,
    }
    metrics.update(overrides)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_dir


@pytest.fixture(scope="session")
def chain_results_builder():
    """Para fixtures de escopo maior que a funcao, que nao podem usar `tmp_path`."""
    return _build_chain_results


@pytest.fixture
def fake_chain_results(tmp_path):
    """Factory de um diretorio de cadeia com `predictions.csv` + `metrics.json`."""

    def build(chain_key_value="distortion", y_true=None, y_pred=None, root=None, **overrides):
        return _build_chain_results(
            chain_key_value, y_true, y_pred, tmp_path / "results" if root is None else root,
            **overrides,
        )

    return build
