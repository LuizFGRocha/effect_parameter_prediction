"""Sidecar `metadata.csv`: escrita, leitura e checagem de integridade.

O sidecar fica na raiz do dataset e e a fonte dos alvos: `data/dataset.py` casa
cada wav em cache com a sua linha aqui pelo nome do arquivo.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from gefx.effects.catalog import chain_key

METADATA_FILENAME = "metadata.csv"

METADATA_COLUMNS = [
    "file_name",
    "chain_key",
    "chain_length",
    "effect_order",
    "effect_presence",
    "normalized_parameter_vector",
    "raw_parameter_dict",
    "source_audio_id",
    "random_seed",
]


@dataclass
class RenderRecord:
    """Uma linha do sidecar: o que foi renderizado e com quais parametros."""

    file_name: str
    chain_key: str
    chain_length: int
    effect_order: List[str]
    effect_presence: Dict[str, int]
    normalized_parameter_vector: List[float]
    raw_parameter_dict: Dict[str, Dict[str, float]]
    source_audio_id: str
    random_seed: int

    def as_row(self) -> Dict[str, object]:
        return {
            "file_name": self.file_name,
            "chain_key": self.chain_key,
            "chain_length": self.chain_length,
            "effect_order": json.dumps(self.effect_order),
            "effect_presence": json.dumps(self.effect_presence, sort_keys=True),
            "normalized_parameter_vector": json.dumps(self.normalized_parameter_vector),
            "raw_parameter_dict": json.dumps(self.raw_parameter_dict, sort_keys=True),
            "source_audio_id": self.source_audio_id,
            "random_seed": self.random_seed,
        }


def write_metadata_csv(path: Path, records: Sequence[RenderRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for item in records:
            writer.writerow(item.as_row())


def append_metadata_csv(root: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Funde `rows` no sidecar de `root`, substituindo linhas de mesmo `file_name`."""
    path = root / METADATA_FILENAME
    frame = pd.DataFrame(list(rows))
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old[~old["file_name"].isin(frame["file_name"])], frame], ignore_index=True)
    frame.to_csv(path, index=False)


def read_metadata(dataset_root: Path) -> pd.DataFrame:
    metadata_path = Path(dataset_root) / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata sidecar: {metadata_path}")
    return ensure_chain_key(pd.read_csv(metadata_path))


def ensure_chain_key(metadata: pd.DataFrame) -> pd.DataFrame:
    """Deriva `chain_key` de `effect_order` em sidecars antigos que nao a tinham."""
    if "chain_key" in metadata.columns:
        return metadata

    metadata = metadata.copy()
    metadata["chain_key"] = metadata["effect_order"].map(
        lambda value: chain_key(json.loads(value))
    )
    return metadata


def chain_folder(dataset_root: Path, chain_key_value: str) -> Path:
    folder = Path(dataset_root) / chain_key_value
    if not folder.exists():
        raise FileNotFoundError(f"Missing chain folder: {folder}")
    return folder


def list_chain_keys(dataset_root: Path) -> List[str]:
    return sorted(read_metadata(Path(dataset_root))["chain_key"].unique().tolist())


def target_lookup(metadata: pd.DataFrame, chain_key_value: str) -> Dict[str, List[float]]:
    """`file_name` -> vetor normalizado, para a cadeia pedida."""
    subset = metadata.loc[metadata["chain_key"] == chain_key_value, ["file_name", "normalized_parameter_vector"]]
    vectors = subset["normalized_parameter_vector"].map(json.loads)
    return dict(zip(subset["file_name"], vectors))


def validate_sidecar_integrity(dataset_root: str | Path) -> Dict[str, int]:
    """Confere que sidecar e audio batem, e que os valores estao em [0,1].

    Roda no inicio de todo treino; e a checagem de consistencia de facto do
    dataset, ja que o projeto nao tem suite de testes.
    """
    root = Path(dataset_root).resolve()
    metadata = read_metadata(root)

    missing_cols = sorted(set(METADATA_COLUMNS) - set(metadata.columns))
    if missing_cols:
        raise RuntimeError(f"Missing required metadata columns: {missing_cols}")

    counts: Dict[str, int] = {}
    for chain_key_value in sorted(metadata["chain_key"].unique().tolist()):
        folder = chain_folder(root, chain_key_value)
        wavs = sorted(path.name for path in folder.glob("*.wav") if path.is_file())
        rows = metadata[metadata["chain_key"] == chain_key_value]

        if len(wavs) != len(rows):
            raise RuntimeError(
                f"Chain {chain_key_value}: wav/metadata mismatch wav={len(wavs)} rows={len(rows)}"
            )

        row_names = set(rows["file_name"])
        missing_rows = [name for name in wavs if name not in row_names]
        if missing_rows:
            raise RuntimeError(
                f"Chain {chain_key_value}: metadata missing for wav {missing_rows[0]}"
            )

        vectors = rows["normalized_parameter_vector"].map(json.loads)
        out_of_range = [
            file_name
            for file_name, vector in zip(rows["file_name"], vectors)
            if not all(0.0 <= float(value) <= 1.0 for value in vector)
        ]
        if out_of_range:
            raise RuntimeError(
                f"Chain {chain_key_value}: normalized value outside [0,1] in {out_of_range[0]}"
            )

        counts[chain_key_value] = len(wavs)

    return counts
