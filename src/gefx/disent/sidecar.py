"""Sidecar do dataset do POC II: uma linha por render, com os fatores como colunas.

Um `metadata.csv` por arm, em `<raiz>/<arm>/`. `file_name` e identico entre
arms, e `validate_pairing` cobra isso.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

SIDECAR_FILENAME = "metadata.csv"
EFFECT_FOLDER = "distortion"


@dataclass(frozen=True)
class DisentRecord:
    """Uma linha do sidecar."""

    file_name: str
    arm: str
    content_id: str
    source_audio_id: str
    segment_start: int
    config_index: int
    config_key: str
    drive_level: int
    tone_level: int
    drive_knob: float
    tone_cutoff_hz: float
    drive_db_equivalente: float
    split: str

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


SIDECAR_COLUMNS: List[str] = [item.name for item in fields(DisentRecord)]


def write_sidecar(path: Path, records: Sequence[DisentRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIDECAR_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def read_sidecar(arm_dir: Path) -> pd.DataFrame:
    path = Path(arm_dir) / SIDECAR_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"sidecar ausente: {path}")
    return pd.read_csv(path)


def read_dataset(root: Path) -> pd.DataFrame:
    """Concatena os sidecars de todos os arms presentes na raiz."""
    root = Path(root)
    frames = [read_sidecar(arm_dir) for arm_dir in sorted(root.iterdir()) if arm_dir.is_dir()]
    if not frames:
        raise FileNotFoundError(f"nenhum arm em {root}")
    return pd.concat(frames, ignore_index=True)


def arm_dirs(root: Path) -> List[Path]:
    return [
        path
        for path in sorted(Path(root).iterdir())
        if path.is_dir() and (path / SIDECAR_FILENAME).exists()
    ]


def validate_pairing(root: Path) -> Dict[str, int]:
    """Checa que a grade e cruzada: todos os arms com os mesmos `file_name` e fatores."""
    dirs = arm_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"nenhum arm com sidecar em {root}")

    reference_dir = dirs[0]
    reference = read_sidecar(reference_dir).set_index("file_name").sort_index()
    factors = ["content_id", "config_index", "drive_level", "tone_level", "split"]

    for arm_dir in dirs[1:]:
        current = read_sidecar(arm_dir).set_index("file_name").sort_index()
        missing = set(reference.index) - set(current.index)
        extra = set(current.index) - set(reference.index)
        if missing or extra:
            raise ValueError(
                f"{arm_dir.name} nao esta pareado com {reference_dir.name}: "
                f"{len(missing)} faltando, {len(extra)} sobrando "
                f"(ex.: {sorted(missing | extra)[:3]})"
            )
        for column in factors:
            if not reference[column].equals(current[column]):
                raise ValueError(
                    f"{arm_dir.name} difere de {reference_dir.name} na coluna {column!r}"
                )

    for arm_dir in dirs:
        frame = read_sidecar(arm_dir)
        folder = arm_dir / EFFECT_FOLDER
        absent = [name for name in frame["file_name"] if not (folder / name).exists()]
        if absent:
            raise ValueError(f"{arm_dir.name}: {len(absent)} wavs no sidecar sem arquivo "
                             f"(ex.: {absent[:3]})")

    return {
        "arms": len(dirs),
        "rows_per_arm": len(reference),
        "contents": int(reference["content_id"].nunique()),
        "configs": int(reference["config_index"].nunique()),
    }
