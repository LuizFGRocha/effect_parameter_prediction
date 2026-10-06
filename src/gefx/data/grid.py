"""O dataset do POC II: a grade de niveis, a particao e o sidecar de cada arm.

- A grade: os niveis de drive. O nivel e um indice (no sidecar, `config_index ==
  drive_level`); o valor de knob depende do arm e sai do arquivo de niveis. Nao
  ha tone: a EQ propria de cada pedal, que em alguns muda com o ganho, impedia
  um fator de tone igual em todos.
- A particao: treino / catalogo / consulta, disjuntas por conteudo, para a busca
  nao acertar reconhecendo o que foi tocado. A validacao sai do treino, na
  leitura (`validation_recordings`), com a mesma forma do teste: uma consulta e
  um catalogo de gravacoes disjuntas.
- O sidecar: uma linha por render, com os fatores como colunas. Um
  `metadata.csv` por arm, em `<raiz>/<arm>/`, com os wavs em
  `<raiz>/<arm>/distortion/`. `file_name` e identico entre arms, e
  `validate_pairing` cobra isso.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.data.metadata import METADATA_FILENAME

SPLITS: Tuple[str, ...] = ("train", "catalog", "query")
#: As particoes de validacao, tiradas do treino. Os nomes espelham as do teste.
VALIDATION_SPLITS: Tuple[str, ...] = ("val_query", "val_catalog")
#: Gravacoes em cada lado da validacao: 2 x 20 das 240 de treino.
VALIDATION_RECORDINGS = 20
VALIDATION_SEED = 20260930
#: As duas buscas: (consulta, catalogo). O desenvolvimento so le a validacao; o
#: teste e lido uma vez, no fim, com os modelos finais (`gefx disent evaluate`).
EVAL_SPLITS: Mapping[str, Tuple[str, str]] = {
    "validacao": ("val_query", "val_catalog"),
    "teste": ("query", "catalog"),
}


def split_file(stem: str, split: str, suffix: str) -> str:
    """O nome de um resultado de uma busca: `<stem>_<validacao|teste>.<suffix>`.

    Quem grava e quem le passam por aqui, para os nomes nao divergirem.
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"particao desconhecida: {split!r}. Ha {list(EVAL_SPLITS)}")
    return f"{stem}_{split}.{suffix}"


DEFAULT_SPLIT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "catalog": 0.2, "query": 0.2}


def level_key(level: int) -> str:
    """O rotulo de um nivel de drive (`config_key` no sidecar): `d<nivel>`."""
    return f"d{level}"


def render_name(content_id: str, level: int) -> str:
    """Nome do wav, identico entre arms: e o que torna o pareamento verificavel."""
    return f"{content_id}__{level_key(level)}__{level:05d}.wav"


# --- particao por conteudo ----------------------------------------------------
def split_contents(
    content_ids: Sequence[str],
    seed: int = 20260906,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
) -> Dict[str, List[str]]:
    """Divide os conteudos em treino/catalogo/consulta, disjuntos.

    Embaralha antes porque os nomes vem agrupados por guitarra, captador e tecnica.
    """
    if set(fractions) != set(SPLITS):
        raise ValueError(f"fracoes devem cobrir exatamente {SPLITS}")
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"fracoes devem somar 1.0, somam {total}")

    unique = sorted(set(content_ids))
    if len(unique) != len(content_ids):
        raise ValueError("ha content_id repetido")
    if len(unique) < len(SPLITS):
        raise ValueError(f"sao precisos ao menos {len(SPLITS)} conteudos")

    shuffled = list(np.random.default_rng(seed).permutation(unique))
    total_items = len(shuffled)
    n_train = max(1, int(round(fractions["train"] * total_items)))
    n_catalog = max(1, int(round(fractions["catalog"] * total_items)))
    # Garante ao menos um conteudo para a consulta, que leva o resto.
    while n_train + n_catalog > total_items - 1:
        if n_train >= n_catalog:
            n_train -= 1
        else:
            n_catalog -= 1

    out = {
        "train": shuffled[:n_train],
        "catalog": shuffled[n_train : n_train + n_catalog],
        "query": shuffled[n_train + n_catalog :],
    }
    return {name: [str(item) for item in items] for name, items in out.items()}


def validation_recordings(
    recordings: Sequence[str],
    per_side: int = VALIDATION_RECORDINGS,
    seed: int = VALIDATION_SEED,
) -> Dict[str, List[str]]:
    """Separa, das gravacoes de treino, as da consulta e as do catalogo de validacao.

    Por gravacao, como o teste: os trechos de uma execucao ficam do mesmo lado. O
    sorteio depende so da lista e da semente, entao a validacao e a mesma em toda
    execucao sobre o mesmo dataset e os mesmos arms.
    """
    unique = sorted(set(recordings))
    if per_side < 1:
        raise ValueError("a validacao precisa de ao menos uma gravacao de cada lado")
    if len(unique) <= 2 * per_side:
        raise ValueError(
            f"{len(unique)} gravacoes de treino nao comportam 2 x {per_side} de "
            "validacao e ainda deixar treino"
        )
    shuffled = [str(item) for item in np.random.default_rng(seed).permutation(unique)]
    return {
        "val_query": shuffled[:per_side],
        "val_catalog": shuffled[per_side : 2 * per_side],
        "train": shuffled[2 * per_side :],
    }


# --- sidecar ------------------------------------------------------------------
SIDECAR_FILENAME = METADATA_FILENAME
EFFECT_FOLDER = "distortion"


@dataclass(frozen=True)
class GridRecord:
    """Uma linha do sidecar."""

    file_name: str
    arm: str
    stratum: str
    content_id: str
    source_audio_id: str
    segment_start: int
    config_index: int
    config_key: str
    drive_level: int
    drive_knob: float
    drive_db_equivalente: float
    split: str

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


SIDECAR_COLUMNS: List[str] = [item.name for item in fields(GridRecord)]


def write_sidecar(path: Path, records: Sequence[GridRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIDECAR_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def arm_folder(root: Path, arm: str) -> Path:
    """A pasta dos wavs (e do cache de features) de um arm."""
    return Path(root) / arm / EFFECT_FOLDER


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


def split_frames(
    root: Path,
    arms: Optional[Sequence[str]] = None,
    validation: int = VALIDATION_RECORDINGS,
) -> Dict[str, pd.DataFrame]:
    """As particoes por gravacao, ja restritas aos arms pedidos.

    `train`, `catalog` e `query` vem do sidecar; `val_query` e `val_catalog` saem
    do treino, `validation` gravacoes de cada lado (`grid.validation_recordings`),
    e o que sobra e o `train`. A validacao nunca toca o teste.
    """
    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    frames = {
        name: data[data["split"] == name].reset_index(drop=True)
        for name in ("train", "catalog", "query")
    }
    if not frames["train"].empty:
        train = frames["train"]
        recording = train["source_audio_id"].astype(str)
        sides = validation_recordings(recording, validation)
        for name in ("train", *VALIDATION_SPLITS):
            frames[name] = train[recording.isin(sides[name])].reset_index(drop=True)
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(f"particoes vazias: {empty}")
    return frames


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
    factors = ["content_id", "config_index", "drive_level", "split"]

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
