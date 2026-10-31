"""Renderizador da grade completa conteudo x nivel de drive x arm.

Cadeia de cada render, a do POC I:

    segmento -> normalize_loudness -> nao-linearidade do arm -> normalize_loudness

Cada gravacao da `segments_per_file` conteudos: trechos consecutivos, sem
sobreposicao, a partir de um deslocamento sorteado. O sorteio depende so de
`seed` e do indice da gravacao, entao o mesmo `content_id` e o mesmo trecho em
toda a grade. A particao e por gravacao: os trechos de uma execucao ficam todos
no mesmo split, senao a busca poderia acertar reconhecendo a execucao. Os knobs
de drive vem do arquivo de niveis; ele e o roster sao copiados para a raiz do dataset.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from gefx.audio import export_audio, load_audio_file, normalize_loudness
from gefx.data.grid import (
    DATASET_ROOT,
    SIDECAR_FILENAME,
    GridRecord,
    arm_folder,
    level_key,
    render_name,
    split_contents,
    validate_pairing,
    write_sidecar,
)
from gefx.effects.drive_arms import DEFAULT_LEVELS, DEFAULT_ROSTER, Arm, LoadedArm, load_roster


@dataclass(frozen=True)
class ContentItem:
    """Um conteudo: um trecho de uma gravacao (`<gravacao>_s<k>`)."""

    content_id: str
    source_path: str
    segment_start: int
    split: str


@dataclass
class RenderOptions:
    input_dir: Path = Path("datasets/unprocessed_samples")
    output_root: Path = DATASET_ROOT
    roster: Path = DEFAULT_ROSTER
    levels: Path = DEFAULT_LEVELS
    n_contents: int = 400  # gravacoes; cada uma da `segments_per_file` conteudos
    segments_per_file: int = 5
    segment_seconds: float = 2.0
    seed: int = 20260906
    split_seed: int = 20260906
    arms: Optional[Sequence[str]] = None
    #: So estas particoes (os niveis intermediarios so precisam das consultas).
    splits: Optional[Sequence[str]] = None
    workers: int = 4


def content_items(options: RenderOptions) -> List[ContentItem]:
    """Escolhe as gravacoes e os trechos de cada uma, de forma deterministica."""
    paths = sorted(Path(options.input_dir).glob("*.wav"))
    if len(paths) < options.n_contents:
        raise ValueError(
            f"{options.input_dir} tem {len(paths)} gravacoes, menos que os "
            f"{options.n_contents} conteudos pedidos"
        )
    chosen = paths[: options.n_contents]
    splits = split_contents([path.stem for path in chosen], seed=options.split_seed)
    split_of = {content: name for name, items in splits.items() for content in items}

    items: List[ContentItem] = []
    for index, path in enumerate(chosen):
        audio, sr = load_audio_file(path)
        frames = int(round(options.segment_seconds * sr))
        margin = int(round(0.5 * sr))  # mesmo descarte de bordas do POC I
        slack = audio.shape[1] - 2 * margin - options.segments_per_file * frames
        if slack < 0:
            raise ValueError(f"{path} e curta demais para {options.segments_per_file} "
                             f"trechos de {options.segment_seconds}s")
        rng = np.random.default_rng(options.seed * 1000003 + index)
        first = margin + int(rng.integers(0, slack + 1))
        for k in range(options.segments_per_file):
            items.append(ContentItem(f"{path.stem}_s{k}", str(path), first + k * frames,
                                     split_of[path.stem]))
    return items


def _render_chunk(
    spec: Arm,
    items: Sequence[ContentItem],
    reference_levels: Sequence[float],
    output_root: str,
    segment_seconds: float,
) -> List[Dict[str, object]]:
    """Renderiza um lote de conteudos num arm, carregando o plugin uma vez. Roda no trabalhador."""
    loaded = LoadedArm(spec)
    folder = arm_folder(Path(output_root), spec.key)
    folder.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for item in items:
        audio, sr = load_audio_file(Path(item.source_path))
        frames = int(round(segment_seconds * sr))
        segment = normalize_loudness(
            audio[:, item.segment_start : item.segment_start + frames], sr
        )
        for level, drive in enumerate(spec.levels):
            wet = normalize_loudness(loaded.render(segment, sr, drive), sr)

            name = render_name(item.content_id, level)
            export_audio(wet, sr, folder / name)
            rows.append(
                GridRecord(
                    file_name=name,
                    arm=spec.key,
                    stratum=spec.stratum,
                    content_id=item.content_id,
                    source_audio_id=Path(item.source_path).name,
                    segment_start=item.segment_start,
                    config_index=level,
                    config_key=level_key(level),
                    drive_level=level,
                    drive_knob=float(drive),
                    # Pareado pelo Rnonlin: o nivel vale o que a referencia vale nele.
                    drive_db_equivalente=float(reference_levels[level]),
                    split=item.split,
                ).as_row()
            )
    return rows


def _chunks(items: Sequence[ContentItem], n: int) -> List[List[ContentItem]]:
    size = max(1, (len(items) + n - 1) // n)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def render(options: RenderOptions) -> Dict[str, int]:
    """Renderiza a grade inteira e escreve um sidecar por arm."""
    root = Path(options.output_root)
    roster = load_roster(options.roster, options.levels)
    wanted = list(options.arms) if options.arms else roster.keys()
    arms = [roster.arm(key) for key in wanted]

    items = content_items(options)
    if options.splits:
        items = [item for item in items if item.split in options.splits]
    levels = roster.drive_levels
    print(f"{len(items)} conteudos ({options.n_contents} gravacoes x "
          f"{options.segments_per_file} trechos) x {levels} niveis x {len(arms)} arms "
          f"= {len(items) * levels * len(arms)} renders")

    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(options.roster, root / "roster.yaml")
    shutil.copyfile(options.levels, root / "levels.yaml")
    chunks = _chunks(items, options.workers)
    totals: Dict[str, int] = {}
    for spec in arms:
        with ProcessPoolExecutor(max_workers=options.workers) as pool:
            futures = [
                pool.submit(
                    _render_chunk, spec, chunk, roster.reference_levels(),
                    str(root), options.segment_seconds,
                )
                for chunk in chunks
            ]
            rows = [row for future in futures for row in future.result()]

        rows.sort(key=lambda row: (row["content_id"], row["drive_level"]))
        write_sidecar(root / spec.key / SIDECAR_FILENAME, [GridRecord(**row) for row in rows])
        totals[spec.key] = len(rows)
        print(f"  {spec.key:16s} {len(rows)} renders")

    summary = validate_pairing(root)
    print(f"grade pareada: {summary}")
    return totals
