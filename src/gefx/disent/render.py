"""Renderizador da grade completa conteudo x configuracao x arm.

Cadeia de cada render, a do POC I com o tone depois da nao-linearidade:

    segmento -> normalize_loudness -> nao-linearidade do arm -> tone -> normalize_loudness

O inicio do segmento de cada conteudo depende so de `seed` e do indice do
conteudo, entao o mesmo `content_id` e o mesmo trecho em toda a grade. Os knobs
de drive vem do roster, que e copiado para a raiz do dataset.
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from gefx.audio import export_audio, load_audio_file, normalize_loudness
from gefx.disent.arms import DEFAULT_ROSTER, Arm, LoadedArm, apply_tone, load_roster, tone_cutoff_hz
from gefx.disent.grid import Config, all_configs, render_name, split_contents
from gefx.disent.sidecar import (
    EFFECT_FOLDER,
    DisentRecord,
    SIDECAR_FILENAME,
    validate_pairing,
    write_sidecar,
)


@dataclass(frozen=True)
class ContentItem:
    """Um conteudo: uma execucao concreta, ja com o trecho escolhido."""

    content_id: str
    source_path: str
    segment_start: int
    split: str


@dataclass
class RenderOptions:
    input_dir: Path = Path("datasets/unprocessed_samples")
    output_root: Path = Path("datasets/disent")
    roster: Path = DEFAULT_ROSTER
    n_contents: int = 100
    segment_seconds: float = 2.0
    seed: int = 20260906
    split_seed: int = 20260906
    arms: Optional[Sequence[str]] = None
    workers: int = 4


def content_items(options: RenderOptions) -> List[ContentItem]:
    """Escolhe os conteudos e o trecho de cada um, de forma deterministica."""
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
        highest = audio.shape[1] - margin - frames
        if highest <= margin:
            raise ValueError(f"{path} e curta demais para um segmento de {options.segment_seconds}s")
        rng = np.random.default_rng(options.seed * 1000003 + index)
        start = int(rng.integers(margin, highest + 1))
        items.append(
            ContentItem(path.stem, str(path), start, split_of[path.stem])
        )
    return items


def _render_chunk(
    spec: Arm,
    items: Sequence[ContentItem],
    configs: Sequence[Config],
    reference_levels: Sequence[float],
    output_root: str,
    segment_seconds: float,
) -> List[Dict[str, object]]:
    """Renderiza um lote de conteudos num arm, carregando o plugin uma vez. Roda no trabalhador."""
    loaded = LoadedArm(spec)
    folder = Path(output_root) / spec.key / EFFECT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for item in items:
        audio, sr = load_audio_file(Path(item.source_path))
        frames = int(round(segment_seconds * sr))
        segment = normalize_loudness(
            audio[:, item.segment_start : item.segment_start + frames], sr
        )
        for index, config in enumerate(configs):
            drive = spec.levels[config.drive_level]
            cutoff = tone_cutoff_hz(config.tone_level)
            wet = loaded.render(segment, sr, drive)
            wet = apply_tone(wet, sr, cutoff)
            wet = normalize_loudness(wet, sr)

            name = render_name(item.content_id, config, index)
            export_audio(wet, sr, folder / name)
            rows.append(
                DisentRecord(
                    file_name=name,
                    arm=spec.key,
                    stratum=spec.stratum,
                    content_id=item.content_id,
                    source_audio_id=Path(item.source_path).name,
                    segment_start=item.segment_start,
                    config_index=index,
                    config_key=config.key,
                    drive_level=config.drive_level,
                    tone_level=config.tone_level,
                    drive_knob=float(drive),
                    tone_cutoff_hz=float(cutoff),
                    # Pareado de ouvido: o nivel vale o que a referencia vale nele.
                    drive_db_equivalente=float(reference_levels[config.drive_level]),
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
    roster = load_roster(options.roster)
    wanted = list(options.arms) if options.arms else roster.keys()
    arms = [roster.arm(key) for key in wanted]

    items = content_items(options)
    configs = all_configs(roster.drive_levels)
    print(f"{len(items)} conteudos x {len(configs)} configuracoes x {len(arms)} arms "
          f"= {len(items) * len(configs) * len(arms)} renders")

    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(options.roster, root / "roster.yaml")
    chunks = _chunks(items, options.workers)
    totals: Dict[str, int] = {}
    for spec in arms:
        with ProcessPoolExecutor(max_workers=options.workers) as pool:
            futures = [
                pool.submit(
                    _render_chunk, spec, chunk, configs, roster.reference_levels(),
                    str(root), options.segment_seconds,
                )
                for chunk in chunks
            ]
            rows = [row for future in futures for row in future.result()]

        rows.sort(key=lambda row: (row["content_id"], row["config_index"]))
        write_sidecar(root / spec.key / SIDECAR_FILENAME, [DisentRecord(**row) for row in rows])
        totals[spec.key] = len(rows)
        print(f"  {spec.key:16s} {len(rows)} renders")

    (root / "content_items.json").write_text(
        json.dumps([item.__dict__ for item in items], indent=2), encoding="utf-8"
    )

    summary = validate_pairing(root)
    print(f"grade pareada: {summary}")
    return totals
