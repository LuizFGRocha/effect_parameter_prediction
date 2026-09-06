"""Renderizacao do conjunto de avaliacao pareado.

Os mesmos segmentos de audio e os mesmos vetores normalizados sao renderizados
por varios bracos (implementacoes). O vetor normalizado e o ground truth comum a
todos, e e o que torna a comparacao entre bracos legitima.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from gefx.audio import (
    export_audio,
    iter_wav_files,
    load_audio_file,
    normalize_loudness,
    select_random_segment,
)
from gefx.data.metadata import append_metadata_csv
from gefx.effects.catalog import EFFECT_PARAMETER_RANGES, effect_predictable_params
from gefx.effects.parameters import effect_presence, raw_params_for_effect
from gefx.effects.pedalboard_backend import build_chain
from gefx.effects.registry import REFERENCE_ARM, REGISTRY, all_arms
from gefx.effects.vst_adapter import load_arm, process_mono, set_parameter


@dataclass
class RenderOptions:
    input_dir: str = "datasets/unprocessed_samples"
    output_root: str = "datasets/cross_impl"
    n: int = 300
    segment_seconds: float = 2.0
    seed: int = 7
    arms: Optional[Sequence[str]] = None
    effects: Optional[Sequence[str]] = None


def _apply_mapping(plugin, spec: dict, raw: dict, norm_row: np.ndarray, names: List[str]) -> None:
    """Escreve no plugin os parametros mapeados a partir dos valores de referencia."""
    for ref_name, (plugin_param, convert, exact) in spec["map"].items():
        if ref_name == "_mix_dry":  # metade seca do crossfade
            from gefx.effects.registry import to_db

            set_parameter(plugin, plugin_param, to_db(1.0 - raw["mix"]))
            continue
        # Invariante: mapeamento exato consome o valor cru de referencia (mesma
        # unidade fisica); mapeamento nao-exato consome o normalizado.
        source = raw[ref_name] if exact else float(norm_row[names.index(ref_name)])
        set_parameter(plugin, plugin_param, float(convert(source)))


def render(options: RenderOptions) -> None:
    rng = np.random.default_rng(options.seed)
    sources = list(iter_wav_files(Path(options.input_dir).resolve()))
    if not sources:
        raise SystemExit(f"Nenhum wav em {options.input_dir}")

    picks = rng.choice(len(sources), size=options.n, replace=options.n > len(sources))
    arms = list(options.arms) if options.arms else all_arms()
    effects = list(options.effects) if options.effects else sorted(EFFECT_PARAMETER_RANGES)

    for effect in effects:
        predictable = effect_predictable_params(effect)
        names = [param["name"] for param in predictable]
        # O MESMO vetor normalizado e o ground truth de todos os bracos.
        norm = rng.random((options.n, len(predictable)))

        for arm in arms:
            if arm != REFERENCE_ARM and arm not in REGISTRY.get(effect, {}):
                continue

            spec = None if arm == REFERENCE_ARM else REGISTRY[effect][arm]
            plugin, stereo = (None, False) if spec is None else load_arm(spec)
            out_dir = Path(options.output_root) / arm / effect
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            print(f"  {effect:<16} braco={arm:<12}", end="", flush=True)

            for index in range(options.n):
                source = sources[picks[index]]
                audio, sr = load_audio_file(source)
                segment_rng = np.random.default_rng(options.seed * 1000003 + index)
                segment = select_random_segment(
                    audio, sr, segment_rng, options.segment_seconds, 0.5, 0.5
                )
                segment = normalize_loudness(segment, sr)

                raw = raw_params_for_effect(effect, list(norm[index]))
                if plugin is None:
                    out = build_chain([effect], {effect: raw})(segment, sr)
                else:
                    _apply_mapping(plugin, spec, raw, norm[index], names)
                    out = process_mono(plugin, stereo, segment, sr)

                out = normalize_loudness(out, sr)
                name = f"{source.stem}__{effect}__{index:05d}.wav"
                export_audio(out, sr, out_dir / name)
                rows.append(
                    {
                        "file_name": name,
                        "chain_key": effect,
                        "chain_length": 1,
                        "effect_order": json.dumps([effect]),
                        "effect_presence": json.dumps(effect_presence([effect]), sort_keys=True),
                        "normalized_parameter_vector": json.dumps(
                            [float(value) for value in norm[index]]
                        ),
                        "raw_parameter_dict": json.dumps({effect: raw}, sort_keys=True),
                        "source_audio_id": source.name,
                        "random_seed": int(options.seed),
                    }
                )

            print(f" -> {len(rows)} renders")
            append_metadata_csv(Path(options.output_root) / arm, rows)
