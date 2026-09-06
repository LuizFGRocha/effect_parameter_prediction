"""Geracao do dataset: renderiza o audio limpo pelas cadeias de efeito.

Por arquivo-fonte: escolhe um segmento aleatorio, normaliza o loudness, renderiza
com parametros normalizados amostrados ao acaso e normaliza de novo. O
paralelismo e por arquivo-fonte; o determinismo vem de uma `SeedSequence` gerada
por arquivo, entao o resultado independe da ordem em que os processos terminam.

A ordem dos saques do RNG dentro de `render_source_file` faz parte do contrato:
alterar a sequencia muda o dataset gerado com o mesmo seed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from gefx.audio import (
    export_audio,
    iter_wav_files,
    load_audio_file,
    normalize_loudness,
    select_random_segment,
)
from gefx.config import DatasetConfig
from gefx.data.metadata import RenderRecord, write_metadata_csv
from gefx.effects.catalog import EFFECT_CHAINS, chain_key, chain_output_dim
from gefx.effects.parameters import (
    binary_suffix,
    effect_presence,
    sample_normalized_matrix,
    split_vector_by_effect,
)
from gefx.effects.pedalboard_backend import build_chain


def source_id_to_prefix(source_audio_id: str) -> str:
    sanitized = source_audio_id.replace("/", "__").replace("\\", "__")
    if sanitized.lower().endswith(".wav"):
        sanitized = sanitized[:-4]
    return sanitized


def render_source_file(
    file: Path,
    input_dir: Path,
    output_dir: Path,
    seed_sequence: np.random.SeedSequence,
    config: DatasetConfig,
) -> List[RenderRecord]:
    """Renderiza um arquivo-fonte por todas as cadeias. Roda em processo separado."""
    records: List[RenderRecord] = []
    file_rng = np.random.default_rng(seed_sequence)

    clean_audio, sr = load_audio_file(file)
    clean_audio_id = str(file.relative_to(input_dir))
    file_prefix = source_id_to_prefix(clean_audio_id)

    for chain_effects in EFFECT_CHAINS:
        chain_key_value = chain_key(chain_effects)
        parameter_matrix = sample_normalized_matrix(
            rng=file_rng,
            amount_of_samples=config.samples_per_file,
            amount_of_parameters=chain_output_dim(chain_key_value),
        )
        chain_dir = output_dir / chain_key_value
        chain_dir.mkdir(parents=True, exist_ok=True)
        bits = binary_suffix(chain_effects)

        for sample_index in range(config.samples_per_file):
            norm_vector = [float(value) for value in parameter_matrix[sample_index].tolist()]
            raw_by_effect = split_vector_by_effect(chain_effects, norm_vector)

            if config.use_full_audio:
                segment = clean_audio
            else:
                segment = select_random_segment(
                    clean_audio,
                    sr,
                    file_rng,
                    segment_seconds=config.segment_seconds,
                    ignore_start_seconds=config.ignore_start_seconds,
                    ignore_end_seconds=config.ignore_end_seconds,
                )

            processed = build_chain(chain_effects, raw_by_effect)(normalize_loudness(segment, sr), sr)
            processed = normalize_loudness(processed, sr)

            output_name = f"{file_prefix}__{bits}__s{sample_index:04d}.wav"
            export_audio(processed, sr, chain_dir / output_name)

            render_seed = int(file_rng.integers(0, np.iinfo(np.int32).max))
            records.append(
                RenderRecord(
                    file_name=output_name,
                    chain_key=chain_key_value,
                    chain_length=len(chain_effects),
                    effect_order=list(chain_effects),
                    effect_presence=effect_presence(chain_effects),
                    normalized_parameter_vector=norm_vector,
                    raw_parameter_dict=raw_by_effect,
                    source_audio_id=clean_audio_id,
                    random_seed=render_seed,
                )
            )

    return records


def generate_dataset(config: DatasetConfig) -> Path:
    """Renderiza o dataset inteiro e grava o sidecar. Devolve o caminho do sidecar."""
    input_dir = Path(config.input_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_audio_files = list(iter_wav_files(input_dir))
    if not clean_audio_files:
        raise RuntimeError(f"No wav files found under: {input_dir}")

    child_seeds = np.random.SeedSequence(config.seed).spawn(len(clean_audio_files))

    records: List[RenderRecord] = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(render_source_file, file, input_dir, output_dir, seed, config)
            for file, seed in zip(clean_audio_files, child_seeds)
        ]
        for future in as_completed(futures):
            records.extend(future.result())

    metadata_path = output_dir / "metadata.csv"
    write_metadata_csv(metadata_path, records)
    print(f"Renderizados {len(records)} arquivos")
    print(f"Audio:    {output_dir}")
    print(f"Metadata: {metadata_path}")
    return metadata_path
