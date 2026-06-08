from __future__ import annotations

import argparse
import csv
import functools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyloudnorm as pyln
from pedalboard import Chorus, Distortion, Pedalboard, Reverb, Phaser, Delay, load_plugin
from pedalboard.io import AudioFile

from chain_definitions import (
    EFFECT_CHAINS,
    EFFECT_PARAMETER_RANGES,
    chain_key,
    effect_fixed_params,
    effect_predictable_params,
)


DEFAULT_LOUDNESS_LEVEL = -26.0

BUILT_IN_PLUGINS = {
    "distortion": Distortion,
    "chorus": Chorus,
    "vibrato": Chorus,
    "flanger": Chorus,
    "feedback_delay": Delay,
    "slapback_delay": Delay,
    "phaser": Phaser,
    "reverb": Reverb,
}

@dataclass
class ProcessedRecordMetadata:
    file_name: str
    chain_key: str
    chain_length: int
    effect_order: List[str]
    effect_presence: Dict[str, int]
    normalized_parameter_vector: List[float]
    raw_parameter_dict: Dict[str, Dict[str, float]]
    source_audio_id: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified fixed-order chain parameter dataset.")
    parser.add_argument("--input-dir", default="datasets/unprocessed_samples")
    parser.add_argument("--output-dir", required=True, help="Output root for rendered chain wav files.")
    parser.add_argument("--samples-per-file", type=int, default=56, help="Renders per source file per chain length.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic generation.")
    parser.add_argument(
        "--use-full-audio",
        action="store_true",
        help="If set, processes the entire audio file instead of a random segment.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="Length of the random segment to render (in seconds).",
    )
    parser.add_argument(
        "--ignore-start-seconds",
        type=float,
        default=0.5,
        help="Amount of audio to ignore at the start (in seconds).",
    )
    parser.add_argument(
        "--ignore-end-seconds",
        type=float,
        default=0.5,
        help="Amount of audio to ignore at the end (in seconds).",
    )

    return parser.parse_args()


@functools.lru_cache(maxsize=8)
def get_meter(sr: int) -> pyln.Meter:
    return pyln.Meter(sr)


def normalize_loudness(audio: np.ndarray, sr: int, loudness_level: float = DEFAULT_LOUDNESS_LEVEL) -> np.ndarray:
    flat = np.reshape(audio, np.shape(audio)[1])
    meter = get_meter(sr)
    loudness = meter.integrated_loudness(flat)
    normalized = pyln.normalize.loudness(flat, loudness, loudness_level)
    return np.reshape(normalized, (1, np.shape(audio)[1]))


def select_random_segment(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    segment_seconds: float,
    ignore_start_seconds: float,
    ignore_end_seconds: float,
) -> np.ndarray:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if ignore_start_seconds < 0 or ignore_end_seconds < 0:
        raise ValueError("ignore_start_seconds and ignore_end_seconds must be >= 0")

    total_frames = audio.shape[1]
    segment_frames = int(round(segment_seconds * sr))
    if segment_frames <= 0:
        raise ValueError("segment_seconds is too small for the current sample rate")

    min_start = int(round(ignore_start_seconds * sr))
    max_end = total_frames - int(round(ignore_end_seconds * sr))
    max_start = max_end - segment_frames
    if max_start < min_start:
        total_seconds = total_frames / sr
        raise ValueError(
            "Segment selection exceeds audio length. "
            f"segment_seconds={segment_seconds}, ignore_start_seconds={ignore_start_seconds}, "
            f"ignore_end_seconds={ignore_end_seconds}, audio_seconds={total_seconds:.3f}"
        )

    start = int(rng.integers(min_start, max_start + 1))
    end = start + segment_frames
    return audio[:, start:end]


def load_audio_file(file_path: Path) -> Tuple[np.ndarray, int]:
    with AudioFile(str(file_path), "r") as handle:
        audio = handle.read(handle.frames)
        sample_rate = handle.samplerate
    return audio, sample_rate


def export_audio(audio: np.ndarray, sr: int, path: Path) -> None:
    with AudioFile(str(path), "w", sr, audio.shape[0]) as handle:
        handle.write(audio)


def effect_presence(chain_effects: Iterable[str]) -> Dict[str, int]:
    selected = set(chain_effects)
    return {name: int(name in selected) for name in EFFECT_PARAMETER_RANGES.keys()}


def binary_suffix(chain_effects: Iterable[str]) -> str:
    selected = set(chain_effects)
    bits = ["1" if effect in selected else "0" for effect in EFFECT_PARAMETER_RANGES.keys()]
    return "".join(bits)


def clean_audio_id_to_filename_prefix(source_audio_id: str) -> str:
    sanitized = source_audio_id.replace("/", "__").replace("\\", "__")
    if sanitized.lower().endswith(".wav"):
        sanitized = sanitized[:-4]
    return sanitized


def iter_wav_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.wav")):
        if path.is_file():
            yield path


def convert_normalized_to_raw(params: List[Dict[str, float]], norm_values: List[float]) -> Dict[str, float]:
    raw_values: Dict[str, float] = {}
    for value, param in zip(norm_values, params):
        raw = param["min"] + (param["max"] - param["min"]) * float(value)
        raw_values[param["name"]] = float(raw)
    return raw_values


def generate_parameter_matrix(
    rng: np.random.Generator,
    amount_of_samples: int,
    amount_of_parameters: int,
) -> np.ndarray:
    return rng.random((amount_of_samples, amount_of_parameters), dtype=np.float64)


def build_chain(
    chain_effects: List[str],
    raw_param_dict: Dict[str, Dict[str, float]],
):
    plugins = []
    for effect in chain_effects:
        if effect in BUILT_IN_PLUGINS:
            plugin_class = BUILT_IN_PLUGINS[effect]
            params = raw_param_dict[effect]
            plugin = plugin_class(**params)
        else:
            raise ValueError(f"Effect '{effect}' not recognized in built-in plugins.")
        plugins.append(plugin)
        
    return Pedalboard(plugins)


def make_record(
    output_name: str,
    chain_key_value: str,
    chain_length: int,
    chain_effects: List[str],
    norm_vector: List[float],
    raw_dict: Dict[str, Dict[str, float]],
    source_audio_id: str,
    render_seed: int,
) -> ProcessedRecordMetadata:
    return ProcessedRecordMetadata(
        file_name=output_name,
        chain_key=chain_key_value,
        chain_length=chain_length,
        effect_order=chain_effects,
        effect_presence=effect_presence(chain_effects),
        normalized_parameter_vector=norm_vector,
        raw_parameter_dict=raw_dict,
        source_audio_id=source_audio_id,
        random_seed=render_seed,
    )


def write_metadata_csv(path: Path, records: List[ProcessedRecordMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    "file_name": item.file_name,
                    "chain_key": item.chain_key,
                    "chain_length": item.chain_length,
                    "effect_order": json.dumps(item.effect_order),
                    "effect_presence": json.dumps(item.effect_presence, sort_keys=True),
                    "normalized_parameter_vector": json.dumps(item.normalized_parameter_vector),
                    "raw_parameter_dict": json.dumps(item.raw_parameter_dict, sort_keys=True),
                    "source_audio_id": item.source_audio_id,
                    "random_seed": item.random_seed,
                }
            )


def process_file(
    file: Path,
    input_dir: Path,
    output_dir: Path,
    seed_sequence: np.random.SeedSequence,
    samples_per_file: int,
    use_full_audio: bool,
    segment_seconds: float,
    ignore_start_seconds: float,
    ignore_end_seconds: float,
) -> List[ProcessedRecordMetadata]:
    records: List[ProcessedRecordMetadata] = []
    file_rng = np.random.default_rng(seed_sequence)
    
    clean_audio, sr = load_audio_file(file)
    clean_audio_id = str(file.relative_to(input_dir))
    file_prefix = clean_audio_id_to_filename_prefix(clean_audio_id)

    for chain_effects in EFFECT_CHAINS:
        chain_key_value = chain_key(chain_effects)
        chain_length = len(chain_effects)
        total_amount_of_parameters = sum(len(effect_predictable_params(fx)) for fx in chain_effects)
        parameter_matrix = generate_parameter_matrix(
            rng=file_rng,
            amount_of_samples=samples_per_file,
            amount_of_parameters=total_amount_of_parameters,
        )

        for sample_index in range(samples_per_file):
            sample_parameters = parameter_matrix[sample_index]
            parameter_list = [float(v) for v in sample_parameters.tolist()]

            effect_parameter_dict: Dict[str, Dict[str, float]] = {}
            offset = 0
            for effect in chain_effects:
                predictable_params = effect_predictable_params(effect)
                amount_of_parameters = len(predictable_params)
                effect_parameter_dict[effect] = convert_normalized_to_raw(
                    predictable_params,
                    parameter_list[offset : offset + amount_of_parameters],
                )
                for fixed_param in effect_fixed_params(effect):
                    effect_parameter_dict[effect][fixed_param["name"]] = float(fixed_param["min"])
                offset += amount_of_parameters

            if use_full_audio:
                segment = clean_audio
            else:
                segment = select_random_segment(
                    clean_audio,
                    sr,
                    file_rng,
                    segment_seconds=segment_seconds,
                    ignore_start_seconds=ignore_start_seconds,
                    ignore_end_seconds=ignore_end_seconds,
                )
                
            normalized_segment = normalize_loudness(segment, sr)

            board = build_chain(chain_effects, effect_parameter_dict)
            processed = board(normalized_segment, sr)
            processed = normalize_loudness(processed, sr)

            bits = binary_suffix(chain_effects)
            output_name = f"{file_prefix}__{bits}__s{sample_index:04d}.wav"
            chain_dir = output_dir / chain_key_value
            chain_dir.mkdir(parents=True, exist_ok=True)
            output_path = chain_dir / output_name

            export_audio(processed, sr, output_path)

            render_seed = int(file_rng.integers(0, np.iinfo(np.int32).max))
            records.append(
                make_record(
                    output_name=output_name,
                    chain_key_value=chain_key_value,
                    chain_length=chain_length,
                    chain_effects=chain_effects,
                    norm_vector=parameter_list,
                    raw_dict=effect_parameter_dict,
                    source_audio_id=clean_audio_id,
                    render_seed=render_seed,
                )
            )
            
    return records


def generate_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_audio_files = list(iter_wav_files(input_dir))
    if not clean_audio_files:
        raise RuntimeError(f"No wav files found under: {input_dir}")

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(clean_audio_files))

    records: List[ProcessedRecordMetadata] = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_file,
                file,
                input_dir,
                output_dir,
                seed,
                args.samples_per_file,
                args.use_full_audio,
                args.segment_seconds,
                args.ignore_start_seconds,
                args.ignore_end_seconds,
            )
            for file, seed in zip(clean_audio_files, child_seeds)
        ]

        for future in as_completed(futures):
            records.extend(future.result())

    write_metadata_csv(metadata_path, records)
    print(f"Rendered {len(records)} files")
    print(f"Audio root: {output_dir}")
    print(f"Metadata:  {metadata_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)