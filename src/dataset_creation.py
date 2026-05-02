"""Create fixed-order multi-effect chain datasets with explicit parameter metadata.

This script keeps the binary effect suffix in generated filenames (e.g. __101)
while storing full parameter targets in a metadata CSV sidecar.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyloudnorm as pyln
from pedalboard import Chorus, Distortion, Pedalboard, Reverb, load_plugin
from pedalboard.io import AudioFile


DEFAULT_LOUDNESS_LEVEL = -26.0
CANONICAL_EFFECT_ORDER = ["overdrive", "chorus", "reverb"]

EFFECT_CHAINS = {
    1: ["overdrive"],
    2: ["overdrive", "reverb"],
    3: ["overdrive", "chorus", "reverb"],
}

EFFECT_PARAMETER_RANGES = {
    "overdrive": [{"name": "gain", "min": 0.20, "max": 1.00}],
    "chorus": [{"name": "mix", "min": 0.20, "max": 0.50}],
    "reverb": [{"name": "room_size", "min": 0.20, "max": 0.70}],
}


@dataclass
class ProcessedRecordMetadata:
    file_name: str
    chain_length: int
    effect_order: List[str]
    effect_presence: Dict[str, int]
    normalized_parameter_vector: List[float]
    raw_parameter_dict: Dict[str, Dict[str, float]]
    source_audio_id: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified fixed-order chain parameter dataset.")
    parser.add_argument("--input-dir", required=True, help="Directory containing clean source wav files.")
    parser.add_argument("--output-dir", required=True, help="Output root for rendered chain wav files.")
    parser.add_argument(
        "--overdrive-plugin",
        required=True,
        help="Path to The Klone.vst3 (or equivalent overdrive plugin).",
    )
    parser.add_argument("--samples-per-file", type=int, default=8, help="Renders per source file per chain length.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic generation.")
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Parameter sampling strategy in normalized [0,1] space.",
    )
    parser.add_argument(
        "--allow-overdrive-fallback",
        action="store_true",
        help="If Klone VST3 fails to load, use built-in Distortion for pilot runs.",
    )
    return parser.parse_args()


def normalize_loudness(audio: np.ndarray, sr: int, loudness_level: float = DEFAULT_LOUDNESS_LEVEL) -> np.ndarray:
    flat = np.reshape(audio, np.shape(audio)[1])
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(flat)
    normalized = pyln.normalize.loudness(flat, loudness, loudness_level)
    return np.reshape(normalized, (1, np.shape(audio)[1]))


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
    return {name: int(name in selected) for name in CANONICAL_EFFECT_ORDER}


def binary_suffix(chain_effects: Iterable[str]) -> str:
    selected = set(chain_effects)
    bits = ["1" if effect in selected else "0" for effect in CANONICAL_EFFECT_ORDER]
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


def convert_normalized_to_raw(effect: str, norm_values: List[float]) -> Dict[str, float]:
    raw_values: Dict[str, float] = {}
    for value, param in zip(norm_values, EFFECT_PARAMETER_RANGES[effect]):
        raw = param["min"] + (param["max"] - param["min"]) * float(value)
        raw_values[param["name"]] = float(raw)
    return raw_values


def generate_parameter_matrix(
    rng: np.random.Generator,
    amount_of_samples: int,
    amount_of_parameters: int,
    strategy: str,
) -> np.ndarray:
    """Sample a matrix of normalized parameter vectors in [0,1] space. Each row corresponds to a parameter vector for one chain instance (sample)."""
    
    if strategy == "random":
        return rng.random((amount_of_samples, amount_of_parameters), dtype=np.float64)

    # If not random, use stratified sampling: independently permute each dimension of a regular grid
    # Maybe throw this away? 
    samples = np.zeros((amount_of_samples, amount_of_parameters), dtype=np.float64)
    for dim in range(amount_of_parameters):
        perm = rng.permutation(amount_of_samples)
        samples[:, dim] = (perm + rng.random(amount_of_samples)) / amount_of_samples
    return samples


def build_chain(
    overdrive_plugin,
    chain_effects: List[str],
    raw_param_dict: Dict[str, Dict[str, float]],
):
    plugins = []
    for effect in chain_effects:
        if effect == "overdrive":
            gain = raw_param_dict["overdrive"]["gain"]
            if hasattr(overdrive_plugin, "gain"):
                overdrive_plugin.gain = gain
                plugins.append(overdrive_plugin)
            else:
                drive_db = float(5.0 + (gain * 35.0))
                plugins.append(Distortion(drive_db=drive_db))
        elif effect == "chorus":
            plugins.append(Chorus(mix=raw_param_dict["chorus"]["mix"]))
        elif effect == "reverb":
            plugins.append(
                Reverb(
                    wet_level=0.5,
                    dry_level=0.5,
                    room_size=raw_param_dict["reverb"]["room_size"],
                )
            )
        else:
            raise ValueError(f"Unsupported effect in chain: {effect}")
    return Pedalboard(plugins)


def make_record(
    output_name: str,
    chain_length: int,
    chain_effects: List[str],
    norm_vector: List[float],
    raw_dict: Dict[str, Dict[str, float]],
    source_audio_id: str,
    render_seed: int,
) -> ProcessedRecordMetadata:
    return ProcessedRecordMetadata(
        file_name=output_name,
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
                    "chain_length": item.chain_length,
                    "effect_order": json.dumps(item.effect_order),
                    "effect_presence": json.dumps(item.effect_presence, sort_keys=True),
                    "normalized_parameter_vector": json.dumps(item.normalized_parameter_vector),
                    "raw_parameter_dict": json.dumps(item.raw_parameter_dict, sort_keys=True),
                    "source_audio_id": item.source_audio_id,
                    "random_seed": item.random_seed,
                }
            )


def create_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    
    overdrive_plugin = None
    try:
        overdrive_plugin = load_plugin(str(Path(args.overdrive_plugin).resolve()))
    except Exception as exc:
        if not args.allow_overdrive_fallback:
            raise RuntimeError(f"Failed to load overdrive plugin at {args.overdrive_plugin}: {exc}")
        print(f"Warning: failed to load overdrive plugin ({exc}). Using Distortion fallback.")
        overdrive_plugin = "builtin_distortion"
        
    global_rng = np.random.default_rng(args.seed)

    records: List[ProcessedRecordMetadata] = []

    clean_audio_files = list(iter_wav_files(input_dir))
    if not clean_audio_files:
        raise RuntimeError(f"No wav files found under: {input_dir}")

    for file in clean_audio_files:
        clean_audio, sr = load_audio_file(file)
        normalized_clean_audio = normalize_loudness(clean_audio, sr)
        clean_audio_id = str(file.relative_to(input_dir))
        file_prefix = clean_audio_id_to_filename_prefix(clean_audio_id)

        for chain_length, chain_effects in EFFECT_CHAINS.items():
            total_amount_of_parameters = sum(len(EFFECT_PARAMETER_RANGES[fx]) for fx in chain_effects)
            paramter_matrix = generate_parameter_matrix(
                rng=global_rng,
                amount_of_samples=args.samples_per_file,
                amount_of_parameters=total_amount_of_parameters,
                strategy=args.sampling_strategy,
            )

            for sample_index in range(args.samples_per_file):
                sample_parameters = paramter_matrix[sample_index]
                parameter_list = [float(v) for v in sample_parameters.tolist()]

                effect_parameter_dict: Dict[str, Dict[str, float]] = {}
                offset = 0
                for effect in chain_effects:
                    amount_of_parameters = len(EFFECT_PARAMETER_RANGES[effect])
                    effect_parameter_dict[effect] = convert_normalized_to_raw(effect, parameter_list[offset : offset + amount_of_parameters])
                    offset += amount_of_parameters

                board = build_chain(overdrive_plugin, chain_effects, effect_parameter_dict)
                processed = board(normalized_clean_audio, sr)
                processed = normalize_loudness(processed, sr)

                bits = binary_suffix(chain_effects)
                output_name = f"{file_prefix}__{bits}__s{sample_index:04d}.wav"
                chain_dir = output_dir / f"L{chain_length}"
                chain_dir.mkdir(parents=True, exist_ok=True)
                output_path = chain_dir / output_name

                export_audio(processed, sr, output_path)

                render_seed = int(global_rng.integers(0, np.iinfo(np.int32).max))
                records.append(
                    make_record(
                        output_name=output_name,
                        chain_length=chain_length,
                        chain_effects=chain_effects,
                        norm_vector=parameter_list,
                        raw_dict=effect_parameter_dict,
                        source_audio_id=clean_audio_id,
                        render_seed=render_seed,
                    )
                )

    write_metadata_csv(metadata_path, records)
    print(f"Rendered {len(records)} files")
    print(f"Audio root: {output_dir}")
    print(f"Metadata:  {metadata_path}")


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args)
