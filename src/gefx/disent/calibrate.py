"""Calibracao do knob de drive de cada arm contra a referencia.

Implementacoes nao compartilham unidade de ganho. Para cada arm:

1. varre-se o knob e mede-se um descritor de quantidade de distorcao;
2. o arm so e aceito se o knob for monotono no descritor (rho de Spearman);
3. os extremos casam a faixa comum a todos os arms aceitos, e os niveis
   interiores seguem `level_mode` (ver `LEVEL_MODES`).

A unidade de leitura e o `drive_db` da tanh de referencia que produz o mesmo
descritor. Checagem do protocolo: em `lsp-tanh` a calibracao tem de devolver a
faixa da referencia.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gefx.audio import DEFAULT_LOUDNESS_LEVEL, load_audio_file, normalize_loudness
from gefx.disent.arms import (DRIVE_LEVELS, PERCEPTUAL_BIAS, REFERENCE_ARM, Arm,
                              LoadedArm, arm, arm_keys)

# Cordas soltas de E2 a E4: quantos harmonicos cabem abaixo de Nyquist depende de f0.
PROBE_FREQUENCIES_HZ: Tuple[float, ...] = (82.41, 110.0, 164.81, 246.94, 329.63)

DESCRIPTORS: Tuple[str, ...] = ("thd", "crest_drop", "hf_ratio", "flatness",
                                "thd_flatness")
# Media geometrica de THD e planicidade (esta relativa a do sinal seco). Cada um
# sozinho deita numa ponta do eixo: o THD satura no extremo sujo, a planicidade
# nao se mexe no extremo limpo.
PRIMARY_DESCRIPTOR = "thd_flatness"

DEFAULT_SWEEP_POINTS = 33
HF_CUTOFF_HZ = 2000.0

# |rho| de Spearman minimo entre knob e descritor para o arm ser aceito.
MONOTONICITY_THRESHOLD = 0.98


# --- descritores (funcoes puras, sem plugin) ---------------------------------
def _flat(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def harmonic_powers(
    signal: np.ndarray, sr: int, f0: float, n_harmonics: int = 12
) -> np.ndarray:
    """Potencia em cada harmonico de `f0`, somando o lobo principal da Hann (+-2 bins).

    Harmonicos acima de Nyquist saem como zero.
    """
    x = _flat(signal)
    n = x.size
    if n == 0:
        raise ValueError("sinal vazio")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    bin_hz = sr / n
    nyquist = sr / 2.0

    powers = np.zeros(n_harmonics, dtype=np.float64)
    for index in range(n_harmonics):
        frequency = f0 * (index + 1)
        if frequency >= nyquist:
            break
        center = int(round(frequency / bin_hz))
        lo = max(center - 2, 0)
        hi = min(center + 3, spectrum.size)
        powers[index] = spectrum[lo:hi].sum()
    return powers


def total_harmonic_distortion(
    signal: np.ndarray, sr: int, f0: float, n_harmonics: int = 12
) -> float:
    """THD = sqrt(soma da potencia dos harmonicos >= 2 / potencia do fundamental)."""
    powers = harmonic_powers(signal, sr, f0, n_harmonics)
    fundamental = powers[0]
    if fundamental <= 0.0:
        return 0.0
    return float(np.sqrt(powers[1:].sum() / fundamental))


def spectral_flatness(
    signal: np.ndarray, sr: int, lo_hz: float = 50.0, hi_hz: float = 16000.0
) -> float:
    """Media geometrica / media aritmetica do espectro de potencia, na banda util.

    Distorcao pesada preenche os vales entre os harmonicos e a planicidade sobe.
    Fora da banda, DC e bins quase nulos derrubariam a media geometrica.
    """
    x = _flat(signal)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    band = spectrum[(freqs > lo_hz) & (freqs < hi_hz)]
    total = band.sum()
    if total <= 0.0 or band.size == 0:
        return 0.0
    p = band / total + 1e-20
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def crest_factor_db(signal: np.ndarray) -> float:
    """20*log10(pico/RMS). Independente de nivel, cai quando o sinal e comprimido."""
    x = _flat(signal)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    if rms <= 0.0 or peak <= 0.0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def high_frequency_ratio(signal: np.ndarray, sr: int, cutoff_hz: float = HF_CUTOFF_HZ) -> float:
    """Fracao da energia acima de `cutoff_hz`; sobe com os harmonicos novos."""
    x = _flat(signal)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    total = spectrum.sum()
    if total <= 0.0:
        return 0.0
    return float(spectrum[freqs >= cutoff_hz].sum() / total)


# --- probes -------------------------------------------------------------------
@dataclass(frozen=True)
class Probes:
    """Sinais de sondagem no ponto de operacao do pipeline.

    As senoides tem o mesmo pico medio da guitarra a -26 LUFS, e nao a mesma
    loudness: a distorcao depende do nivel instantaneo.
    """

    sr: int
    sines: Tuple[np.ndarray, ...]
    frequencies: Tuple[float, ...]
    guitar: Tuple[np.ndarray, ...]


def make_sine(f0: float, sr: int, seconds: float, peak: float) -> np.ndarray:
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return (peak * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32).reshape(1, -1)


def build_probes(
    guitar_segments: Sequence[np.ndarray],
    sr: int,
    seconds: float = 2.0,
    frequencies: Sequence[float] = PROBE_FREQUENCIES_HZ,
) -> Probes:
    if not guitar_segments:
        raise ValueError("e preciso ao menos um segmento de guitarra para fixar o pico")
    target_peak = float(np.mean([np.max(np.abs(_flat(seg))) for seg in guitar_segments]))
    sines = tuple(make_sine(f0, sr, seconds, target_peak) for f0 in frequencies)
    return Probes(sr=sr, sines=sines, frequencies=tuple(frequencies), guitar=tuple(guitar_segments))


def load_guitar_probes(
    input_dir: Path,
    n: int = 8,
    sr_expected: Optional[int] = None,
    segment_seconds: float = 2.0,
    seed: int = 20260906,
) -> Tuple[List[np.ndarray], int]:
    """Segmentos de guitarra a -26 LUFS, tirados do fim da lista (fora dos splits do render)."""
    paths = sorted(Path(input_dir).glob("*.wav"))
    if len(paths) < n:
        raise ValueError(f"{input_dir} tem {len(paths)} wavs, menos que os {n} probes pedidos")

    rng = np.random.default_rng(seed)
    segments: List[np.ndarray] = []
    sr = sr_expected or 0
    for path in paths[-n:]:
        audio, file_sr = load_audio_file(path)
        if sr and file_sr != sr:
            raise ValueError(f"{path} tem sr={file_sr}, esperado {sr}")
        sr = file_sr
        frames = int(round(segment_seconds * file_sr))
        start = int(rng.integers(0, max(audio.shape[1] - frames, 1)))
        segments.append(normalize_loudness(audio[:, start : start + frames], file_sr))
    return segments, sr


# --- varredura ----------------------------------------------------------------
def describe(dry: np.ndarray, wet: np.ndarray, sr: int, f0: Optional[float]) -> Dict[str, float]:
    """Descritores de um par (seco, molhado). `f0=None` desabilita o THD."""
    values = {
        "crest_drop": crest_factor_db(dry) - crest_factor_db(wet),
        "hf_ratio": high_frequency_ratio(wet, sr) - high_frequency_ratio(dry, sr),
        "flatness": spectral_flatness(wet, sr),
    }
    values["thd"] = total_harmonic_distortion(wet, sr, f0) if f0 is not None else float("nan")
    return values


def sweep_descriptors(
    render: Callable[[np.ndarray, float], np.ndarray],
    knobs: Sequence[float],
    probes: Probes,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Descritores medios em cada posicao de knob.

    `render(segmento, knob) -> molhado` e injetado para testar sem VST3. O THD sai
    das senoides; os demais, dos segmentos de guitarra. `only` restringe o que e
    calculado e, com isso, o que e renderizado.
    """
    wanted = tuple(DESCRIPTORS) if only is None else tuple(only)
    unknown = set(wanted) - set(DESCRIPTORS)
    if unknown:
        raise ValueError(f"descritor desconhecido: {sorted(unknown)}; ha {DESCRIPTORS}")

    combinado = "thd_flatness" in wanted
    needs_sines = "thd" in wanted or combinado
    needs_guitar = bool({"crest_drop", "hf_ratio", "flatness"} & set(wanted)) or combinado
    dry_flatness = (
        float(np.mean([spectral_flatness(seg, probes.sr) for seg in probes.guitar]))
        if combinado else 1.0
    )

    out: Dict[str, List[float]] = {name: [] for name in wanted}
    for knob in knobs:
        if needs_sines:
            thd = [
                total_harmonic_distortion(render(sine, knob), probes.sr, f0)
                for sine, f0 in zip(probes.sines, probes.frequencies)
            ]
            if "thd" in out:
                out["thd"].append(float(np.mean(thd)))
        if needs_guitar:
            per_probe: Dict[str, List[float]] = {"crest_drop": [], "hf_ratio": [], "flatness": []}
            for segment in probes.guitar:
                wet = render(segment, knob)
                values = describe(segment, wet, probes.sr, f0=None)
                for name in per_probe:
                    per_probe[name].append(values[name])
            for name, collected in per_probe.items():
                if name in out:
                    out[name].append(float(np.mean(collected)))
            if combinado:
                flat_mean = float(np.mean(per_probe["flatness"]))
                thd_mean = float(np.mean(thd))
                out["thd_flatness"].append(
                    float(np.sqrt(max(thd_mean, 0.0) * max(flat_mean / dry_flatness, 0.0)))
                )
    return out


def monotonicity(knobs: Sequence[float], values: Sequence[float]) -> float:
    """rho de Spearman com sinal entre knob e descritor; degenerado vira 0.

    A porteira (`is_monotone`) olha o modulo: um knob decrescente tambem serve.
    """
    from scipy.stats import spearmanr

    value_array = np.asarray(values, dtype=float)
    knob_array = np.asarray(knobs, dtype=float)
    finite = np.isfinite(value_array)
    if finite.sum() < 3:
        return 0.0
    # Constante: correlacao indefinida (e o scipy avisaria).
    if np.ptp(value_array[finite]) == 0.0 or np.ptp(knob_array[finite]) == 0.0:
        return 0.0
    result = spearmanr(knob_array[finite], value_array[finite]).statistic
    return 0.0 if not np.isfinite(result) else float(result)


def is_monotone(rho: float, threshold: float = MONOTONICITY_THRESHOLD) -> bool:
    """Porteira: o knob de drive tem de ser monotono na quantidade de distorcao."""
    return bool(abs(rho) >= threshold)


def match_descriptor(
    knobs: Sequence[float], values: Sequence[float], target: float
) -> Tuple[float, bool]:
    """Knob cujo descritor vale `target`, por interpolacao linear na varredura.

    Devolve `(knob, alcancavel)`; fora da faixa, grampeia no extremo mais proximo.
    """
    knob_array = np.asarray(knobs, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(value_array)
    sorted_values, sorted_knobs = value_array[order], knob_array[order]

    reachable = bool(sorted_values[0] <= target <= sorted_values[-1])
    return float(np.interp(target, sorted_values, sorted_knobs)), reachable


def levels_from_range(k_lo: float, k_hi: float, n_levels: int = DRIVE_LEVELS) -> List[float]:
    """Os N niveis de drive uniformes no knob nativo entre os extremos.

    E como a referencia sempre e construida: o knob dela e a unidade de leitura.
    """
    return [float(value) for value in np.linspace(k_lo, k_hi, n_levels)]


def levels_from_targets(
    measure: Callable[[float], float],
    knobs: Sequence[float],
    values: Sequence[float],
    targets: Sequence[float],
    iterations: int = 6,
) -> Tuple[List[float], List[bool]]:
    """Um knob por alvo, cada um achado por bissecao medida."""
    out, reachable = [], []
    lo_v, hi_v = float(np.min(values)), float(np.max(values))
    for target in targets:
        ok = bool(lo_v <= target <= hi_v)
        reachable.append(ok)
        if not ok:
            out.append(float(knobs[int(np.argmin(np.abs(np.asarray(values) - target)))]))
            continue
        knob, _ = refine_knob(measure, *bracket(knobs, values, target), target,
                              iterations=iterations)
        out.append(float(knob))
    return out, reachable


# --- driver (carrega plugins) -------------------------------------------------
CALIBRATION_FILENAME = "arms_calibration.json"

RANGE_MODES = ("intersection", "reference")

# Como os niveis interiores sao posicionados:
#
#   "descriptor"  cada nivel casa o descritor que a referencia produz nele. O
#                 rotulo vale sonicamente entre arms, mas o oraculo deixa de
#                 poder validar a grade.
#   "knob"        uniformes no knob entre os extremos casados. O oraculo descobre
#                 a correspondencia, mas o interior pode derrapar varios niveis.
LEVEL_MODES = ("descriptor", "knob")


def sweep_arm(arm_spec: Arm, probes: Probes, points: int = DEFAULT_SWEEP_POINTS) -> Dict[str, object]:
    """Varre o knob de drive de um arm no seu dominio de sondagem."""
    lo, hi = arm_spec.sweep_range
    knobs = [float(value) for value in np.linspace(lo, hi, points)]
    loaded = LoadedArm(arm_spec, sr=probes.sr)

    def render(segment: np.ndarray, knob: float) -> np.ndarray:
        return loaded.render(segment, probes.sr, knob)

    values = sweep_descriptors(render, knobs, probes)
    return {
        "drive_param": arm_spec.drive_param,
        "knob": knobs,
        **values,
        "monotonicity": {name: monotonicity(knobs, values[name]) for name in DESCRIPTORS},
    }


def measure_at(
    arm_spec: Arm, probes: Probes, knobs: Sequence[float],
    only: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Mede os descritores exatamente nos knobs dados, sem interpolar."""
    loaded = LoadedArm(arm_spec, sr=probes.sr)
    return sweep_descriptors(
        lambda segment, knob: loaded.render(segment, probes.sr, knob), knobs, probes, only=only
    )


def bracket(
    knobs: Sequence[float], values: Sequence[float], target: float
) -> Tuple[float, float]:
    """Par de knobs adjacentes da varredura que cerca `target` (ou o do extremo mais proximo)."""
    knob_array = np.asarray(knobs, dtype=float)
    value_array = np.asarray(values, dtype=float)
    for index in range(len(knob_array) - 1):
        lo, hi = value_array[index], value_array[index + 1]
        if min(lo, hi) <= target <= max(lo, hi):
            return float(knob_array[index]), float(knob_array[index + 1])
    nearest = int(np.argmin(np.abs(value_array - target)))
    other = max(nearest - 1, 0) if nearest else 1
    return float(knob_array[min(nearest, other)]), float(knob_array[max(nearest, other)])


def refine_knob(
    measure: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    iterations: int = 8,
) -> Tuple[float, float]:
    """Bissecao no knob para atingir `target`, medindo em vez de interpolar.

    A interpolacao na varredura erra onde a curva tem joelho. Devolve
    `(knob, valor medido)` do melhor ponto visitado.
    """
    value_lo, value_hi = measure(lo), measure(hi)
    increasing = value_hi >= value_lo
    best = min(((lo, value_lo), (hi, value_hi)), key=lambda item: abs(item[1] - target))

    for _ in range(iterations):
        middle = 0.5 * (lo + hi)
        value = measure(middle)
        if abs(value - target) < abs(best[1] - target):
            best = (middle, value)
        if (value < target) == increasing:
            lo = middle
        else:
            hi = middle
    return best


def descriptor_probe(
    arm_spec: Arm, probes: Probes, descriptor: str
) -> Callable[[float], float]:
    """`measure(knob) -> descritor`, com o plugin carregado uma unica vez."""
    loaded = LoadedArm(arm_spec, sr=probes.sr)

    def measure(knob: float) -> float:
        values = sweep_descriptors(
            lambda segment, value: loaded.render(segment, probes.sr, value), [knob], probes,
            only=(descriptor,),
        )
        return float(values[descriptor][0])

    return measure


def common_range(
    sweeps: Dict[str, Dict[str, object]],
    descriptor: str,
    accepted: Sequence[str],
) -> Tuple[float, float]:
    """Faixa do descritor que todos os arms aceitos conseguem produzir.

    A intersecao, e nao a faixa da referencia: senao um arm mais suave teria varios
    niveis grampeados no mesmo knob.
    """
    lows, highs = [], []
    for key in accepted:
        values = np.asarray(sweeps[key][descriptor], dtype=float)
        values = values[np.isfinite(values)]
        lows.append(float(values.min()))
        highs.append(float(values.max()))
    return max(lows), min(highs)


def calibrate_arms(
    arm_keys_wanted: Optional[Sequence[str]] = None,
    input_dir: Path = Path("datasets/unprocessed_samples"),
    output: Optional[Path] = None,
    points: int = DEFAULT_SWEEP_POINTS,
    n_probes: int = 8,
    segment_seconds: float = 2.0,
    descriptor: str = PRIMARY_DESCRIPTOR,
    range_mode: str = "intersection",
    level_mode: str = "descriptor",
) -> Dict[str, object]:
    """Varre todos os arms, decide a faixa comum de drive e grava o JSON."""
    if range_mode not in RANGE_MODES:
        raise ValueError(f"range_mode deve ser um de {RANGE_MODES}")
    if level_mode not in LEVEL_MODES:
        raise ValueError(f"level_mode deve ser um de {LEVEL_MODES}")

    wanted = list(arm_keys_wanted) if arm_keys_wanted else arm_keys()
    if REFERENCE_ARM not in wanted:
        wanted = [REFERENCE_ARM] + wanted

    guitar, sr = load_guitar_probes(input_dir, n=n_probes, segment_seconds=segment_seconds)
    probes = build_probes(guitar, sr, seconds=segment_seconds)
    print(f"probes: {n_probes} segmentos de guitarra a {DEFAULT_LOUDNESS_LEVEL} LUFS, "
          f"{len(probes.frequencies)} senoides, sr={sr}\n")

    # Passada 1: varrer tudo; a faixa comum depende de todos.
    sweeps: Dict[str, Dict[str, object]] = {}
    accepted: List[str] = []
    print(f"{'arm':16s}{'knob':16s}{'rho':>9s}"
          f"{descriptor + '_min':>12s}{descriptor + '_max':>12s}")
    for key in wanted:
        spec = arm(key)
        sweeps[key] = sweep_arm(spec, probes, points)
        rho = sweeps[key]["monotonicity"][descriptor]
        values = np.asarray(sweeps[key][descriptor], dtype=float)
        ok = is_monotone(rho)
        if ok:
            accepted.append(key)
        flag = "" if ok else "   <-- REJEITADO (nao monotono)"
        print(f"{key:16s}{spec.drive_param:16s}{rho:>+9.4f}"
              f"{values.min():>12.6g}{values.max():>12.6g}{flag}")

    if REFERENCE_ARM not in accepted:
        raise RuntimeError(f"a referencia {REFERENCE_ARM} nao passou na monotonicidade")

    reference = sweeps[REFERENCE_ARM]
    ref_values = np.asarray(reference[descriptor], dtype=float)
    ref_knobs = np.asarray(reference["knob"], dtype=float)
    order = np.argsort(ref_values)
    ref_values_sorted, ref_knobs_sorted = ref_values[order], ref_knobs[order]

    if range_mode == "intersection":
        target_lo, target_hi = common_range(sweeps, descriptor, accepted)
    else:
        target_lo, target_hi = float(ref_values[0]), float(ref_values[-1])
    if target_lo >= target_hi:
        raise RuntimeError(
            f"faixa comum vazia: [{target_lo:.4f}, {target_hi:.4f}]. "
            "Algum arm nao consegue produzir a distorcao dos demais."
        )
    ref_lo, ref_hi = np.interp([target_lo, target_hi], ref_values_sorted, ref_knobs_sorted)
    print(f"\nfaixa de {descriptor} ({range_mode}): [{target_lo:.6g}, {target_hi:.6g}]"
          f"  =  drive-tanh equivalente [{ref_lo:.2f}, {ref_hi:.2f}] dB\n")

    # Passada 2: casar os niveis. A referencia primeiro: os niveis dela sao os alvos.
    ordered = [REFERENCE_ARM] + [key for key in wanted if key != REFERENCE_ARM]
    arms_out: Dict[str, object] = {}
    level_targets: Optional[np.ndarray] = None
    print(f"niveis: {level_mode}")
    for key in ordered:
        spec = arm(key)
        sweep = sweeps[key]
        knobs = np.asarray(sweep["knob"], dtype=float)
        values = np.asarray(sweep[descriptor], dtype=float)
        rho = sweep["monotonicity"][descriptor]
        measure = descriptor_probe(spec, probes, descriptor)

        if level_mode == "descriptor" and key != REFERENCE_ARM and level_targets is not None:
            levels, reach = levels_from_targets(measure, knobs, values, level_targets)
            reachable_lo, reachable_hi = reach[0], reach[-1]
            k_lo, k_hi = levels[0], levels[-1]
        else:
            k_lo, reachable_lo = match_descriptor(knobs, values, target_lo)
            k_hi, reachable_hi = match_descriptor(knobs, values, target_hi)

            if reachable_lo:
                k_lo, _ = refine_knob(measure, *bracket(knobs, values, target_lo), target_lo)
            if reachable_hi:
                k_hi, _ = refine_knob(measure, *bracket(knobs, values, target_hi), target_hi)
            levels = levels_from_range(k_lo, k_hi)

        level_values = np.asarray(
            measure_at(spec, probes, levels, only=(descriptor,))[descriptor], dtype=float
        )
        if key == REFERENCE_ARM:
            level_targets = level_values.copy()

        arms_out[key] = {
            "stratum": spec.stratum,
            "drive_param": spec.drive_param,
            "drive_in_db": spec.drive_in_db,
            "sweep": {name: sweep[name] for name in ("knob",) + DESCRIPTORS},
            "level_mode": level_mode,
            "monotonicity": sweep["monotonicity"],
            "accepted": key in accepted,
            "k_lo": k_lo,
            "k_hi": k_hi,
            "reachable_lo": reachable_lo,
            "reachable_hi": reachable_hi,
            "levels": levels,
            "level_descriptor": [float(value) for value in level_values],
            "drive_db_equivalente": [
                float(value)
                for value in np.interp(level_values, ref_values_sorted, ref_knobs_sorted)
            ],
        }
        reach = "" if reachable_lo and reachable_hi else "   <-- faixa comum nao alcancada"
        span = np.asarray(arms_out[key]["drive_db_equivalente"], dtype=float)
        # Desvio sobre todos os niveis: e no interior que a grade derrapa.
        if key == REFERENCE_ARM:
            reference_ladder = span.copy()
        drift = float(np.max(np.abs(span - reference_ladder)))
        print(f"  {key:16s} k=[{k_lo:8.3f}, {k_hi:8.3f}]  "
              f"dB-equiv=[{span[0]:6.2f}, {span[-1]:6.2f}]  "
              f"pior desvio de nivel={drift:5.2f} dB{reach}")

    report = {
        "sr": sr,
        "loudness_lufs": DEFAULT_LOUDNESS_LEVEL,
        "descriptor": descriptor,
        "range_mode": range_mode,
        "level_mode": level_mode,
        "points": points,
        "drive_levels": DRIVE_LEVELS,
        "monotonicity_threshold": MONOTONICITY_THRESHOLD,
        "probe_frequencies_hz": list(probes.frequencies),
        "n_guitar_probes": n_probes,
        "reference_arm": REFERENCE_ARM,
        "accepted_arms": accepted,
        "target_descriptor_range": [target_lo, target_hi],
        # O que a calibracao nao removeu, medido por escuta depois dela.
        "perceptual_bias": {k: v for k, v in PERCEPTUAL_BIAS.items() if k in wanted},
        "target_drive_db_equivalente": [float(ref_lo), float(ref_hi)],
        "arms": arms_out,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nescrito em {output}")
    return report


def load_calibration(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
