"""Niveis de drive de todos os arms, pelo Rnonlin.

A referencia vai de `reference_db[0]` a `reference_db[1]` dB com os niveis
igualmente espacados em 1 - Rnonlin, e cada outro arm recebe, em cada nivel, o
knob que distorce tanto quanto ela. "Quanto" e o Rnonlin (Tan, Moore, Zacharov e
Mattila, JAES 2004): seco e processado passam por um banco gammatone espacado em
1 ERB, e em quadros de 30 ms o pico da correlacao cruzada normalizada de cada
banda e ponderado pela energia da banda no processado; os quadros sao ponderados
pelo nivel. 1 = sem distorcao.

Por que ele e nao a queda do fator de crista: em niveis da referencia separados
de ouvido por diferenca audivel, os degraus em 1 - Rnonlin foram os mais
regulares (CV 0,15-0,18, contra 0,21 do dB e 0,73 da crista), e a correlacao por
banda nao e enganada por um filtro depois do clipador -- a crista era, e dava ao
BYOD metade da escala. A faixa padrao tambem veio do ouvido: 18,45 dB e o
primeiro nivel audivel, e 39 dB fica abaixo do teto do Big Muff, o arm de menor
alcance (acima disso os niveis dele caem na parte plana da curva e soam iguais).

1. varre-se o knob de cada arm em `sweep`, a referencia inclusive (um processo
   por arm);
2. os alvos sao `n_levels` valores de 1 - Rnonlin igualmente espacados entre os
   da referencia nas duas pontas de `reference_db`;
3. o knob de cada alvo sai por interpolacao na curva. Alvos fora do alcance do
   arm, acima ou abaixo, vao para a ponta do knob e sao marcados em `unmatched`:
   ali o arm distorce menos (ou mais) do que o nivel pede.

Os niveis que saem daqui sao os do render; nao ha ajuste manual.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gefx.audio import load_audio_file, normalize_loudness
from gefx.disent.arms import Arm, LoadedArm, load_roster, write_levels

DESCRIPTOR = "rnonlin"
DEFAULT_REFERENCE_DB: Tuple[float, float] = (18.45, 39.0)
DEFAULT_SWEEP_POINTS = 33
FRAME_S = 0.030
MAX_LAG_S = 0.001
BAND_HZ: Tuple[float, float] = (50.0, 16000.0)


# --- Rnonlin -------------------------------------------------------------------
def erb_space(lo: float = BAND_HZ[0], hi: float = BAND_HZ[1]) -> np.ndarray:
    """Centros espacados em 1 ERB (Glasberg e Moore, 1990)."""
    def erb_n(f):
        return 21.4 * np.log10(4.37e-3 * f + 1)

    return (10 ** (np.arange(erb_n(lo), erb_n(hi), 1.0) / 21.4) - 1) / 4.37e-3


def gammatone_band(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    """Gammatone de 4a ordem: desloca a banda para 0 Hz, 4 polos reais, volta.

    O `scipy.signal.gammatone` em forma b/a diverge abaixo de ~160 Hz.
    """
    from scipy.signal import lfilter

    erb = 24.7 * (4.37e-3 * fc + 1)
    a = np.exp(-2 * np.pi * 1.019 * erb / sr)
    shift = np.exp(-2j * np.pi * fc * np.arange(len(x)) / sr)
    z = x * shift
    for _ in range(4):
        z = lfilter([1 - a], [1, -a], z)
    return np.real(z * np.conj(shift))


def _align(dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Tira a latencia do plugin pelo pico da correlacao de banda larga."""
    from scipy.signal import correlate

    corr = correlate(wet, dry, mode="full", method="fft")
    return np.roll(wet, -(int(np.argmax(np.abs(corr))) - (len(dry) - 1)))


def rnonlin(dry: np.ndarray, wet: np.ndarray, sr: int,
            centers: Optional[np.ndarray] = None) -> float:
    dry = np.asarray(dry, dtype=np.float64).reshape(-1)
    wet = _align(dry, np.asarray(wet, dtype=np.float64).reshape(-1))
    centers = erb_space() if centers is None else centers
    n = int(FRAME_S * sr)
    frames = len(dry) // n
    lag = int(MAX_LAG_S * sr)
    weighted = np.zeros(frames)
    weights = np.zeros(frames)
    for fc in centers:
        x = gammatone_band(dry, fc, sr)[: frames * n].reshape(frames, n)
        y = gammatone_band(wet, fc, sr)[: frames * n].reshape(frames, n)
        cross = np.fft.irfft(np.fft.rfft(y, 2 * n) * np.conj(np.fft.rfft(x, 2 * n)), 2 * n)
        near = np.concatenate([cross[:, : lag + 1], cross[:, -lag:]], axis=1)
        norm = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1)) + 1e-20
        energy = np.sqrt(np.mean(y**2, axis=1))
        weighted += energy * np.max(np.abs(near), axis=1) / norm
        weights += energy
    per_frame = weighted / (weights + 1e-20)
    level = np.sqrt(np.mean(wet[: frames * n].reshape(frames, n) ** 2, axis=1))
    keep = level > 1e-3 * level.max()  # silencio fora
    return float(np.sum(level[keep] * per_frame[keep]) / np.sum(level[keep]))


# --- varredura -------------------------------------------------------------------
def load_segments(input_dir: Path, n: int = 8, seconds: float = 2.0,
                  seed: int = 20260906) -> Tuple[List[np.ndarray], int]:
    """Segmentos a -26 LUFS das ultimas gravacoes da lista, como no render."""
    paths = sorted(Path(input_dir).glob("*.wav"))
    if len(paths) < n:
        raise ValueError(f"{input_dir} tem {len(paths)} wavs, menos que os {n} pedidos")
    rng = np.random.default_rng(seed)
    segments, sr = [], 0
    for path in paths[-n:]:
        audio, sr = load_audio_file(path)
        frames = int(round(seconds * sr))
        start = int(rng.integers(0, max(audio.shape[1] - frames, 1)))
        segments.append(normalize_loudness(audio[:, start : start + frames], sr))
    return segments, sr


def sweep_knobs(loaded: LoadedArm, points: int) -> np.ndarray:
    arm = loaded.arm
    if arm.sweep is not None:
        lo, hi = arm.sweep
    else:
        param = loaded.plugin.parameters[arm.drive_param]
        lo, hi = float(param.min_value), float(param.max_value)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"{arm.key}: o knob vai a {lo}..{hi}; declare `sweep` no roster")
    return np.linspace(lo, hi, points)


def arm_curve(arm: Arm, segments: Sequence[np.ndarray], sr: int,
              knobs: Optional[Sequence[float]] = None, points: int = DEFAULT_SWEEP_POINTS
              ) -> Tuple[np.ndarray, np.ndarray]:
    """(knobs, Rnonlin medio em cada knob). Roda num processo proprio."""
    loaded = LoadedArm(arm, sr)
    knobs = sweep_knobs(loaded, points) if knobs is None else np.asarray(knobs, dtype=float)
    centers = erb_space()
    curve = np.array([np.mean([rnonlin(s, loaded.render(s, sr, k), sr, centers)
                               for s in segments]) for k in knobs])
    return knobs, curve


# --- pareamento ------------------------------------------------------------------
def knob_for(knobs: Sequence[float], curve: Sequence[float], target: float) -> float:
    """Primeiro knob em que a curva (tornada monotona crescente) alcanca `target`."""
    knobs = np.asarray(knobs, dtype=np.float64)
    envelope = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
    if target > envelope[-1]:
        raise ValueError(f"alvo {target:.3f} acima do alcance {envelope[-1]:.3f}")
    i = int(np.argmax(envelope >= target))
    if i == 0:
        return float(knobs[0])
    frac = (target - envelope[i - 1]) / (envelope[i] - envelope[i - 1])
    return float(knobs[i - 1] + frac * (knobs[i] - knobs[i - 1]))


def arm_knobs(knobs: Sequence[float], curve: Sequence[float], targets: Sequence[float]
              ) -> Tuple[List[float], List[int]]:
    """Knob de cada alvo, com curva e alvos crescendo com a distorcao, e os niveis
    (base 1) fora do alcance. Os de baixo vao do inicio do knob ao primeiro
    pareado; os de cima, do ultimo pareado ao fim.

    A curva comeca no ponto mais limpo do arm: com o knob muito baixo, o ruido e
    os filtros do circuito pesam mais que o clipping (o MXR fica *menos* limpo
    abaixo de ~-35), e esse trecho nao e distorcao.
    """
    start = int(np.argmin(curve))
    knobs, curve = np.asarray(knobs, dtype=np.float64)[start:], np.asarray(curve)[start:]
    envelope = np.maximum.accumulate(curve.astype(np.float64))
    lo_knob, hi_knob = float(knobs[0]), float(knobs[-1])
    n_below = sum(t < envelope[0] for t in targets)
    n_above = sum(t > envelope[-1] for t in targets)
    matched = [knob_for(knobs, curve, t) for t in targets[n_below : len(targets) - n_above]]
    if not matched:
        values = list(np.linspace(lo_knob, hi_knob, len(targets)))
    else:
        values = (list(np.linspace(lo_knob, matched[0], n_below + 1)[:-1]) + matched
                  + list(np.linspace(matched[-1], hi_knob, n_above + 1)[1:]))
    n = len(targets)
    unmatched = list(range(1, n_below + 1)) + list(range(n - n_above + 1, n + 1))
    return [float(v) for v in values], unmatched


def calibrate(
    roster_path: Path,
    levels_path: Path,
    input_dir: Path,
    curves_csv: Path,
    points: int = DEFAULT_SWEEP_POINTS,
    n_segments: int = 8,
    workers: int = 8,
    reference_db: Tuple[float, float] = DEFAULT_REFERENCE_DB,
    n_levels: int = 8,
) -> Dict[str, List[float]]:
    roster = load_roster(roster_path, levels_path=None)
    ref = roster.reference

    segments, sr = load_segments(input_dir, n_segments)
    others = [arm for arm in roster.arms if arm.key != ref]
    with ProcessPoolExecutor(max_workers=min(workers, len(others) + 1)) as pool:
        ref_job = pool.submit(arm_curve, roster.arm(ref), segments, sr, None, points)
        jobs = {arm.key: pool.submit(arm_curve, arm, segments, sr, None, points)
                for arm in others}
        ref_knobs, ref_r = ref_job.result()
        curves = {key: job.result() for key, job in jobs.items()}

    ends = 1 - np.interp(reference_db, ref_knobs, ref_r)
    targets = list(np.linspace(ends[0], ends[1], n_levels))  # cresce com a distorcao
    reference_levels = [round(v, 4) for v in arm_knobs(ref_knobs, 1 - ref_r, targets)[0]]
    print(f"referencia: {reference_levels} dB, Rnonlin {1 - targets[0]:.3f} a "
          f"{1 - targets[-1]:.3f}")
    arms: Dict[str, Dict[str, object]] = {
        ref: {"levels": reference_levels, "unmatched": []},
    }
    for arm in others:
        knobs, curve = curves[arm.key]
        values, unmatched = arm_knobs(knobs, 1 - curve, targets)
        values = [round(v, 4) for v in values]
        arms[arm.key] = {"levels": values, "unmatched": unmatched}
        aviso = f"  [aviso] niveis {unmatched} fora do alcance" if unmatched else ""
        print(f"  {arm.key:16s} Rnonlin {curve.max():.3f} a {curve.min():.3f}{aviso}")

    write_levels(levels_path, {
        "descriptor": DESCRIPTOR,
        "targets": [round(1 - float(t), 4) for t in targets],
        "arms": arms,
    })
    _write_curves(curves_csv, {ref: (ref_knobs, ref_r), **curves})
    return {key: spec["levels"] for key, spec in arms.items()}  # type: ignore[misc]


def _write_curves(path: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"arm": key, "knob": float(k), DESCRIPTOR: float(v)}
        for key, (knobs, curve) in curves.items() for k, v in zip(knobs, curve)
    ]).to_csv(path, index=False)
