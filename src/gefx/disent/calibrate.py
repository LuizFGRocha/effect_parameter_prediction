"""Calibracao da faixa util do knob de drive de cada arm.

O problema: implementacoes diferentes de distorcao nao compartilham unidade de
ganho, entao "o mesmo valor de knob" nao quer dizer "a mesma quantidade de
distorcao". A solucao adotada tem duas partes deliberadamente separadas:

1. **Aqui** calibram-se APENAS os extremos. Varre-se o knob de cada arm, mede-se
   um descritor de quantidade de distorcao (THD num probe padronizado) e
   escolhem-se `k_lo`/`k_hi` que casem os extremos da referencia. Os 8 niveis de
   drive sao entao **uniformes no knob nativo, nao no descritor**.
2. O interior da correspondencia entre arms fica livre e e medido a posteriori
   pelo oraculo (`disent/oracle.py`).

Essa divisao e o que mantem a avaliacao nao-circular: se a grade fosse construida
casando o descritor em todos os 8 niveis, o oraculo seria obrigado a devolver a
diagonal e nao mediria nada.

A hipotese declarada e **monotonicidade**: o knob de drive de cada arm e monotono
na quantidade de distorcao. Ela e verificada aqui (rho de Spearman) e um arm que
nao passe e rejeitado -- e por isso que esta checagem serve de porteira barata
para candidatos novos.

Validacao do proprio protocolo: em `lsp-tanh`, que compartilha a funcao tanh e a
unidade de dB com a referencia, a calibracao tem de devolver ~[5, 40] dB. Se
devolver, o protocolo se justifica para os arms sem unidade comum.
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

# Notas de guitarra em corda solta, de Mi grave (E2) a Mi agudo (E4). O THD e
# medido nelas e nao numa senoide unica porque a quantidade de harmonicos que
# cabe abaixo de Nyquist depende de f0.
PROBE_FREQUENCIES_HZ: Tuple[float, ...] = (82.41, 110.0, 164.81, 246.94, 329.63)

DESCRIPTORS: Tuple[str, ...] = ("thd", "crest_drop", "hf_ratio", "flatness",
                                "thd_flatness")
# Media geometrica de THD e planicidade, e nao um dos dois sozinho. Os dois
# falham em pontas opostas do eixo, e a falha nao e de monotonicidade -- e de
# IDENTIFICABILIDADE, que e pior porque nao aparece no rho:
#
#   THD        no extremo sujo, 10,1 dB de knob mapeiam para o mesmo valor: as
#              sigmoides saturam perto de THD 0,43 e a curva deita.
#   planicidade no extremo limpo, 13,5 dB mapeiam para o mesmo valor: abaixo de
#              ~12 dB o clipper e transparente e a planicidade fica cravada na da
#              guitarra seca.
#
# Extremo inidentificavel nao e detalhe de relatorio: `k_lo` e `k_hi` definem a
# escada inteira de 8 niveis, que e uniforme no knob entre eles. Uma janela de
# 13 dB ambigua pendura os 8 niveis num ponto arbitrario.
#
# A media geometrica herda a sensibilidade de quem esta sensivel localmente: o
# THD carrega o extremo limpo, a planicidade o sujo. Medido sobre as mesmas
# varreduras: 7/7 arms aceitos, eixo de 29,1 dB equivalentes (contra 22,1 do THD
# e 27,1 da planicidade) e ambiguidade ZERO nos dois extremos.
#
# A planicidade entra normalizada pela do sinal SECO, entao ela e adimensional e
# vale 1 quando o arm esta transparente -- isso a poe na mesma escala do THD, que
# ja e uma razao.
PRIMARY_DESCRIPTOR = "thd_flatness"

DEFAULT_SWEEP_POINTS = 33
HF_CUTOFF_HZ = 2000.0

# rho de Spearman minimo entre knob e descritor para o arm ser aceito.
MONOTONICITY_THRESHOLD = 0.98


# --- descritores (funcoes puras, sem plugin) ---------------------------------
def _flat(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def harmonic_powers(
    signal: np.ndarray, sr: int, f0: float, n_harmonics: int = 12
) -> np.ndarray:
    """Potencia em cada harmonico de `f0`, do fundamental para cima.

    Janela de Hann e soma de +-2 bins em torno de cada harmonico, que e a largura
    do lobo principal da Hann. Harmonicos acima de Nyquist saem como zero.
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
    """THD = sqrt(soma da potencia dos harmonicos >= 2 / potencia do fundamental).

    E a medida padrao de "quantidade de distorcao" em engenharia de audio e nao
    depende de unidade de ganho nenhuma, que e exatamente o que se precisa para
    comparar implementacoes.
    """
    powers = harmonic_powers(signal, sr, f0, n_harmonics)
    fundamental = powers[0]
    if fundamental <= 0.0:
        return 0.0
    return float(np.sqrt(powers[1:].sum() / fundamental))


def spectral_flatness(
    signal: np.ndarray, sr: int, lo_hz: float = 50.0, hi_hz: float = 16000.0
) -> float:
    """Media geometrica / media aritmetica do espectro de potencia.

    Mede o quanto o espectro e ruidoso em vez de tonal: distorcao pesada preenche
    os vales entre os harmonicos e a planicidade sobe. Ao contrario do crest, nao
    depende do fator de crista do sinal; ao contrario do centroide, nao depende da
    inclinacao espectral.

    Foi ela que reproduziu a ordenacao de distorcao percebida num teste de escuta
    onde THD, crest e centroide falharam -- com o THD casado em 0,327 nos
    extremos, a planicidade ainda variava 10x entre os arms, e a ordem que ela dava
    (bigmuff > mxr > resto) foi a que o ouvido apontou.

    A banda e limitada porque DC e o lixo acima da banda util dominariam a media
    geometrica: um unico bin quase nulo derruba o produto inteiro.
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
    """Sinais de sondagem, ja no ponto de operacao do pipeline.

    As senoides sao escaladas para o mesmo **pico** medio dos segmentos de
    guitarra normalizados a -26 LUFS. Isso importa: uma senoide a -26 LUFS tem
    pico muito mais baixo que guitarra a -26 LUFS (fator de crista bem menor), e
    a distorcao e uma funcao do nivel instantaneo -- medir com a senoide no nivel
    errado leria a curva de transferencia no lugar errado.
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
    """Segmentos de guitarra normalizados a -26 LUFS, tirados do fim da lista.

    Do FIM de proposito: a renderizacao da grade consome as gravacoes a partir do
    inicio, entao os probes ficam fora dos splits de treino/catalogo/consulta.
    """
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
        # Absoluta, nao diferenca: a planicidade do seco e praticamente a mesma
        # para qualquer trecho de guitarra, e subtrair so acrescentaria ruido.
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

    `render(segmento, knob) -> molhado` e injetado para que esta funcao -- o miolo
    da calibracao -- seja testavel com um waveshaper sintetico, sem VST3.

    O THD sai das senoides (precisa de f0 conhecido); crest e razao de agudos
    saem dos segmentos de guitarra, que sao o material real.

    `only` restringe o que e calculado, e com isso o que e renderizado. Pedir so
    o THD pula os 8 segmentos de guitarra por knob e deixa 5 renders no lugar de
    13 -- o que importa porque a bissecao chama isto uma vez por iteracao e o
    BYOD renderiza mais devagar que tempo real (2,2 s para 2 s de audio).
    """
    wanted = tuple(DESCRIPTORS) if only is None else tuple(only)
    unknown = set(wanted) - set(DESCRIPTORS)
    if unknown:
        raise ValueError(f"descritor desconhecido: {sorted(unknown)}; ha {DESCRIPTORS}")

    combinado = "thd_flatness" in wanted
    needs_sines = "thd" in wanted or combinado
    needs_guitar = bool({"crest_drop", "hf_ratio", "flatness"} & set(wanted)) or combinado
    # Referencia da normalizacao: a planicidade do proprio material seco. Fica
    # fora do laco de knob porque nao depende dele.
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
    """rho de Spearman com sinal entre knob e descritor. Degenerado vira 0.

    O sinal e informativo mas a porteira usa `is_monotone`, que olha o modulo: um
    knob monotono decrescente e perfeitamente utilizavel (basta inverter), e
    existem circuitos assim -- o Tube Screamer do BYOD perde THD conforme o ganho
    de entrada sobe, porque empurra o sinal para um estagio compressivo.
    """
    from scipy.stats import spearmanr

    value_array = np.asarray(values, dtype=float)
    knob_array = np.asarray(knobs, dtype=float)
    finite = np.isfinite(value_array)
    if finite.sum() < 3:
        return 0.0
    # Descritor constante nao tem correlacao definida; devolver 0 evita o
    # ConstantInputWarning do scipy num caminho que e esperado (knob inerte).
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

    Devolve `(knob, alcancavel)`. Fora da faixa medida o valor e grampeado no
    extremo mais proximo e `alcancavel` sai False -- e o caso de um arm que nao
    consegue chegar a distorcao da referencia, que deve ser reportado e nao
    escondido.
    """
    knob_array = np.asarray(knobs, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(value_array)
    sorted_values, sorted_knobs = value_array[order], knob_array[order]

    reachable = bool(sorted_values[0] <= target <= sorted_values[-1])
    return float(np.interp(target, sorted_values, sorted_knobs)), reachable


def levels_from_range(k_lo: float, k_hi: float, n_levels: int = DRIVE_LEVELS) -> List[float]:
    """Os N niveis de drive uniformes NO KNOB NATIVO entre os extremos calibrados.

    E assim que a REFERENCIA e sempre construida, em qualquer `level_mode`: o
    knob dela e o `drive_db` do POC I, entao uniforme no knob nativo quer dizer
    uniforme na unidade de leitura do trabalho todo.
    """
    return [float(value) for value in np.linspace(k_lo, k_hi, n_levels)]


def levels_from_targets(
    measure: Callable[[float], float],
    knobs: Sequence[float],
    values: Sequence[float],
    targets: Sequence[float],
    iterations: int = 6,
) -> Tuple[List[float], List[bool]]:
    """Um knob por alvo, cada um achado por bissecao medida.

    Menos iteracoes que nos extremos (6 contra 8) porque aqui o erro nao se
    propaga: em `level_mode="knob"` os extremos definem a escada inteira, entao
    errar `k_lo` desloca os 8 niveis; aqui cada nivel e resolvido sozinho.
    """
    out, reachable = [], []
    lo_v, hi_v = float(np.min(values)), float(np.max(values))
    for target in targets:
        ok = bool(lo_v <= target <= hi_v)
        reachable.append(ok)
        if not ok:
            # Fora do alcance, o extremo mais proximo e a resposta certa e deve
            # ser reportado, nao escondido.
            out.append(float(knobs[int(np.argmin(np.abs(np.asarray(values) - target)))]))
            continue
        knob, _ = refine_knob(measure, *bracket(knobs, values, target), target,
                              iterations=iterations)
        out.append(float(knob))
    return out, reachable


# --- driver (carrega plugins) -------------------------------------------------
CALIBRATION_FILENAME = "arms_calibration.json"

RANGE_MODES = ("intersection", "reference")

# Como os niveis INTERIORES sao posicionados.
#
#   "knob"        uniformes no knob nativo entre os extremos casados. Preserva o
#                 argumento de nao-circularidade (o oraculo DESCOBRE a
#                 correspondencia em vez de confirma-la), mas mede-se um desvio
#                 de ate 11,9 dB equivalentes no `lsp-hardclip` -- quase tres
#                 niveis. Um rotulo errado por tres niveis nao serve de positivo
#                 contrastivo entre arms.
#   "descriptor"  cada nivel casado contra o valor que a REFERENCIA produz
#                 naquele nivel. O rotulo passa a valer sonicamente entre
#                 implementacoes; em troca, o oraculo deixa de poder validar a
#                 grade e vira medida independente de concordancia.
#
# Os dois sao renderizados: o desenho da grade vira variavel do experimento.
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
    """Par de knobs adjacentes da varredura que cerca `target`.

    Se `target` cair fora do medido, devolve o intervalo do extremo mais proximo.
    """
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

    Interpolar linearmente na varredura falha onde a curva tem joelho, e nao e um
    erro so de relatorio: `k_lo` e `k_hi` definem os niveis que vao para o disco.
    Medido no hard clip, a interpolacao errava o extremo inferior em 4,4 dB
    equivalentes -- o nivel 0 desse arm sairia bem mais limpo que o dos demais e a
    grade deixaria de estar alinhada entre implementacoes.

    Devolve `(knob, valor medido)` do melhor ponto visitado.
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
    """Faixa do descritor que TODOS os arms aceitos conseguem produzir.

    E a intersecao das faixas alcancaveis, nao a faixa da referencia. Sem isso um
    arm mais suave teria varios niveis grampeados no mesmo valor de knob e a
    grade degeneraria -- 8 niveis dos quais so 3 soam diferentes.
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
        # Sem a referencia nao ha unidade de leitura.
        wanted = [REFERENCE_ARM] + wanted

    guitar, sr = load_guitar_probes(input_dir, n=n_probes, segment_seconds=segment_seconds)
    probes = build_probes(guitar, sr, seconds=segment_seconds)
    print(f"probes: {n_probes} segmentos de guitarra a {DEFAULT_LOUDNESS_LEVEL} LUFS, "
          f"{len(probes.frequencies)} senoides, sr={sr}\n")

    # Passada 1: varrer tudo, para so entao decidir a faixa comum.
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

    # Passada 2: casar os extremos e distribuir os niveis.
    #
    # A referencia vem primeiro porque em level_mode="descriptor" sao os valores
    # que ELA produz em cada nivel que servem de alvo para todos os outros.
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

            # Refina os extremos medindo: e a interpolacao na varredura que errava
            # o joelho do hard clip em 4,4 dB. So refina o que e alcancavel --
            # fora da faixa o grampeamento e a resposta certa.
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
            # Unidade de leitura: o drive da tanh de referencia que produz o mesmo
            # descritor. Nao e conversao de unidade, e casamento de medida.
            "drive_db_equivalente": [
                float(value)
                for value in np.interp(level_values, ref_values_sorted, ref_knobs_sorted)
            ],
        }
        reach = "" if reachable_lo and reachable_hi else "   <-- faixa comum nao alcancada"
        span = np.asarray(arms_out[key]["drive_db_equivalente"], dtype=float)
        # Desvio sobre TODOS os niveis, nao so os extremos: e no interior que a
        # grade uniforme-no-knob derrapa (ate 11,9 dB no hard clip), e esconder
        # isso num numero de extremos foi o que deixou o defeito passar antes.
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
        # Vies que a calibracao NAO removeu, medido por escuta depois dela. Vai
        # junto para que ninguem leia os desvios de nivel como alinhamento
        # perceptual completo.
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


def drive_knob(calibration: Dict[str, object], arm_key: str, level: int) -> float:
    """Valor de knob do nivel de drive `level` daquele arm."""
    levels = calibration["arms"][arm_key]["levels"]
    if not 0 <= level < len(levels):
        raise ValueError(f"nivel {level} fora de [0, {len(levels)})")
    return float(levels[level])
