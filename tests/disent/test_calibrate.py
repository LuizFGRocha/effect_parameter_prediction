"""Calibracao: descritores puros e a logica de casamento de extremos.

Nada aqui carrega VST3. O miolo (`sweep_descriptors`) recebe a funcao de
renderizacao injetada justamente para poder ser exercitado com um waveshaper
sintetico de ganho conhecido.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent import calibrate as cal

from gefx.disent.calibrate import (
    MONOTONICITY_THRESHOLD,
    bracket,
    common_range,
    is_monotone,
    refine_knob,
    Probes,
    build_probes,
    crest_factor_db,
    describe,
    harmonic_powers,
    high_frequency_ratio,
    levels_from_range,
    make_sine,
    match_descriptor,
    monotonicity,
    sweep_descriptors,
    total_harmonic_distortion,
)

SR = 22050


def _tanh_shaper(gain_db: float):
    def render(segment, knob=None):
        gain = 10.0 ** (float(gain_db if knob is None else knob) / 20.0)
        return np.tanh(gain * np.asarray(segment, dtype=np.float64))

    return render


def _harmonic_stack(f0: float, seconds: float, peak: float) -> np.ndarray:
    """Guitarra sintetica: harmonicos com amplitude 1/k.

    Ruido branco nao serve de probe: ja e espectralmente plano, entao distorce-lo
    nao aumenta a razao de agudos e `hf_ratio` deixa de ser monotona. Um espectro
    decrescente, que e o de uma corda, e o caso representativo.
    """
    t = np.arange(int(round(seconds * SR)), dtype=np.float64) / SR
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (peak * wave / np.max(np.abs(wave))).astype(np.float32).reshape(1, -1)


def _probes(n_guitar: int = 2, seconds: float = 0.5) -> Probes:
    guitar = [_harmonic_stack(110.0 * (index + 1), seconds, 0.3) for index in range(n_guitar)]
    return build_probes(guitar, SR, seconds=seconds, frequencies=(220.0, 440.0))


# --- descritores --------------------------------------------------------------
def test_pure_sine_has_essentially_no_thd():
    sine = make_sine(220.0, SR, 0.5, 0.5)
    assert total_harmonic_distortion(sine, SR, 220.0) < 1e-3


def test_thd_grows_with_drive():
    sine = make_sine(220.0, SR, 0.5, 0.5)
    values = [total_harmonic_distortion(np.tanh(g * sine), SR, 220.0) for g in (1, 3, 10, 40)]
    assert values == sorted(values)


def test_thd_of_a_square_wave_approaches_the_theoretical_limit():
    # Onda quadrada: harmonicos impares com amplitude 1/k, THD = sqrt(pi^2/8 - 1)
    # ~ 0,4834. E o teto do descritor, o que explica ele saturar em drive alto.
    sine = make_sine(220.0, SR, 0.5, 0.5)
    square = np.sign(sine)
    assert total_harmonic_distortion(square, SR, 220.0, n_harmonics=40) == pytest.approx(
        np.sqrt(np.pi**2 / 8 - 1), rel=0.05
    )


def test_harmonic_powers_stops_at_nyquist():
    powers = harmonic_powers(make_sine(5000.0, SR, 0.2, 0.5), SR, 5000.0, n_harmonics=8)
    # 5 kHz com Nyquist em 11025 Hz: so o fundamental e o 2o harmonico cabem.
    assert powers[0] > 0 and powers[1] >= 0
    assert np.all(powers[2:] == 0.0)


def test_thd_is_zero_when_there_is_no_fundamental():
    assert total_harmonic_distortion(np.zeros((1, 1024)), SR, 220.0) == 0.0


def test_crest_factor_of_a_sine_is_three_db():
    assert crest_factor_db(make_sine(220.0, SR, 0.5, 0.5)) == pytest.approx(3.0103, abs=1e-2)


def test_crest_factor_is_level_independent():
    sine = make_sine(220.0, SR, 0.5, 0.5)
    assert crest_factor_db(sine) == pytest.approx(crest_factor_db(0.1 * sine))


def test_crest_factor_of_silence_is_zero():
    assert crest_factor_db(np.zeros((1, 512))) == 0.0


def test_high_frequency_ratio_separates_low_from_high():
    low = make_sine(200.0, SR, 0.5, 0.5)
    high = make_sine(5000.0, SR, 0.5, 0.5)
    assert high_frequency_ratio(low, SR) < 0.05
    assert high_frequency_ratio(high, SR) > 0.95


def test_high_frequency_ratio_of_silence_is_zero():
    assert high_frequency_ratio(np.zeros((1, 512)), SR) == 0.0


def test_describe_without_f0_reports_nan_thd():
    sine = make_sine(220.0, SR, 0.5, 0.5)
    values = describe(sine, np.tanh(4 * sine), SR, f0=None)
    assert np.isnan(values["thd"])
    assert values["crest_drop"] > 0
    # `describe` devolve so o que sai de UM par (seco, molhado). O combinado nao
    # esta aqui de proposito: ele precisa das senoides e da guitarra juntas, e
    # quem as tem e o `sweep_descriptors`.
    assert set(values) == {"thd", "crest_drop", "hf_ratio", "flatness"}


# --- probes -------------------------------------------------------------------
def test_probes_put_the_sines_at_the_guitar_peak():
    # A senoide tem de ler a curva de transferencia no mesmo ponto de operacao
    # que a guitarra; a -26 LUFS os picos sao muito diferentes.
    probes = _probes()
    guitar_peak = float(np.mean([np.max(np.abs(seg)) for seg in probes.guitar]))
    for sine in probes.sines:
        assert float(np.max(np.abs(sine))) == pytest.approx(guitar_peak, rel=1e-3)


def test_build_probes_requires_a_guitar_segment():
    with pytest.raises(ValueError, match="ao menos um segmento"):
        build_probes([], SR)


# --- varredura e casamento ----------------------------------------------------
def test_sweep_is_monotone_for_a_waveshaper_with_growing_gain():
    probes = _probes()
    knobs = list(np.linspace(0.0, 40.0, 9))
    values = sweep_descriptors(lambda seg, knob: _tanh_shaper(knob)(seg), knobs, probes)

    assert set(values) == {"thd", "crest_drop", "hf_ratio", "flatness", "thd_flatness"}
    for name in values:
        assert is_monotone(monotonicity(knobs, values[name])), name


def test_monotonicity_rejects_a_flat_knob():
    # Porteira para arms candidatos: um knob que nao muda o som e rejeitado.
    knobs = list(np.linspace(0.0, 1.0, 9))
    probes = _probes(n_guitar=1)
    values = sweep_descriptors(lambda seg, knob: np.asarray(seg, dtype=np.float64), knobs, probes)
    assert not is_monotone(monotonicity(knobs, values["thd"]))


def test_monotonicity_of_a_degenerate_curve_is_zero():
    assert monotonicity([0, 1], [0.0, 1.0]) == 0.0
    assert monotonicity([0, 1, 2], [float("nan")] * 3) == 0.0


def test_calibration_recovers_a_known_gain():
    # O teste que vale: dado um waveshaper tanh identico ao da referencia, casar o
    # THD dos extremos tem de devolver os mesmos dB. E a versao sintetica da
    # validacao com o `lsp-tanh`.
    probes = _probes(n_guitar=1)
    knobs = list(np.linspace(0.0, 60.0, 61))
    values = sweep_descriptors(lambda seg, knob: _tanh_shaper(knob)(seg), knobs, probes)["thd"]

    for expected in (5.0, 20.0, 40.0):
        target = float(np.interp(expected, knobs, values))
        recovered, reachable = match_descriptor(knobs, values, target)
        assert reachable
        assert recovered == pytest.approx(expected, abs=0.5)


def test_match_descriptor_flags_an_unreachable_target():
    knobs, values = [0.0, 1.0, 2.0, 3.0], [0.0, 0.1, 0.4, 0.9]
    inside, reachable = match_descriptor(knobs, values, 0.25)
    assert reachable and inside == pytest.approx(1.5)

    clamped, reachable = match_descriptor(knobs, values, 1.5)
    assert not reachable and clamped == 3.0


def test_match_descriptor_handles_a_decreasing_curve():
    # `match_descriptor` ordena pelo descritor, entao serve para um descritor que
    # caia com o knob sem precisar de caso especial.
    knob, reachable = match_descriptor([0.0, 1.0, 2.0], [0.9, 0.4, 0.0], 0.65)
    assert reachable and knob == pytest.approx(0.5)


def test_levels_span_the_calibrated_range_uniformly_in_the_knob():
    # Uniforme NO KNOB e nao no descritor: e o que deixa o interior da
    # correspondencia livre e a avaliacao pelo oraculo nao-circular.
    levels = levels_from_range(5.0, 40.0, n_levels=8)
    assert levels[0] == 5.0 and levels[-1] == 40.0
    steps = np.diff(levels)
    assert steps == pytest.approx([steps[0]] * len(steps))


def test_a_decreasing_knob_passes_the_gate():
    # Um knob monotono decrescente e utilizavel: basta inverter. Existe de
    # verdade -- o Tube Screamer do BYOD perde THD conforme o ganho de entrada
    # sobe. So o nao-monotono e que deve ser rejeitado.
    knobs = [0.0, 1.0, 2.0, 3.0]
    assert monotonicity(knobs, [0.9, 0.6, 0.3, 0.0]) == pytest.approx(-1.0)
    assert is_monotone(monotonicity(knobs, [0.9, 0.6, 0.3, 0.0]))
    assert not is_monotone(monotonicity(knobs, [0.0, 0.9, 0.1, 0.5]))


def test_monotonicity_does_not_warn_on_a_constant_descriptor():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert monotonicity([0.0, 1.0, 2.0], [0.4, 0.4, 0.4]) == 0.0


def test_common_range_is_the_intersection_of_what_the_arms_reach():
    # Sem a intersecao, um arm mais suave teria varios niveis grampeados no mesmo
    # knob e a grade degeneraria: 8 niveis dos quais so alguns soam diferentes.
    sweeps = {
        "ref": {"thd": [0.01, 0.20, 0.43]},
        "suave": {"thd": [0.05, 0.12, 0.21]},
        "duro": {"thd": [0.00, 0.30, 0.60]},
    }
    assert common_range(sweeps, "thd", ["ref", "suave", "duro"]) == (0.05, 0.21)
    # Um arm rejeitado nao estreita a faixa dos demais.
    assert common_range(sweeps, "thd", ["ref", "duro"]) == (0.01, 0.43)


def test_measuring_at_the_levels_beats_interpolating_on_a_knee():
    # Motivo de `measure_at` existir: num waveshaper com joelho (hard clip), a
    # interpolacao linear numa malha grossa subestima o descritor perto do
    # inicio da distorcao o bastante para o dB equivalente ser grampeado.
    probes = _probes(n_guitar=1)

    def hard_clip(segment, knob):
        gain = 10.0 ** (knob / 20.0)
        return np.clip(gain * np.asarray(segment, dtype=np.float64), -0.3, 0.3)

    coarse = list(np.linspace(0.0, 48.0, 33))
    values = np.asarray(sweep_descriptors(hard_clip, coarse, probes)["thd"])
    levels = levels_from_range(10.0, 20.0, n_levels=5)

    interpolated = np.interp(levels, coarse, values)
    measured = np.asarray(sweep_descriptors(hard_clip, levels, probes)["thd"])
    assert np.max(np.abs(measured - interpolated)) > 1e-4


def test_bracket_finds_the_adjacent_pair_that_straddles_the_target():
    knobs, values = [0.0, 1.0, 2.0, 3.0], [0.0, 0.1, 0.4, 0.9]
    assert bracket(knobs, values, 0.25) == (1.0, 2.0)
    assert bracket(knobs, values, 0.05) == (0.0, 1.0)


def test_bracket_of_an_out_of_range_target_is_the_nearest_interval():
    knobs, values = [0.0, 1.0, 2.0, 3.0], [0.0, 0.1, 0.4, 0.9]
    assert bracket(knobs, values, 5.0) == (2.0, 3.0)


def test_bracket_handles_a_decreasing_curve():
    assert bracket([0.0, 1.0, 2.0], [0.9, 0.4, 0.0], 0.6) == (0.0, 1.0)


def test_refine_beats_interpolation_on_a_knee():
    # O caso que motivou `refine_knob`: uma curva que sai de zero abruptamente.
    # A interpolacao linear numa malha grossa erra feio; a bissecao acerta.
    def curve(knob):
        return 0.0 if knob < 10.0 else (knob - 10.0) ** 0.5

    coarse = list(np.linspace(0.0, 20.0, 11))  # passo de 2,0
    values = [curve(knob) for knob in coarse]
    target = 0.5

    interpolated, _ = match_descriptor(coarse, values, target)
    refined, achieved = refine_knob(curve, *bracket(coarse, values, target), target, iterations=20)

    assert achieved == pytest.approx(target, abs=1e-3)
    assert abs(curve(refined) - target) < abs(curve(interpolated) - target)


def test_refine_is_exact_on_a_straight_line():
    refined, achieved = refine_knob(lambda knob: 2.0 * knob, 0.0, 10.0, 7.0, iterations=30)
    assert refined == pytest.approx(3.5, abs=1e-3)
    assert achieved == pytest.approx(7.0, abs=1e-3)


def test_refine_works_on_a_decreasing_curve():
    _, achieved = refine_knob(lambda knob: 10.0 - knob, 0.0, 10.0, 3.0, iterations=30)
    assert achieved == pytest.approx(3.0, abs=1e-3)


# --- planicidade espectral ----------------------------------------------------
def test_flatness_is_low_for_a_pure_tone_and_high_for_noise():
    # E a definicao: senoide concentra toda a energia num bin (media geometrica
    # perto de zero), ruido branco espalha (as duas medias coincidem).
    sr = 44100
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440.0 * t)
    noise = np.random.default_rng(0).standard_normal(sr)
    assert cal.spectral_flatness(tone, sr) < 1e-3
    assert cal.spectral_flatness(noise, sr) > 0.3


def test_flatness_rises_with_clipping():
    # O que a torna util como descritor de quantidade de distorcao: cortar a
    # senoide preenche os vales entre os harmonicos.
    sr = 44100
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 220.0 * t)
    suave = np.tanh(1.5 * tone)
    duro = np.clip(20.0 * tone, -1.0, 1.0)
    assert cal.spectral_flatness(tone, sr) < cal.spectral_flatness(suave, sr)
    assert cal.spectral_flatness(suave, sr) < cal.spectral_flatness(duro, sr)


def test_flatness_ignores_level():
    # Nao pode depender de ganho: os renders sao normalizados a -26 LUFS, mas os
    # probes da calibracao nao passam por essa normalizacao.
    sr = 22050
    t = np.arange(sr) / sr
    x = np.tanh(3.0 * np.sin(2 * np.pi * 330.0 * t))
    assert cal.spectral_flatness(x, sr) == pytest.approx(
        cal.spectral_flatness(0.01 * x, sr), rel=1e-9
    )


def test_flatness_of_silence_is_zero_not_nan():
    assert cal.spectral_flatness(np.zeros(1000), 22050) == 0.0


def test_describe_includes_flatness_as_absolute_value():
    sr = 22050
    t = np.arange(sr) / sr
    dry = np.sin(2 * np.pi * 220.0 * t)
    wet = np.clip(5.0 * dry, -1.0, 1.0)
    values = cal.describe(dry, wet, sr, f0=220.0)
    # Absoluta do molhado, nao diferenca: bate com a funcao pura.
    assert values["flatness"] == pytest.approx(cal.spectral_flatness(wet, sr))


# --- descritor combinado ------------------------------------------------------
def _probes_for_combo(sr=22050):
    t = np.arange(sr) / sr
    guitar = [np.sin(2*np.pi*110.0*k*t)/k for k in (1, 2)]
    guitar = [(0.3 * g / np.max(np.abs(g))).astype(np.float32).reshape(1, -1) for g in guitar]
    return cal.build_probes(guitar, sr, seconds=1.0)


def test_combined_descriptor_is_the_geometric_mean_of_its_parts():
    pr = _probes_for_combo()
    render = lambda seg, k: np.tanh(float(k) * np.asarray(seg))
    out = cal.sweep_descriptors(render, [3.0], pr, only=("thd", "flatness", "thd_flatness"))
    dry = float(np.mean([cal.spectral_flatness(g, pr.sr) for g in pr.guitar]))
    esperado = np.sqrt(out["thd"][0] * (out["flatness"][0] / dry))
    assert out["thd_flatness"][0] == pytest.approx(esperado)


def test_combined_descriptor_needs_both_probe_kinds():
    # Pedir so o combinado tem de renderizar senoides E guitarra: se `only`
    # pulasse um dos dois, o descritor sairia zerado sem erro nenhum.
    pr = _probes_for_combo()
    render = lambda seg, k: np.tanh(float(k) * np.asarray(seg))
    out = cal.sweep_descriptors(render, [4.0], pr, only=("thd_flatness",))
    assert out["thd_flatness"][0] > 0.0


def test_combined_descriptor_is_monotone_where_each_part_saturates():
    # A razao de existir do combinado: cada parte deita numa ponta do eixo.
    pr = _probes_for_combo()
    render = lambda seg, k: np.tanh(float(k) * np.asarray(seg))
    knobs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    out = cal.sweep_descriptors(render, knobs, pr, only=("thd_flatness",))
    assert cal.monotonicity(knobs, out["thd_flatness"]) == pytest.approx(1.0)


def test_combined_descriptor_is_one_scale_not_two():
    # A planicidade entra normalizada pela do seco, entao um arm transparente da
    # razao 1 e o combinado vira sqrt(thd) -- mesma escala do THD, sem unidade.
    pr = _probes_for_combo()
    identidade = lambda seg, k: np.asarray(seg, dtype=float)
    out = cal.sweep_descriptors(identidade, [1.0], pr, only=("thd", "flatness", "thd_flatness"))
    dry = float(np.mean([cal.spectral_flatness(g, pr.sr) for g in pr.guitar]))
    assert out["flatness"][0] / dry == pytest.approx(1.0, rel=1e-6)
    assert out["thd_flatness"][0] == pytest.approx(np.sqrt(out["thd"][0]), rel=1e-6)


# --- posicionamento dos niveis interiores --------------------------------------
def test_levels_from_targets_hits_each_target_independently():
    # Waveshaper sintetico com curva torta: interpolar erra, medir acerta.
    knobs = list(np.linspace(0.0, 10.0, 21))
    curva = lambda k: k ** 3                      # joelho forte perto de zero
    values = [curva(k) for k in knobs]
    alvos = [8.0, 125.0, 512.0]
    achados, alcancavel = cal.levels_from_targets(curva, knobs, values, alvos)
    assert all(alcancavel)
    for knob, alvo in zip(achados, alvos):
        assert curva(knob) == pytest.approx(alvo, rel=0.02)


def test_levels_from_targets_flags_what_is_out_of_reach():
    knobs = list(np.linspace(0.0, 4.0, 9))
    curva = lambda k: 2.0 * k
    values = [curva(k) for k in knobs]
    achados, alcancavel = cal.levels_from_targets(curva, knobs, values, [1.0, 99.0])
    assert alcancavel == [True, False]
    # Fora do alcance, grampeia no extremo mais proximo em vez de extrapolar.
    assert achados[1] == pytest.approx(4.0)


def test_level_mode_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="level_mode"):
        cal.calibrate_arms(level_mode="qualquer")
