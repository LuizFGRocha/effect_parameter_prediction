"""Invariantes do roster de implementacoes do POC II."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.arms import (
    ARMS,
    ARMS_BY_KEY,
    FINGERPRINT_RTOL,
    REJECTED_ARMS,
    DRIVE_LEVELS,
    REFERENCE_ARM,
    TONE_CUTOFF_HZ,
    TONE_LEVELS,
    apply_tone,
    arm,
    arm_keys,
    arms_in_stratum,
    tone_cutoff_hz,
)


def test_reference_arm_comes_first_and_is_pedalboard():
    assert arm_keys()[0] == REFERENCE_ARM
    assert arm(REFERENCE_ARM).backend == "pedalboard"
    # A faixa da referencia e a do catalogo do POC I; e ela que define a unidade
    # de leitura "drive-tanh equivalente em dB".
    assert arm(REFERENCE_ARM).sweep_range == (5.0, 40.0)


def test_strata_partition_the_roster():
    strata = [arms_in_stratum(name) for name in ("S1", "S2", "S3")]
    assert sum(len(group) for group in strata) == len(ARMS)
    assert sorted(key for group in strata for key in group) == sorted(arm_keys())


def test_keys_are_unique():
    assert len(ARMS_BY_KEY) == len(ARMS)


def test_arm_raises_on_unknown_key():
    with pytest.raises(KeyError, match="nao existe"):
        arm("nao-existe")


@pytest.mark.parametrize("spec", ARMS, ids=lambda spec: spec.key)
def test_specs_are_well_formed(spec):
    assert spec.backend in {"pedalboard", "vst"}
    assert spec.stratum in {"S1", "S2", "S3"}
    lo, hi = spec.sweep_range
    assert lo < hi
    if spec.backend == "vst":
        assert spec.path.endswith(".vst3")
        assert spec.plugin_spec()["fixed"] is not spec.fixed  # copia, nao alias
    else:
        assert spec.path is None
        with pytest.raises(ValueError, match="nao e VST3"):
            spec.plugin_spec()


def test_lsp_arms_differ_only_in_the_sigmoid():
    # E o que torna o S2 uma ablacao da FORMA da nao-linearidade: mesmo plugin,
    # mesmo ganho de entrada, so a sigmoide muda.
    lsp = [spec for spec in ARMS if spec.key.startswith("lsp-")]
    assert len(lsp) >= 2
    sigmoids = {spec.fixed["clipper_sigmoid_function"] for spec in lsp}
    assert len(sigmoids) == len(lsp)
    for spec in lsp:
        assert spec.drive_param == "input_gain_db"
        assert spec.drive_in_db is True
        rest = {k: v for k, v in spec.fixed.items() if k != "clipper_sigmoid_function"}
        assert rest == {
            k: v for k, v in lsp[0].fixed.items() if k != "clipper_sigmoid_function"
        }


def test_only_db_arms_are_flagged_as_such():
    # `drive_in_db` autoriza reportar erro em dB nativo; os arms do S3 nao podem.
    assert {spec.key for spec in ARMS if spec.drive_in_db} == {
        "pedalboard-tanh", "lsp-tanh", "lsp-hardclip", "lsp-arctan", "lsp-sine"
    }


def test_native_tone_controls_are_pinned_neutral():
    # Decisao de desenho: o tone e um estagio nosso, o dos plugins fica neutro.
    # Nos arms do BYOD isso e o `dry_wet` em 100 e o `out_gain` em 0: o circuito
    # entra inteiro e sem ganho de saida, e nada colore alem da nao-linearidade.
    for key in arms_in_stratum("S3"):
        assert arm(key).fixed["dry_wet"] == 100.0
        assert arm(key).fixed["out_gain"] == 0.0


def test_byod_arms_are_identified_by_audio_not_by_name():
    # A guarda antiga comparava o nome exibido, que o plugin atualiza antes de o
    # DSP trocar de circuito -- ela passava enquanto o audio vinha de outro pedal.
    for key in arms_in_stratum("S3"):
        spec = arm(key)
        assert spec.raw_fixed["program"] is not None
        assert spec.fingerprint is not None, f"{key} sem assinatura de audio"
        assert len(spec.fingerprint) == 3


def test_byod_fingerprints_separate_every_circuit():
    # Se duas assinaturas caissem dentro da tolerancia, a guarda deixaria passar
    # um circuito pelo outro -- que e exatamente a falha que ela existe para pegar.
    specs = [arm(key) for key in arms_in_stratum("S3")]
    for i, a in enumerate(specs):
        for b in specs[i + 1:]:
            assert not np.allclose(a.fingerprint, b.fingerprint, rtol=FINGERPRINT_RTOL), (
                f"{a.key} e {b.key} tem assinaturas indistinguiveis"
            )


# --- estagio de tone ----------------------------------------------------------
def test_tone_cutoff_hits_the_declared_endpoints():
    assert tone_cutoff_hz(0) == pytest.approx(TONE_CUTOFF_HZ[0])
    assert tone_cutoff_hz(TONE_LEVELS - 1) == pytest.approx(TONE_CUTOFF_HZ[1])


def test_tone_cutoff_is_log_spaced():
    cutoffs = [tone_cutoff_hz(level) for level in range(TONE_LEVELS)]
    ratios = [b / a for a, b in zip(cutoffs, cutoffs[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_tone_cutoff_rejects_levels_out_of_range():
    with pytest.raises(ValueError, match="fora de"):
        tone_cutoff_hz(TONE_LEVELS)
    with pytest.raises(ValueError, match="fora de"):
        tone_cutoff_hz(-1)


def test_apply_tone_attenuates_more_at_lower_cutoff():
    sr = 44100
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal((1, sr)) * 0.1).astype(np.float32)
    energies = [
        float(np.sum(apply_tone(signal, sr, tone_cutoff_hz(level)) ** 2))
        for level in range(TONE_LEVELS)
    ]
    assert energies == sorted(energies)


def test_drive_and_tone_levels_form_the_declared_grid():
    # 8x5: o drive tem MAIS niveis que o tone porque e o eixo em disputa; o tone
    # e controle positivo. O produto continua 40, ou seja, o mesmo custo de render
    # da grade anterior (5x8) -- a forma mudou, o tamanho nao.
    assert (DRIVE_LEVELS, TONE_LEVELS) == (8, 5)
    assert DRIVE_LEVELS * TONE_LEVELS == 40


def test_rejected_arms_are_documented_and_not_in_the_roster():
    # A medicao que reprovou cada um custou caro; o motivo fica registrado para
    # nao ser refeito.
    for key, reason in REJECTED_ARMS.items():
        assert key not in ARMS_BY_KEY
        assert len(reason) > 20


# --- vies perceptual residual -------------------------------------------------
def test_perceptual_bias_only_documents_arms_that_exist():
    from gefx.disent.arms import PERCEPTUAL_BIAS
    for key in PERCEPTUAL_BIAS:
        assert key in ARMS_BY_KEY, f"{key} nao esta no roster"


def test_perceptual_bias_records_what_the_report_needs():
    # Um numero solto nao serve: sem n, IC e metodo ninguem consegue julgar o
    # peso da medida nem repeti-la.
    from gefx.disent.arms import PERCEPTUAL_BIAS
    for key, info in PERCEPTUAL_BIAS.items():
        assert set(info) >= {"bias_levels", "ci95_levels", "n_trials",
                             "n_listeners", "measured_on", "method", "sign"}, key
        lo, hi = info["ci95_levels"]
        assert lo <= info["bias_levels"] <= hi, f"{key}: vies fora do proprio IC"
        assert info["n_trials"] > 0 and info["n_listeners"] > 0


def test_perceptual_bias_is_not_applied_to_the_knobs():
    # A grade tem de continuar reproduzivel so a partir do codigo: o vies e
    # RELATADO, nunca consumido. A calibracao pode mencionar `PERCEPTUAL_BIAS`
    # (ela o copia para o JSON), mas as funcoes que decidem valor de knob nao.
    # Se alguem aplicar a correcao um dia, que seja decisao declarada e nao
    # efeito colateral de uma edicao aqui.
    import inspect
    from gefx.disent import calibrate

    for func in (calibrate.levels_from_range, calibrate.levels_from_targets,
                 calibrate.refine_knob, calibrate.match_descriptor,
                 calibrate.bracket, calibrate.common_range):
        fonte = inspect.getsource(func)
        assert "PERCEPTUAL_BIAS" not in fonte and "bias" not in fonte, (
            f"{func.__name__} passou a depender do vies de escuta; isso tira a "
            "reprodutibilidade da grade a partir do codigo"
        )

    # E o relatorio tem de continuar carregando o dado, senao ele se perde.
    assert "PERCEPTUAL_BIAS" in inspect.getsource(calibrate.calibrate_arms)
