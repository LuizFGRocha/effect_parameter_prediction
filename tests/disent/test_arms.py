"""O roster como dado: o que o YAML precisa ter para o render nao gastar horas a toa."""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.arms import (
    DEFAULT_ROSTER,
    TONE_CUTOFF_HZ,
    TONE_LEVELS,
    apply_tone,
    load_roster,
    parse_roster,
    tone_cutoff_hz,
)


def _data(**arms):
    base = {
        "ref": {"stratum": "S1", "backend": "pedalboard", "drive_param": "drive_db",
                "levels": [5.0, 10.0, 15.0]},
        "outro": {"stratum": "S2", "path": "x.vst3", "drive_param": "gain",
                  "levels": [0.1, 0.2, 0.3]},
    }
    base.update(arms)
    return {"reference": "ref", "arms": base}


def test_the_versioned_roster_loads():
    roster = load_roster(DEFAULT_ROSTER)
    assert roster.reference in roster.keys()
    assert roster.drive_levels >= 2


def test_the_reference_levels_are_the_common_scale():
    roster = parse_roster(_data())
    assert roster.reference_levels() == (5.0, 10.0, 15.0)
    assert roster.drive_levels == 3


def test_arms_with_different_level_counts_are_refused():
    with pytest.raises(ValueError, match="mesmo numero de niveis"):
        parse_roster(_data(outro={"stratum": "S2", "path": "x.vst3", "drive_param": "g",
                                  "levels": [0.1, 0.2]}))


def test_a_reference_outside_the_roster_is_refused():
    with pytest.raises(ValueError, match="referencia"):
        parse_roster({**_data(), "reference": "nao-existe"})


def test_a_vst_arm_without_a_path_is_refused():
    with pytest.raises(ValueError, match="path"):
        parse_roster(_data(outro={"stratum": "S2", "drive_param": "g",
                                  "levels": [0.1, 0.2, 0.3]}))


def test_a_typo_in_a_field_name_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="desconhecidos"):
        parse_roster(_data(outro={"stratum": "S2", "path": "x.vst3", "drive_param": "g",
                                  "levels": [0.1, 0.2, 0.3], "fxied": {}}))


@pytest.mark.parametrize("missing", ["stratum", "drive_param", "levels"])
def test_a_required_field_missing_is_refused(missing):
    spec = {"stratum": "S2", "path": "x.vst3", "drive_param": "g", "levels": [1, 2, 3]}
    del spec[missing]
    with pytest.raises(ValueError, match=missing):
        parse_roster(_data(outro=spec))


def test_an_unknown_arm_is_refused_by_name():
    with pytest.raises(KeyError, match="nao esta no roster"):
        parse_roster(_data()).arm("z")


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
