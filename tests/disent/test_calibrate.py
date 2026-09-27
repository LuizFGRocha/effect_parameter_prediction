"""Pareamento pelo Rnonlin e o servidor do pareamento de ouvido, sem plugin."""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from gefx.disent.arms import Arm
from gefx.disent.calibrate import arm_curve, arm_knobs, knob_for, rnonlin
from gefx.disent.tune import slider_bounds, wav_bytes

SR = 44100


def _note(seconds=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.exp(-3.0 * t) * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)


def test_rnonlin_is_one_without_distortion_whatever_the_gain_or_latency():
    x = _note() + 0.01 * np.random.default_rng(0).standard_normal(int(0.5 * SR))
    assert rnonlin(x, x, SR) == pytest.approx(1.0, abs=1e-6)
    assert rnonlin(x, 0.1 * x, SR) == pytest.approx(1.0, abs=1e-6)
    assert rnonlin(x, np.roll(x, 200), SR) == pytest.approx(1.0, abs=1e-3)


def test_rnonlin_falls_with_the_reference_drive():
    arm = Arm(key="ref", stratum="S1", drive_param="drive_db", backend="pedalboard",
              sweep=(0.0, 40.0))
    knobs, curve = arm_curve(arm, [_note().reshape(1, -1)], SR, points=5)
    assert list(knobs) == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert np.all(np.diff(curve) < 0)


def test_knob_for_interpolates_and_follows_the_first_crossing():
    knobs = [0.0, 10.0, 20.0, 30.0]
    assert knob_for(knobs, [0.0, 2.0, 4.0, 6.0], 3.0) == pytest.approx(15.0)
    # Recuo na curva: vale o envelope, a primeira vez que o alvo e alcancado.
    assert knob_for(knobs, [0.0, 4.0, 3.0, 6.0], 3.5) == pytest.approx(8.75)


def test_knob_for_refuses_a_target_beyond_the_reach():
    with pytest.raises(ValueError, match="alcance"):
        knob_for([0.0, 1.0], [0.0, 1.0], 2.0)


def test_an_arm_that_reaches_everything_has_nothing_flagged():
    values, unmatched = arm_knobs([0.0, 10.0], [0.0, 10.0], [2.0, 8.0])
    assert values == pytest.approx([2.0, 8.0])
    assert unmatched == []


def test_targets_above_the_reach_go_to_the_end_of_the_knob_and_are_flagged():
    values, unmatched = arm_knobs([0.0, 10.0, 20.0, 30.0], [0.0, 2.0, 4.0, 4.0],
                                  [1.0, 3.0, 5.0, 6.0])
    assert values == pytest.approx([5.0, 15.0, 22.5, 30.0])
    assert unmatched == [3, 4]


def test_targets_below_the_floor_go_to_the_start_of_the_knob_and_are_flagged():
    # O MXR nunca fica tao limpo quanto o nivel 1 da referencia.
    values, unmatched = arm_knobs([0.0, 10.0, 20.0], [2.0, 4.0, 6.0], [1.0, 3.0, 5.0])
    assert values == pytest.approx([0.0, 5.0, 15.0])
    assert unmatched == [1]


def test_the_noisy_stretch_below_the_cleanest_knob_is_ignored():
    # Distorcao caindo ate o knob 10 e ruido, nao drive: o alvo 1,5 casa depois dele.
    values, unmatched = arm_knobs([0.0, 10.0, 20.0, 30.0], [2.0, 1.0, 2.0, 4.0], [1.5, 3.0])
    assert values == pytest.approx([15.0, 25.0])
    assert unmatched == []


def test_an_arm_that_reaches_nothing_spreads_over_the_whole_knob():
    values, unmatched = arm_knobs([0.0, 10.0], [0.0, 1.0], [2.0, 3.0, 4.0])
    assert values == pytest.approx([0.0, 5.0, 10.0])
    assert unmatched == [1, 2, 3]


def test_each_slider_spans_its_neighbours():
    assert slider_bounds([1.0, 2.0, 3.0], (0.0, 5.0)) == [(0.0, 2.0), (1.0, 3.0), (2.0, 5.0)]


def test_wav_bytes_is_a_mono_16_bit_wav():
    with wave.open(io.BytesIO(wav_bytes(_note(), SR))) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, SR)
        assert handle.getnframes() == _note().shape[0]
