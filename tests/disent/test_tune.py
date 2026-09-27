"""A pagina de escuta, sem plugin: o que ela serve e que nao grava nada."""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

import gefx.disent.tune as tune
from gefx.disent.arms import write_levels

SR = 44100
ROSTER = """
reference: ref
arms:
  ref: {stratum: S1, backend: pedalboard, drive_param: drive_db, sweep: [0.0, 60.0]}
  a: {stratum: S2, path: a.vst3, drive_param: g}
  b: {stratum: S3, path: b.vst3, drive_param: g}
"""
LEVELS = {"ref": [10.0, 20.0], "a": [1.0, 2.0], "b": [3.0, 4.0]}


@pytest.fixture
def tuner(monkeypatch, tmp_path):
    """Um Tuner com dubles no lugar dos plugins; `rendered` guarda os knobs pedidos."""
    rendered = []

    class FakeLoaded:
        def __init__(self, arm, sr):
            self.arm = arm

        def render(self, segment, sr, knob):
            rendered.append((self.arm.key, knob))
            return segment

    monkeypatch.setattr(tune, "LoadedArm", FakeLoaded)
    monkeypatch.setattr(tune, "load_audio_file",
                        lambda path: (np.full((1, SR), 0.1, dtype=np.float32), SR))
    monkeypatch.setattr(tune, "normalize_loudness", lambda audio, sr: audio)
    roster = tmp_path / "roster.yaml"
    roster.write_text(ROSTER, encoding="utf-8")
    levels = tmp_path / "levels.yaml"
    write_levels(levels, {"descriptor": "rnonlin", "targets": [0.98, 0.9], "arms": {
        key: {"levels": values, "unmatched": [1] if key == "b" else []}
        for key, values in LEVELS.items()}})

    def build():
        return tune.Tuner(tmp_path / "x.wav", roster, levels)

    build.rendered = rendered
    build.levels = levels
    return build


def test_wav_bytes_is_a_mono_16_bit_wav():
    audio = np.zeros(SR // 2, dtype=np.float32)
    with wave.open(io.BytesIO(tune.wav_bytes(audio, SR))) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, SR)
        assert handle.getnframes() == SR // 2


def test_the_state_shows_the_calibrated_levels_and_targets(tuner):
    state = tuner().state()
    assert state["reference"] == "ref"
    assert state["targets"] == [0.98, 0.9]
    arms = {arm["key"]: arm for arm in state["arms"]}
    assert arms["b"]["levels"] == [3.0, 4.0] and arms["b"]["unmatched"] == [1]
    assert arms["a"]["stratum"] == "S2"


def test_a_level_renders_at_the_knob_from_the_levels_file(tuner):
    listener = tuner()
    listener.render("b", 1)
    listener.render("b", 1)  # do cache
    assert tuner.rendered == [("b", 4.0)]


def test_listening_never_writes_the_levels_file(tuner):
    before = tuner.levels.read_bytes()
    listener = tuner()
    listener.state()
    listener.render("a", 0)
    assert tuner.levels.read_bytes() == before
    assert not hasattr(listener, "save")
