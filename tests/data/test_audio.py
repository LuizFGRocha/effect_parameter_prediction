"""Recorte de segmento, normalizacao de loudness e varredura de wavs.

`select_random_segment` consome o mesmo `Generator` que amostra os parametros, e
a quantidade de saques faz parte do contrato de reproducao do dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.audio import (
    iter_wav_files,
    normalize_loudness,
    select_random_segment,
)

SR = 100


class CountingRng:
    """Envolve um Generator e registra os saques, para fixar a ordem do RNG."""

    def __init__(self, rng):
        self._rng = rng
        self.calls = []

    def integers(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._rng.integers(*args, **kwargs)


def make_audio(frames: int) -> np.ndarray:
    return np.arange(frames, dtype=np.float32).reshape(1, frames)


def test_select_random_segment_length_and_bounds():
    audio = make_audio(1000)
    segment = select_random_segment(audio, SR, np.random.default_rng(0), 2.0, 0.5, 0.5)

    assert segment.shape == (1, 200)
    start = int(segment[0, 0])
    assert 50 <= start <= 750


def test_select_random_segment_consumes_exactly_one_draw():
    rng = CountingRng(np.random.default_rng(0))
    select_random_segment(make_audio(1000), SR, rng, 2.0, 0.5, 0.5)

    assert len(rng.calls) == 1
    # `integers` e inclusivo no limite superior por causa do `+ 1`.
    assert rng.calls[0][0] == (50, 751)


def test_select_random_segment_is_deterministic_per_seed():
    audio = make_audio(1000)
    first = select_random_segment(audio, SR, np.random.default_rng(3), 2.0, 0.5, 0.5)
    second = select_random_segment(audio, SR, np.random.default_rng(3), 2.0, 0.5, 0.5)
    assert np.array_equal(first, second)


def test_select_random_segment_exact_fit_has_a_single_position():
    # 50 ignorados + 200 de segmento + 50 ignorados = 300 frames.
    segment = select_random_segment(make_audio(300), SR, np.random.default_rng(0), 2.0, 0.5, 0.5)
    assert segment.shape == (1, 200)
    assert int(segment[0, 0]) == 50


def test_select_random_segment_rejects_audio_that_is_one_frame_too_short():
    with pytest.raises(ValueError, match="exceeds audio length"):
        select_random_segment(make_audio(299), SR, np.random.default_rng(0), 2.0, 0.5, 0.5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(segment_seconds=0.0), "segment_seconds must be > 0"),
        (dict(segment_seconds=-1.0), "segment_seconds must be > 0"),
        (dict(ignore_start_seconds=-0.1), "must be >= 0"),
        (dict(ignore_end_seconds=-0.1), "must be >= 0"),
        (dict(segment_seconds=0.001), "too small for the current sample rate"),
    ],
)
def test_select_random_segment_guards(kwargs, message):
    call = dict(segment_seconds=2.0, ignore_start_seconds=0.5, ignore_end_seconds=0.5)
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        select_random_segment(make_audio(1000), SR, np.random.default_rng(0), **call)


def test_normalize_loudness_rejects_stereo():
    # CARACTERIZACAO: o `reshape(audio, shape(audio)[1])` assume mono (1, N).
    # Todo o pipeline e mono, mas um wav estereo entrando aqui quebra em vez de
    # ser rebaixado.
    sr = 44100
    stereo = np.zeros((2, sr), dtype=np.float64)
    with pytest.raises(ValueError):
        normalize_loudness(stereo, sr)


def test_iter_wav_files_is_sorted_and_recursive(tmp_path):
    # A ordem e contrato: a lista e zipada com as SeedSequence filhas, entao ela
    # determina qual seed cai em qual arquivo.
    (tmp_path / "b").mkdir()
    for name in ["z.wav", "a.wav", "b/c.wav", "b/a.wav"]:
        (tmp_path / name).touch()
    (tmp_path / "nota.txt").touch()
    (tmp_path / "d.WAV").touch()

    found = [path.relative_to(tmp_path).as_posix() for path in iter_wav_files(tmp_path)]
    assert found == ["a.wav", "b/a.wav", "b/c.wav", "z.wav"]


def test_iter_wav_files_skips_directories_named_like_wavs(tmp_path):
    (tmp_path / "pasta.wav").mkdir()
    (tmp_path / "real.wav").touch()
    assert [path.name for path in iter_wav_files(tmp_path)] == ["real.wav"]
