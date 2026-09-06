"""Adaptador de VST3. So o que roda sem plugin: parsing de texto, clamp e mono/estereo."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gefx.effects import vst_adapter
from gefx.effects.vst_adapter import display_as, process_mono, set_parameter


class FakeParam:
    def __init__(self, min_value=0.0, max_value=1.0, valid_values=()):
        self.min_value = min_value
        self.max_value = max_value
        self.valid_values = list(valid_values)
        self.raw_value = 0.0


class FakePlugin:
    """Dublê com o mesmo formato do plugin do pedalboard: `.parameters` + atributos."""

    def __init__(self, parameters, explode=()):
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "written", {})
        object.__setattr__(self, "_explode", set(explode))

    def __setattr__(self, name, value):
        if name in self._explode:
            raise RuntimeError(f"plugin recusou {name}")
        self.written[name] = value


@pytest.mark.parametrize(
    "text, unit, expected",
    [
        ("200.00 ms", "ms", 200.0),
        ("1.50 s", "ms", 1500.0),
        ("35%", "%", 35.0),
        ("-12.5 dB", "dB", -12.5),
        ("  8 ", "ms", 8.0),
        ("1.50 s", "%", 1.5),  # so a unidade "ms" dispara a conversao s->ms
    ],
)
def test_display_as(text, unit, expected):
    assert display_as(text, unit) == pytest.approx(expected)


def test_display_as_empty_is_nan():
    assert math.isnan(display_as("", "ms"))
    assert math.isnan(display_as("sem numero", "ms"))


def test_display_as_breaks_on_scientific_notation():
    # CARACTERIZACAO: o filtro de caracteres transforma "1.5e-3" em "1.5-3".
    # Nenhum plugin do registro atual exibe assim, mas um que exibisse quebraria.
    with pytest.raises(ValueError):
        display_as("1.5e-3 s", "ms")


def test_set_parameter_writes_through():
    plugin = FakePlugin({"rate_hz": FakeParam(0.0, 20.0)})
    set_parameter(plugin, "rate_hz", 3.0)
    assert plugin.written == {"rate_hz": 3.0}


def test_set_parameter_ignores_unknown_name(capsys):
    plugin = FakePlugin({"rate_hz": FakeParam()})
    set_parameter(plugin, "nao_existe", 1.0)
    assert plugin.written == {}
    assert "nao existe neste plugin" in capsys.readouterr().out


def test_set_parameter_clamps_into_range():
    plugin = FakePlugin({"depth": FakeParam(0.0, 1.0)})
    set_parameter(plugin, "depth", 5.0)
    set_parameter(plugin, "depth_low", 5.0)  # inexistente, so para nao poluir
    assert plugin.written["depth"] == 1.0

    set_parameter(plugin, "depth", -2.0)
    assert plugin.written["depth"] == 0.0


def test_set_parameter_warns_once_per_message(capsys):
    plugin = FakePlugin({"depth": FakeParam(0.0, 1.0)})
    set_parameter(plugin, "depth", 5.0)
    set_parameter(plugin, "depth", 5.0)
    assert capsys.readouterr().out.count("fora de") == 1
    assert len(vst_adapter._WARNED) == 1


def test_set_parameter_passes_non_numeric_values_untouched():
    plugin = FakePlugin({"mode": FakeParam(min_value="a", max_value="z")})
    set_parameter(plugin, "mode", "Traditional")
    assert plugin.written == {"mode": "Traditional"}


def test_set_parameter_swallows_plugin_errors(capsys):
    plugin = FakePlugin({"depth": FakeParam(0.0, 1.0)}, explode={"depth"})
    set_parameter(plugin, "depth", 0.5)  # nao levanta
    assert "falhou setar" in capsys.readouterr().out


def test_set_parameter_by_display_picks_nearest_valid_value():
    param = FakeParam(valid_values=["0.00 ms", "100.00 ms", "200.00 ms"])
    plugin = FakePlugin({"node_1_delay": param})
    set_parameter(plugin, "node_1_delay", 105.0)
    # indice 1 de 3 -> raw 1/(3-1)
    assert param.raw_value == pytest.approx(0.5)
    assert plugin.written == {}  # enderecado por raw_value, nao por setattr


def test_process_mono_passes_mono_through():
    segment = np.arange(6, dtype=np.float32).reshape(1, 6)
    captured = {}

    def plugin(audio, sr, reset):
        captured["shape"] = audio.shape
        captured["reset"] = reset
        return audio * 2

    out = process_mono(plugin, False, segment, 44100)
    assert captured["shape"] == (1, 6)
    assert captured["reset"] is True
    assert np.array_equal(out, segment * 2)


def test_process_mono_duplicates_and_averages_for_stereo_plugins():
    segment = np.arange(6, dtype=np.float32).reshape(1, 6)

    def plugin(audio, sr, reset):
        assert audio.shape == (2, 6)
        return audio * np.array([[1.0], [3.0]])  # canais diferentes

    out = process_mono(plugin, True, segment, 44100)
    assert out.shape == (1, 6)
    assert np.allclose(out, segment * 2.0)  # media de 1x e 3x
