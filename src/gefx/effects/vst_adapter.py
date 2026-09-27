"""Adaptador para os plugins VST3 de terceiros.

Concentra as manhas descobertas na validacao e que precisam ser preservadas:

- A primeira chamada de processamento depois do `load_plugin` ainda sai com os
  parametros antigos (verificado no LSP: o atraso pedido nao aparece na saida),
  entao ela e gasta com silencio no `load_arm`.
- Alguns plugins exigem entrada estereo; isso e detectado pelo `ValueError` de
  uma sondagem mono.
- Alguns parametros nao expoem faixa numerica e so sao enderecaveis pelo texto
  exibido (`BY_DISPLAY`).
- A primeira instancia de alguns plugins num processo pode sair NaN a partir da
  amostra 8192 de cada chamada, para sempre (o Fuzz da VZtec, em ~metade dos
  processos, sem padrao de tempo ou de concorrencia). A sondagem do `load_arm`
  pega isso; a instancia ruim fica viva em `_BROKEN` e carrega-se outra --
  descarta-la antes nao adianta, a seguinte tende a sair ruim tambem.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Parametro -> unidade em que a conversao entrega o numero, para os que so
# aceitam ser escolhidos pelo texto exibido.
BY_DISPLAY = {"node_1_delay": "ms", "node_1_feedback": "%"}

_WARNED: Set[str] = set()

# Instancias que sairam NaN na sondagem: mantidas vivas de proposito (ver o docstring).
_BROKEN: List[Any] = []
LOAD_ATTEMPTS = 3


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"\n      [aviso] {message}")


def display_as(text: Any, unit: str) -> float:
    """'200.00 ms' -> 200.0 quando unit='ms'; '1.50 s' -> 1500.0; '35%' -> 35.0."""
    text = str(text).strip()
    number = float("".join(c for c in text if c.isdigit() or c in ".-") or "nan")
    if unit == "ms" and text.endswith("s") and not text.endswith("ms"):
        number *= 1000.0
    return number


def _raw_for_display(param, value: float, unit: str) -> float:
    values = [display_as(candidate, unit) for candidate in param.valid_values]
    return int(np.argmin(np.abs(np.asarray(values) - value))) / (len(values) - 1)


def set_parameter(plugin, name: str, value: Any) -> None:
    """Escreve um parametro, tratando faixa, clamp e enderecamento por texto."""
    if name not in plugin.parameters:
        print(f"      [aviso] {name!r} nao existe neste plugin")
        return

    param = plugin.parameters[name]
    if name in BY_DISPLAY and isinstance(value, (int, float)):
        param.raw_value = _raw_for_display(param, float(value), BY_DISPLAY[name])
        return

    lo, hi = param.min_value, param.max_value
    if isinstance(value, (int, float)) and isinstance(lo, (int, float)):
        clamped = min(max(float(value), lo), hi)
        if clamped != value:
            _warn_once(f"{name}: {value:.4g} fora de [{lo}, {hi}], usando {clamped:.4g}")
            value = clamped
    try:
        setattr(plugin, name, value)
    except Exception as exc:
        print(f"      [aviso] falhou setar {name}={value!r}: {exc}")


def load_arm(spec: Dict[str, Any], sr: int = 44100) -> Tuple[Any, bool]:
    """Carrega o plugin, fixa os parametros neutros e devolve (plugin, estereo).

    Gasta aqui a primeira chamada de processamento, que sairia com os parametros
    antigos, descobre de passagem se o plugin exige entrada estereo e troca uma
    instancia que sai NaN por outra.
    """
    for _ in range(LOAD_ATTEMPTS):
        plugin, stereo, finite = _load_once(spec, sr)
        if finite:
            return plugin, stereo
        _BROKEN.append(plugin)
    raise RuntimeError(f"{spec['path']}: {LOAD_ATTEMPTS} instancias seguidas sairam NaN")


def _load_once(spec: Dict[str, Any], sr: int) -> Tuple[Any, bool, bool]:
    from pedalboard import load_plugin

    kwargs = {"plugin_name": spec["plugin_name"]} if spec.get("plugin_name") else {}
    plugin = load_plugin(spec["path"], **kwargs)
    for name, value in spec["fixed"].items():
        set_parameter(plugin, name, value)

    # 0,25 s passa das 8192 amostras em que o NaN aparece (ver o docstring).
    dummy = np.random.default_rng(0).standard_normal((1, int(0.25 * sr))).astype(np.float32) * 0.05
    try:
        plugin(dummy, sr, reset=True)
        stereo = False
    except ValueError:
        dummy = np.repeat(dummy, 2, axis=0)
        stereo = True
    out = plugin(dummy, sr, reset=True)
    return plugin, stereo, bool(np.all(np.isfinite(out)))


def process_mono(plugin, stereo: bool, segment: np.ndarray, sr: int) -> np.ndarray:
    """Processa um segmento mono, duplicando canais se o plugin exigir estereo."""
    audio = np.repeat(segment, 2, axis=0) if stereo else segment
    out = plugin(audio, sr, reset=True)
    return out.mean(axis=0, keepdims=True) if stereo else out
