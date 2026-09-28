"""Registro das implementacoes de terceiros de cada efeito (POC II).

Cada entrada mapeia um parametro de referencia num parametro do plugin, com uma
conversao e um sinalizador `exact`:

- `exact=True`: existe equivalencia fisica, a conversao consome o valor **cru**
  (mesma unidade) e o resultado e lido por MAE.
- `exact=False`: nao existe equivalencia fisica (os `depth`, o `room_size`); a
  conversao consome o valor **normalizado** via `onto(lo, hi)` e o resultado so
  faz sentido por correlacao de Spearman.

ATENCAO: `slapback_delay` e `feedback_delay` ficaram deliberadamente de fora. Nem
o LSP Slapback Delay nem o ChowMatrix passaram na checagem de sanidade (o caminho
wet nao entra na mistura de forma controlavel pelo host), entao os modelos de
delay nao sao avaliados enquanto isso nao for resolvido.
"""
from __future__ import annotations

import math
from typing import Dict, List

LSP = "plugins/real/lsp-plugins.vst3"


def ident(value: float) -> float:
    return value


def to_pct(value: float) -> float:
    return value * 100.0


def to_db(value: float) -> float:
    """Ganho linear -> dB, com piso em -60 dB."""
    return max(20.0 * math.log10(max(value, 1e-4)), -60.0)


def onto(lo: float, hi: float):
    """Mapeia o valor normalizado [0,1] na faixa propria do plugin.

    Usado so onde nao existe equivalencia fisica; marca o parametro como
    nao-exato, e portanto lido por Spearman.
    """
    return lambda value: lo + (hi - lo) * value


# efeito -> braco -> {path, plugin_name?, fixed, map: ref -> (param_plugin, conversao, exato)}
REGISTRY: Dict[str, Dict[str, dict]] = {
    "distortion": {
        "lsp": dict(
            path=LSP, plugin_name="Clipper Mono",
            fixed={"clipper_enable": True, "clipper_sigmoid_function": "Hyperbolic tangent",
                   "enable_input_lufs_limitation": False, "overdrive_protection": False,
                   "boosting_mode": False, "output_gain_db": 0.0, "dithering_mode": "None"},
            map={"drive_db": ("input_gain_db", ident, True)},
        ),
        "chowcentaur": dict(
            path="plugins/real/ChowCentaur.vst3",
            fixed={"mode": "Traditional", "level": 0.5, "treble": 0.5},
            map={"drive_db": ("gain", onto(0.0, 1.0), False)},
        ),
    },
    "chorus": {
        "lsp": dict(
            path=LSP, plugin_name="Chorus Mono",
            fixed={"tempo_sync": False, "time_computing_method": "Rate", "oversampling": "None",
                   "high_pass_filter_mode": "off", "low_pass_filter_mode": "off",
                   "input_gain_db": 0.0, "output_gain_db": 0.0,
                   "dry_amount_db": 0.0, "wet_amount_db": 0.0, "feedback_on": False},
            map={"rate_hz": ("rate_hz", ident, True),
                 "depth": ("depth_ms", onto(0.5, 12.0), False),
                 "mix": ("dry_wet_balance", to_pct, True)},
        ),
    },
    "vibrato": {
        "lsp": dict(
            path=LSP, plugin_name="Chorus Mono",
            fixed={"tempo_sync": False, "time_computing_method": "Rate", "oversampling": "None",
                   "high_pass_filter_mode": "off", "low_pass_filter_mode": "off",
                   "input_gain_db": 0.0, "output_gain_db": 0.0,
                   "dry_amount_db": 0.0, "wet_amount_db": 0.0, "feedback_on": False,
                   "dry_wet_balance": 100.0},
            map={"rate_hz": ("rate_hz", ident, True),
                 "depth": ("depth_ms", onto(0.5, 12.0), False)},
        ),
    },
    "flanger": {
        "lsp": dict(
            path=LSP, plugin_name="Flanger Mono",
            fixed={"tempo_sync": False, "time_computing_method": "Rate",
                   "lfo_type": "Sine", "feedback_on": True,
                   "input_gain_db": 0.0, "output_gain_db": 0.0,
                   "dry_amount_db": 0.0, "wet_amount_db": 0.0, "dry_wet_balance": 50.0},
            map={"rate_hz": ("rate_hz", ident, True),
                 "depth": ("depth_ms", onto(0.5, 12.0), False),
                 "feedback": ("feedback_gain_db", to_db, True),
                 "centre_delay_ms": ("min_depth_ms", ident, True)},
        ),
    },
    "phaser": {
        "lsp": dict(
            path=LSP, plugin_name="Phaser Mono",
            fixed={"tempo_sync": False, "time_computing_method": "Rate",
                   "high_pass_filter_mode": "off", "low_pass_filter_mode": "off",
                   "input_gain_db": 0.0, "output_gain_db": 0.0,
                   "dry_amount_db": 0.0, "wet_amount_db": 0.0, "dry_wet_balance": 50.0,
                   "feedback_on": False},
            map={"rate_hz": ("rate_hz", ident, True),
                 "depth": ("depth_db", onto(0.0, 20.0), False)},
        ),
        "chowphaser": dict(
            path="plugins/real/ChowPhaserMono.vst3",
            fixed={"feedback": 0.0, "freq_mult": False, "skew": 0.0, "stages": 8.0},
            map={"rate_hz": ("lfo_freq", ident, True),
                 "depth": ("lfo_depth", onto(0.0, 0.95), False)},
        ),
    },
    "reverb": {
        "dragonfly": dict(
            path="plugins/real/DragonflyRoomReverb.vst3",
            fixed={"dry_level": 50.0, "early_level": 50.0, "late_level": 50.0,
                   "predelay_ms": 0.0, "decay_s": 1.5},
            map={"room_size": ("size_m", onto(8.0, 32.0), False)},
        ),
    },
}

# Nome do braco de referencia in-domain: e a implementacao com que os modelos
# foram treinados. Sem ele nao da pra separar "o modelo nao generaliza" de
# "esse conjunto de avaliacao e simplesmente diferente".
REFERENCE_ARM = "pedalboard"


def all_arms() -> List[str]:
    return [REFERENCE_ARM] + sorted({arm for arms in REGISTRY.values() for arm in arms})


def is_exact(effect: str, arm: str, parameter: str) -> bool:
    """Se o mapeamento desse parametro tem equivalencia fisica neste braco."""
    if arm == REFERENCE_ARM:
        return True
    spec = REGISTRY.get(effect, {}).get(arm)
    if spec is None:
        return False
    return bool(spec["map"].get(parameter, (None, None, False))[2])
