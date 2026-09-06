"""Renderizacao pela implementacao de referencia (Pedalboard).

E esta a implementacao com que os modelos sao treinados; no estudo cross-impl ela
e o braco `pedalboard`, a referencia in-domain.
"""
from __future__ import annotations

from typing import Dict, Sequence

from pedalboard import Chorus, Delay, Distortion, Pedalboard, Phaser, Reverb

BUILT_IN_PLUGINS = {
    "distortion": Distortion,
    "chorus": Chorus,
    "vibrato": Chorus,
    "flanger": Chorus,
    "feedback_delay": Delay,
    "slapback_delay": Delay,
    "phaser": Phaser,
    "reverb": Reverb,
}


def build_chain(
    chain_effects: Sequence[str],
    raw_param_dict: Dict[str, Dict[str, float]],
) -> Pedalboard:
    """Monta a Pedalboard da cadeia com os valores fisicos ja convertidos."""
    plugins = []
    for effect in chain_effects:
        if effect not in BUILT_IN_PLUGINS:
            raise ValueError(f"Effect '{effect}' not recognized in built-in plugins.")
        plugins.append(BUILT_IN_PLUGINS[effect](**raw_param_dict[effect]))
    return Pedalboard(plugins)
