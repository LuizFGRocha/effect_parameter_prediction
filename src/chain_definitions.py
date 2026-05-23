from __future__ import annotations

from typing import Iterable, List


EFFECT_PARAMETER_RANGES = {
    "distortion": [
        {"name": "drive_db", "min": 5.0, "max": 40.0},
    ],
    "chorus": [
        {"name": "rate_hz", "min": 0.1, "max": 2.0},
        {"name": "depth", "min": 0.0, "max": 0.4},
        {"name": "mix", "min": 0.5, "max": 0.5},
    ],
    "vibrato": [
        {"name": "rate_hz", "min": 0.1, "max": 3.0},
        {"name": "depth", "min": 0.0, "max": 0.5},
        {"name": "mix", "min": 1.0, "max": 1.0},
        {"name": "feedback", "min": 0.0, "max": 0.0},
    ],
    "flanger": [
        {"name": "rate_hz", "min": 0.1, "max": 3.0},
        {"name": "depth", "min": 0.0, "max": 0.4},
        {"name": "feedback", "min": 0.7, "max": 0.9},
        {"name": "centre_delay_ms", "min": 0.1, "max": 3.0},
    ],
    "feedback_delay": [
        {"name": "delay_seconds", "min": 0.1, "max": 5.0},
        {"name": "feedback", "min": 0.0, "max": 0.9},
        {"name": "mix", "min": 0.0, "max": 1.0},
    ],
    "slapback_delay": [
        {"name": "delay_seconds", "min": 0.075, "max": 0.2},
        {"name": "mix", "min": 0.0, "max": 1.0},
        {"name": "feedback", "min": 0.0, "max": 0.0},
    ],
    "phaser": [
        {"name": "rate_hz", "min": 0.1, "max": 4.0},
        {"name": "depth", "min": 0.0, "max": 1.0},
    ],
    "reverb": [
        {"name": "room_size", "min": 0.0, "max": 1.0},
        {"name": "wet_level", "min": 0.5, "max": 0.5},
        {"name": "dry_level", "min": 0.5, "max": 0.5},
    ],
}

# ordem canonica para efeitos em sequencia
CANONICAL_EFFECT_CHAIN_ORDER = ["distortion", "chorus", "slapback_delay"]

# EFFECT_CHAINS: allow all single-effect chains, but only stack the canonical three
EFFECT_CHAINS = (
    [[effect] for effect in EFFECT_PARAMETER_RANGES.keys()]
    + [
        CANONICAL_EFFECT_CHAIN_ORDER[0:2],
        CANONICAL_EFFECT_CHAIN_ORDER[1:3],
        CANONICAL_EFFECT_CHAIN_ORDER[0:3],
    ]
)

CHAIN_KEY_SEPARATOR = "__"


def chain_key(chain_effects: Iterable[str]) -> str:
    return CHAIN_KEY_SEPARATOR.join(chain_effects)


def chain_key_to_effects(chain_key_value: str) -> List[str]:
    if not chain_key_value:
        return []
    return chain_key_value.split(CHAIN_KEY_SEPARATOR)
