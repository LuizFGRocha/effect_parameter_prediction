"""Fonte unica de verdade sobre efeitos, parametros e cadeias.

Os valores de parametro circulam **normalizados em [0,1]** em todo o projeto; o
valor fisico e derivado de `min`/`max` na hora de renderizar. `predict: False`
marca um parametro fixo, que fica de fora do vetor alvo.

ATENCAO: acrescentar ou reordenar entradas aqui muda silenciosamente o tamanho e
a ordem do vetor alvo, invalidando datasets ja renderizados e modelos ja
treinados.
"""
from __future__ import annotations

from typing import Iterable, List


EFFECT_PARAMETER_RANGES = {
    "distortion": [
        {"name": "drive_db", "min": 5.0, "max": 40.0},
    ],
    "chorus": [
        {"name": "rate_hz", "min": 0.1, "max": 4.0},
        {"name": "depth", "min": 0.1, "max": 0.4},
        {"name": "mix", "min": 0.5, "max": 0.5, "predict": False},
    ],
    "vibrato": [
        {"name": "rate_hz", "min": 0.1, "max": 3.0},
        {"name": "depth", "min": 0.0, "max": 0.5},
        {"name": "mix", "min": 1.0, "max": 1.0, "predict": False},
        {"name": "feedback", "min": 0.0, "max": 0.0, "predict": False},
    ],
    "flanger": [
        {"name": "rate_hz", "min": 0.1, "max": 3.0},
        {"name": "depth", "min": 0.0, "max": 0.4},
        {"name": "feedback", "min": 0.7, "max": 0.9},
        {"name": "centre_delay_ms", "min": 0.1, "max": 3.0},
    ],
    "feedback_delay": [
        {"name": "delay_seconds", "min": 0.1, "max": 0.9},
        {"name": "feedback", "min": 0.0, "max": 0.9},
        {"name": "mix", "min": 0.0, "max": 1.0},
    ],
    "slapback_delay": [
        {"name": "delay_seconds", "min": 0.075, "max": 0.2},
        {"name": "mix", "min": 0.0, "max": 1.0},
        {"name": "feedback", "min": 0.0, "max": 0.0, "predict": False},
    ],
    "phaser": [
        {"name": "rate_hz", "min": 0.1, "max": 4.0},
        {"name": "depth", "min": 0.0, "max": 1.0},
    ],
    "reverb": [
        {"name": "room_size", "min": 0.0, "max": 1.0},
        {"name": "wet_level", "min": 0.5, "max": 0.5, "predict": False},
        {"name": "dry_level", "min": 0.5, "max": 0.5, "predict": False},
    ],
}

# Ordem canonica para efeitos em sequencia.
CANONICAL_EFFECT_CHAIN_ORDER = ["distortion", "chorus", "slapback_delay"]

# Todos os efeitos isolados, mais os tres subconjuntos empilhados da ordem canonica.
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
    """Nome dos efeitos unido por `__`; e o nome de pasta em datasets e resultados."""
    return CHAIN_KEY_SEPARATOR.join(chain_effects)


def chain_key_to_effects(chain_key_value: str) -> List[str]:
    if not chain_key_value:
        return []
    return chain_key_value.split(CHAIN_KEY_SEPARATOR)


def effect_predictable_params(effect: str) -> List[dict]:
    return [param for param in EFFECT_PARAMETER_RANGES[effect] if param.get("predict", True)]


def effect_fixed_params(effect: str) -> List[dict]:
    return [param for param in EFFECT_PARAMETER_RANGES[effect] if not param.get("predict", True)]


def parameter_names_for_chain(chain_key_value: str) -> List[str]:
    """Nomes `<efeito>_<parametro>` na mesma ordem do vetor alvo da cadeia."""
    names: List[str] = []
    for effect in chain_key_to_effects(chain_key_value):
        for param in effect_predictable_params(effect):
            names.append(f"{effect}_{param['name']}")
    return names


def chain_output_dim(chain_key_value: str) -> int:
    """Quantidade de parametros previstos pela cadeia (largura da saida do modelo)."""
    return sum(
        len(effect_predictable_params(effect))
        for effect in chain_key_to_effects(chain_key_value)
    )
