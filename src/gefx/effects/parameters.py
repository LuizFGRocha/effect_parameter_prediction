"""Conversao entre valores normalizados [0,1] e valores fisicos, e amostragem."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from gefx.effects.catalog import (
    EFFECT_PARAMETER_RANGES,
    effect_fixed_params,
    effect_predictable_params,
)


def convert_normalized_to_raw(
    params: Sequence[Dict[str, float]],
    norm_values: Sequence[float],
) -> Dict[str, float]:
    """Aplica `min + (max - min) * v` para cada parametro, na ordem dada.

    O `zip` sozinho truncaria em silencio: um vetor curto devolveria um dict
    parcial, e o efeito seria renderizado com os defaults do plugin nos
    parametros que faltassem.
    """
    if len(norm_values) != len(params):
        raise ValueError(
            f"Vetor normalizado com {len(norm_values)} valores para {len(params)} parametros: "
            f"{[param['name'] for param in params]}"
        )

    raw_values: Dict[str, float] = {}
    for value, param in zip(norm_values, params):
        raw = param["min"] + (param["max"] - param["min"]) * float(value)
        raw_values[param["name"]] = float(raw)
    return raw_values


def sample_normalized_matrix(
    rng: np.random.Generator,
    amount_of_samples: int,
    amount_of_parameters: int,
) -> np.ndarray:
    return rng.random((amount_of_samples, amount_of_parameters), dtype=np.float64)


def raw_params_for_effect(effect: str, norm_values: Sequence[float]) -> Dict[str, float]:
    """Valores fisicos de um efeito: os previstos vindos de `norm_values`, mais os fixos."""
    raw = convert_normalized_to_raw(effect_predictable_params(effect), norm_values)
    for fixed_param in effect_fixed_params(effect):
        raw[fixed_param["name"]] = float(fixed_param["min"])
    return raw


def split_vector_by_effect(
    chain_effects: Sequence[str],
    norm_vector: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Fatia o vetor normalizado da cadeia e devolve os valores fisicos por efeito.

    O fatiamento segue a ordem dos efeitos na cadeia e, dentro de cada efeito, a
    ordem de `EFFECT_PARAMETER_RANGES` — a mesma convencao do vetor alvo.
    """
    per_effect: Dict[str, Dict[str, float]] = {}
    offset = 0
    for effect in chain_effects:
        amount = len(effect_predictable_params(effect))
        per_effect[effect] = raw_params_for_effect(effect, norm_vector[offset : offset + amount])
        offset += amount
    return per_effect


def effect_presence(chain_effects: Sequence[str]) -> Dict[str, int]:
    """Indicador 0/1 de cada efeito do catalogo nesta cadeia."""
    selected = set(chain_effects)
    return {name: int(name in selected) for name in EFFECT_PARAMETER_RANGES.keys()}


def binary_suffix(chain_effects: Sequence[str]) -> str:
    """Sufixo binario de presenca usado no nome dos arquivos renderizados."""
    return "".join(str(bit) for bit in effect_presence(chain_effects).values())
