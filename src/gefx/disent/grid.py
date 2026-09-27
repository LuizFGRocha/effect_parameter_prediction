"""Grade de configuracoes e particao de conteudo do POC II.

- A grade: os niveis de drive. O nivel e um indice; o valor de knob depende do
  arm e sai do arquivo de niveis. Nao ha tone: a EQ propria de cada pedal, que
  em alguns muda com o ganho, impedia um fator de tone igual em todos.
- A particao: treino / catalogo / consulta, disjuntas por conteudo, para a busca
  nao acertar reconhecendo o que foi tocado.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

SPLITS: Tuple[str, ...] = ("train", "catalog", "query")
DEFAULT_SPLIT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "catalog": 0.2, "query": 0.2}

_CONFIG_KEY = re.compile(r"^d(\d+)$")


@dataclass(frozen=True, order=True)
class Config:
    """Uma configuracao da grade, em indices de nivel."""

    drive_level: int

    @property
    def key(self) -> str:
        return f"d{self.drive_level}"


def all_configs(drive_levels: int) -> List[Config]:
    """As configuracoes na ordem canonica; o indice de classe e o nivel de drive."""
    return [Config(drive) for drive in range(drive_levels)]


def config_index(config: Config) -> int:
    return config.drive_level


def config_from_index(index: int) -> Config:
    return Config(index)


def parse_config_key(key: str) -> Config:
    match = _CONFIG_KEY.match(key)
    if match is None:
        raise ValueError(f"chave de configuracao invalida: {key!r}")
    return Config(int(match.group(1)))


def render_name(content_id: str, config: Config, index: int) -> str:
    """Nome do wav, identico entre arms: e o que torna o pareamento verificavel."""
    return f"{content_id}__{config.key}__{index:05d}.wav"


# --- particao por conteudo ----------------------------------------------------
def split_contents(
    content_ids: Sequence[str],
    seed: int = 20260906,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
) -> Dict[str, List[str]]:
    """Divide os conteudos em treino/catalogo/consulta, disjuntos.

    Embaralha antes porque os nomes vem agrupados por guitarra, captador e tecnica.
    """
    if set(fractions) != set(SPLITS):
        raise ValueError(f"fracoes devem cobrir exatamente {SPLITS}")
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"fracoes devem somar 1.0, somam {total}")

    unique = sorted(set(content_ids))
    if len(unique) != len(content_ids):
        raise ValueError("ha content_id repetido")
    if len(unique) < len(SPLITS):
        raise ValueError(f"sao precisos ao menos {len(SPLITS)} conteudos")

    shuffled = list(np.random.default_rng(seed).permutation(unique))
    total_items = len(shuffled)
    n_train = max(1, int(round(fractions["train"] * total_items)))
    n_catalog = max(1, int(round(fractions["catalog"] * total_items)))
    # Garante ao menos um conteudo para a consulta, que leva o resto.
    while n_train + n_catalog > total_items - 1:
        if n_train >= n_catalog:
            n_train -= 1
        else:
            n_catalog -= 1

    out = {
        "train": shuffled[:n_train],
        "catalog": shuffled[n_train : n_train + n_catalog],
        "query": shuffled[n_train + n_catalog :],
    }
    return {name: [str(item) for item in items] for name, items in out.items()}
