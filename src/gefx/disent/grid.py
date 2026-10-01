"""Grade de configuracoes e particao de conteudo do POC II.

- A grade: os niveis de drive. O nivel e um indice; o valor de knob depende do
  arm e sai do arquivo de niveis. Nao ha tone: a EQ propria de cada pedal, que
  em alguns muda com o ganho, impedia um fator de tone igual em todos.
- A particao: treino / catalogo / consulta, disjuntas por conteudo, para a busca
  nao acertar reconhecendo o que foi tocado. A validacao sai do treino, na
  leitura (`validation_recordings`), com a mesma forma do teste: uma consulta e
  um catalogo de gravacoes disjuntas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

SPLITS: Tuple[str, ...] = ("train", "catalog", "query")
#: As particoes de validacao, tiradas do treino. Os nomes espelham as do teste.
VALIDATION_SPLITS: Tuple[str, ...] = ("val_query", "val_catalog")
#: Gravacoes em cada lado da validacao: 2 x 20 das 240 de treino.
VALIDATION_RECORDINGS = 20
VALIDATION_SEED = 20260930
#: As duas buscas: (consulta, catalogo). O desenvolvimento so le a validacao; o
#: teste e lido uma vez, no fim, com os modelos finais (`gefx disent evaluate`).
EVAL_SPLITS: Mapping[str, Tuple[str, str]] = {
    "validacao": ("val_query", "val_catalog"),
    "teste": ("query", "catalog"),
}
DEFAULT_SPLIT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "catalog": 0.2, "query": 0.2}


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


def validation_recordings(
    recordings: Sequence[str],
    per_side: int = VALIDATION_RECORDINGS,
    seed: int = VALIDATION_SEED,
) -> Dict[str, List[str]]:
    """Separa, das gravacoes de treino, as da consulta e as do catalogo de validacao.

    Por gravacao, como o teste: os trechos de uma execucao ficam do mesmo lado. O
    sorteio depende so da lista e da semente, entao a validacao e a mesma em toda
    execucao sobre o mesmo dataset e os mesmos arms.
    """
    unique = sorted(set(recordings))
    if per_side < 1:
        raise ValueError("a validacao precisa de ao menos uma gravacao de cada lado")
    if len(unique) <= 2 * per_side:
        raise ValueError(
            f"{len(unique)} gravacoes de treino nao comportam 2 x {per_side} de "
            "validacao e ainda deixar treino"
        )
    shuffled = [str(item) for item in np.random.default_rng(seed).permutation(unique)]
    return {
        "val_query": shuffled[:per_side],
        "val_catalog": shuffled[per_side : 2 * per_side],
        "train": shuffled[2 * per_side :],
    }
