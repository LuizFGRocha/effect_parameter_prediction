"""Grade de configuracoes e particao de conteudo do POC II.

Duas coisas que juntas definem o desenho fatorial do dataset:

- **A grade de configuracoes**: 8 niveis de drive x 8 de tone. O nivel e um
  indice, nao um valor -- o valor de knob correspondente depende do arm e sai da
  calibracao (`disent/calibrate.py`), enquanto o corte do tone e o mesmo em todos
  os arms por construcao.
- **A particao por conteudo**: treino / catalogo / consulta, disjuntas. Particiona
  por conteudo e nao por linha porque a pergunta do trabalho e justamente se o
  codigo de efeito sobrevive a uma execucao diferente: se a mesma execucao
  aparecesse no catalogo e na consulta, a busca poderia acertar reconhecendo o
  que foi tocado, e nao o ajuste.

O produto cartesiano (conteudo x configuracao x arm) e renderizado inteiro. E
isso que torna o alvo exato da troca de codigos uma consulta de tabela, e nao uma
aproximacao -- ver `disent/sampler.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from gefx.disent.arms import DRIVE_LEVELS, TONE_LEVELS, tone_cutoff_hz

SPLITS: Tuple[str, ...] = ("train", "catalog", "query")
DEFAULT_SPLIT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "catalog": 0.2, "query": 0.2}

_CONFIG_KEY = re.compile(r"^d(\d+)t(\d+)$")


@dataclass(frozen=True, order=True)
class Config:
    """Uma configuracao da grade, em indices de nivel."""

    drive_level: int
    tone_level: int

    @property
    def key(self) -> str:
        return f"d{self.drive_level}t{self.tone_level}"


def all_configs(
    drive_levels: int = DRIVE_LEVELS, tone_levels: int = TONE_LEVELS
) -> List[Config]:
    """As configuracoes na ordem canonica (drive externo, tone interno).

    A ordem e o indice: `all_configs()[config_index(c)] is c`. Ela vai para o
    rotulo de classe do contrastivo e para a ordem das entradas do catalogo, e
    por isso nao pode mudar sem invalidar modelos treinados.
    """
    return [
        Config(drive, tone) for drive in range(drive_levels) for tone in range(tone_levels)
    ]


def config_index(config: Config, tone_levels: int = TONE_LEVELS) -> int:
    return config.drive_level * tone_levels + config.tone_level


def config_from_index(index: int, tone_levels: int = TONE_LEVELS) -> Config:
    return Config(index // tone_levels, index % tone_levels)


def parse_config_key(key: str) -> Config:
    match = _CONFIG_KEY.match(key)
    if match is None:
        raise ValueError(f"chave de configuracao invalida: {key!r}")
    return Config(int(match.group(1)), int(match.group(2)))


# --- resolucao nivel -> valor fisico -----------------------------------------
def resolve_drive(calibration: Mapping[str, object], arm_key: str, drive_level: int) -> float:
    """Valor do knob de drive daquele arm para aquele nivel."""
    arms = calibration["arms"]
    if arm_key not in arms:
        raise KeyError(f"arm {arm_key!r} nao esta na calibracao")
    levels = arms[arm_key]["levels"]
    if not 0 <= drive_level < len(levels):
        raise ValueError(f"nivel de drive {drive_level} fora de [0, {len(levels)})")
    return float(levels[drive_level])


def resolve_tone(tone_level: int, tone_levels: int = TONE_LEVELS) -> float:
    """Corte do estagio de tone. Nao depende do arm -- e essa a graca dele."""
    return tone_cutoff_hz(tone_level, tone_levels)


def resolve(
    calibration: Mapping[str, object], arm_key: str, config: Config
) -> Tuple[float, float]:
    """(knob de drive, corte de tone em Hz) para renderizar `config` naquele arm."""
    return resolve_drive(calibration, arm_key, config.drive_level), resolve_tone(
        config.tone_level
    )


# --- identidade de um render --------------------------------------------------
def render_name(content_id: str, config: Config, index: int) -> str:
    """Nome do wav. Identico entre arms de proposito.

    E o que torna o pareamento verificavel: o oraculo compara
    `<A>/<nome>` com `<B>/<nome>` sabendo que so a implementacao mudou.
    """
    return f"{content_id}__{config.key}__{index:05d}.wav"


# --- particao por conteudo ----------------------------------------------------
def split_contents(
    content_ids: Sequence[str],
    seed: int = 20260906,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
) -> Dict[str, List[str]]:
    """Divide os conteudos em treino/catalogo/consulta, disjuntos.

    Embaralha antes de cortar porque os nomes das gravacoes de Rossi vem
    agrupados por guitarra, captador e tecnica; cortar a lista ordenada poria uma
    guitarra inteira num unico split e confundiria "outra execucao" com
    "outro instrumento".
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
    # Cada split fica com ao menos um conteudo: sem isso a pre-condicao de
    # `len(SPLITS)` conteudos seria falsa, porque com exatamente 3 o
    # arredondamento levaria os 3 para treino e catalogo.
    while n_train + n_catalog > total_items - 1:
        if n_train >= n_catalog:
            n_train -= 1
        else:
            n_catalog -= 1

    # A consulta leva o resto, para nenhum conteudo se perder no arredondamento.
    out = {
        "train": shuffled[:n_train],
        "catalog": shuffled[n_train : n_train + n_catalog],
        "query": shuffled[n_train + n_catalog :],
    }
    return {name: [str(item) for item in items] for name, items in out.items()}
