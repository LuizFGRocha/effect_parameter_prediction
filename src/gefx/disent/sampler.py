"""Amostrador de tuplas controladas sobre a grade cruzada.

E aqui que mora a costura que torna a fase 2 barata. A fase 1 (contrastivo +
reversao de gradiente) usa `anchor` e `positive`; a fase 2 (reconstrucao com
troca de codigos) usa `effect_donor` e `swap_target`. Emitir os quatro desde
agora custa uma consulta de tabela e evita reescrever o pipeline de dados
depois -- ver a memoria `poc2-extensao-decoder-troca-de-codigos`.

O que a grade totalmente cruzada da de presente: para qualquer par (ancora `a`,
doador de efeito `b`), o alvo da troca

    x[conteudo(a), configuracao(b), implementacao(a)]

**existe em disco**. A reconstrucao da fase 2 e portanto supervisionada, com alvo
exato, e dispensa o adversario que DrNet e DNA-GAN precisam justamente porque
neles esse alvo nao existe.

`GridIndex` verifica a completude da grade na construcao: um buraco vira erro na
hora de montar o indice, e nao uma tupla silenciosamente errada no meio do treino.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Colunas que definem os fatores. Tem de existir no sidecar.
FACTOR_COLUMNS = ("content_id", "config_index", "arm")


@dataclass(frozen=True)
class SwapTuple:
    """Uma tupla controlada, em indices de linha do `GridIndex`.

    - `anchor`: doador de conteudo.
    - `positive`: mesma configuracao do anchor, outro conteudo e, quando ha mais
      de uma implementacao, outra implementacao. E o par positivo do contrastivo:
      o que ele afirma e que configuracao igual deve ficar junto ainda que
      conteudo e implementacao mudem.
    - `effect_donor`: doador de efeito, de conteudo obrigatoriamente diferente do
      anchor -- se fosse o mesmo, o alvo da troca seria o proprio doador e a
      reconstrucao nao exigiria separar nada.
    - `swap_target`: conteudo do anchor + configuracao do doador + implementacao
      do anchor. Usado so na fase 2.
    """

    anchor: int
    positive: int
    effect_donor: int
    swap_target: int


class GridIndex:
    """Mapa (conteudo, configuracao, implementacao) -> linha, sobre a grade cruzada."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [column for column in FACTOR_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"faltam colunas de fator no sidecar: {missing}")

        self.frame = frame.reset_index(drop=True)
        self.contents: List[str] = sorted(self.frame["content_id"].unique())
        self.configs: List[int] = sorted(self.frame["config_index"].unique())
        self.arms: List[str] = sorted(self.frame["arm"].unique())

        self._content_of = {name: index for index, name in enumerate(self.contents)}
        self._config_of = {value: index for index, value in enumerate(self.configs)}
        self._arm_of = {name: index for index, name in enumerate(self.arms)}

        shape = (len(self.contents), len(self.configs), len(self.arms))
        self.lookup = np.full(shape, -1, dtype=np.int64)
        for row, record in enumerate(self.frame.itertuples(index=False)):
            self.lookup[
                self._content_of[record.content_id],
                self._config_of[record.config_index],
                self._arm_of[record.arm],
            ] = row

        holes = int((self.lookup < 0).sum())
        if holes:
            raise ValueError(
                f"a grade nao esta cruzada: {holes} de {self.lookup.size} combinacoes "
                "de (conteudo, configuracao, implementacao) nao existem. O alvo exato "
                "da troca de codigos deixaria de existir para elas."
            )

        # Rotulos por linha, prontos para as cabecas adversarias.
        self.content_label = np.empty(len(self.frame), dtype=np.int64)
        self.config_label = np.empty(len(self.frame), dtype=np.int64)
        self.arm_label = np.empty(len(self.frame), dtype=np.int64)
        for content in range(shape[0]):
            for config in range(shape[1]):
                for arm in range(shape[2]):
                    row = self.lookup[content, config, arm]
                    self.content_label[row] = content
                    self.config_label[row] = config
                    self.arm_label[row] = arm

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def shape(self):
        return self.lookup.shape

    def row(self, content: int, config: int, arm: int) -> int:
        return int(self.lookup[content, config, arm])

    def file_names(self, rows: Sequence[int]) -> List[str]:
        return [str(name) for name in self.frame["file_name"].to_numpy()[np.asarray(rows)]]

    def labels(self, rows: Sequence[int]) -> Dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        return {
            "content": self.content_label[rows],
            "config": self.config_label[rows],
            "arm": self.arm_label[rows],
        }


def _other(rng: np.random.Generator, size: int, avoid: int) -> int:
    """Sorteia em [0, size) evitando `avoid`. Cai em `avoid` se nao houver outro."""
    if size <= 1:
        return avoid
    draw = int(rng.integers(0, size - 1))
    return draw if draw < avoid else draw + 1


def swap_tuples(index: GridIndex, n: int, seed: int = 0) -> List[SwapTuple]:
    """Sorteia `n` tuplas controladas.

    Nada aqui le audio: sao indices. Quem materializa as features e o laco de
    treino, que assim pode usar o cache do dataset sem copiar nada.
    """
    if n < 0:
        raise ValueError("n deve ser >= 0")
    n_contents, n_configs, n_arms = index.shape
    if n_contents < 2:
        raise ValueError("sao precisos ao menos 2 conteudos para uma troca ser informativa")

    rng = np.random.default_rng(seed)
    out: List[SwapTuple] = []
    for _ in range(n):
        content = int(rng.integers(0, n_contents))
        config = int(rng.integers(0, n_configs))
        arm = int(rng.integers(0, n_arms))

        # Positivo: mesma configuracao, outro conteudo e outra implementacao.
        positive = index.row(_other(rng, n_contents, content), config, _other(rng, n_arms, arm))

        # Doador de efeito: conteudo obrigatoriamente diferente, senao o alvo da
        # troca seria o proprio doador.
        donor_content = _other(rng, n_contents, content)
        donor_config = int(rng.integers(0, n_configs))
        donor_arm = int(rng.integers(0, n_arms))

        out.append(
            SwapTuple(
                anchor=index.row(content, config, arm),
                positive=positive,
                effect_donor=index.row(donor_content, donor_config, donor_arm),
                # A consulta de tabela que e o ponto de toda esta classe.
                swap_target=index.row(content, donor_config, arm),
            )
        )
    return out


def as_arrays(tuples: Sequence[SwapTuple]) -> Dict[str, np.ndarray]:
    """As quatro colunas de indices, para indexar o cache de features de uma vez."""
    return {
        name: np.array([getattr(item, name) for item in tuples], dtype=np.int64)
        for name in ("anchor", "positive", "effect_donor", "swap_target")
    }


def build_index(
    frame: pd.DataFrame,
    split: Optional[str] = None,
    arms: Optional[Sequence[str]] = None,
) -> GridIndex:
    """Indice restrito a um split e a um subconjunto de implementacoes.

    E por aqui que se monta o leave-one-arm-out: passar as N-1 implementacoes de
    treino. O recorte preserva o cruzamento, entao o alvo da troca continua
    existindo dentro do recorte.
    """
    if split is not None:
        frame = frame[frame["split"] == split]
    if arms is not None:
        frame = frame[frame["arm"].isin(list(arms))]
    if frame.empty:
        raise ValueError(f"recorte vazio (split={split!r}, arms={arms!r})")
    return GridIndex(frame)
