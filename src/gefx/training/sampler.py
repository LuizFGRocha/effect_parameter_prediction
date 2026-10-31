"""Amostrador de batches balanceados por classe sobre a grade cruzada.

O batch e o "P x K" de Hermans, Beyer e Leibe (2017, In Defense of the Triplet
Loss for Person Re-Identification): P classes, K exemplos de cada, como o
`sample_k_fids_for_pid` do codigo deles (VisualComputingInstitute/triplet-reid,
MIT). A classe e o nivel de drive, e cada exemplo sorteia conteudo e
implementacao.

Para treinar tambem `z_c`, o batch tem de ter positivos de conteudo, que o P x K
por nivel quase nunca sorteia (~1000 trechos de treino). Com `by="content"` a
classe passa a ser o trecho: P trechos, cada um em K niveis sem reposicao. Com
K igual ao numero de niveis, cada trecho aparece em todos, entao cada nivel
aparece P vezes no batch e o `z_e` continua com positivos.

`GridIndex` recusa uma grade com buracos: cada vista sorteada (conteudo,
implementacao) de um nivel tem de existir em disco.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ("content_id", "drive_level", "arm")


class GridIndex:
    """Mapa (conteudo, nivel, implementacao) -> linha, sobre a grade cruzada."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [column for column in FACTOR_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"faltam colunas de fator no sidecar: {missing}")

        self.frame = frame.reset_index(drop=True)
        self.contents: List[str] = sorted(self.frame["content_id"].unique())
        self.levels: List[int] = sorted(self.frame["drive_level"].unique())
        self.arms: List[str] = sorted(self.frame["arm"].unique())

        #: O indice de cada linha em cada fator: os rotulos do treino.
        self.content_label, self.level_label, arm_label = (
            self.frame[column].map({value: i for i, value in enumerate(values)}).to_numpy()
            for column, values in zip(FACTOR_COLUMNS, (self.contents, self.levels, self.arms))
        )
        self.lookup = np.full((len(self.contents), len(self.levels), len(self.arms)), -1,
                              dtype=np.int64)
        self.lookup[self.content_label, self.level_label, arm_label] = np.arange(len(self.frame))

        holes = int((self.lookup < 0).sum())
        if holes:
            raise ValueError(
                f"a grade nao esta cruzada: {holes} de {self.lookup.size} combinacoes "
                "de (conteudo, nivel, implementacao) nao existem."
            )

    def __len__(self) -> int:
        return len(self.frame)


def build_index(
    frame: pd.DataFrame,
    split: Optional[str] = None,
    arms: Optional[Sequence[str]] = None,
) -> GridIndex:
    """Indice restrito a um split e a um subconjunto de implementacoes (leave-one-out)."""
    if split is not None:
        frame = frame[frame["split"] == split]
    if arms is not None:
        frame = frame[frame["arm"].isin(list(arms))]
    if frame.empty:
        raise ValueError(f"recorte vazio (split={split!r}, arms={arms!r})")
    return GridIndex(frame)


def class_balanced_batch(
    index: GridIndex,
    rng: np.random.Generator,
    classes_per_batch: int = 8,
    views_per_class: int = 8,
    by: str = "level",
) -> np.ndarray:
    """As linhas de um batch de `P` classes x `K` vistas (o P x K de Hermans et
    al. 2017). A classe e o nivel de drive (`by="level"`) ou o trecho de conteudo
    (`by="content"`); cada vista sorteia o outro fator sem reposicao e o arm com
    reposicao.

    Montado por classe porque o contrastivo so produz termo para ancoras com algum
    positivo no batch.
    """
    if by not in ("level", "content"):
        raise ValueError(f"classe do batch desconhecida: {by!r}")
    n_contents, n_levels, n_arms = index.lookup.shape
    n_classes, n_others = (n_levels, n_contents) if by == "level" else (n_contents, n_levels)
    if classes_per_batch > n_classes:
        name = "niveis" if by == "level" else "trechos"
        raise ValueError(f"pedidos {classes_per_batch} {name} por batch, ha {n_classes}")
    if views_per_class < 2:
        raise ValueError("sao precisas ao menos 2 vistas por classe para haver positivo")

    chosen = rng.choice(n_classes, size=classes_per_batch, replace=False)
    rows: List[int] = []
    for label in chosen:
        replace = n_others < views_per_class
        others = rng.choice(n_others, size=views_per_class, replace=replace)
        arms = rng.integers(0, n_arms, size=views_per_class)
        for other, arm in zip(others, arms):
            content, level = (other, label) if by == "level" else (label, other)
            rows.append(index.lookup[content, level, arm])
    return np.array(rows, dtype=np.int64)
