"""Amostrador de batches balanceados por configuracao sobre a grade cruzada.

O batch e o "P x K" de Hermans, Beyer e Leibe (2017, In Defense of the Triplet
Loss for Person Re-Identification): P classes, K exemplos de cada. Aqui a classe
e o nivel de drive, e cada exemplo sorteia conteudo e implementacao.

`GridIndex` recusa uma grade com buracos: cada vista sorteada (conteudo,
implementacao) de uma configuracao tem de existir em disco.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ("content_id", "config_index", "arm")


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
                "de (conteudo, configuracao, implementacao) nao existem."
            )

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


@dataclass(frozen=True)
class Batch:
    """Um batch balanceado por configuracao, em indices de linha do `GridIndex`."""

    rows: np.ndarray
    content: np.ndarray
    config: np.ndarray
    arm: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)


def class_balanced_batch(
    index: GridIndex,
    rng: np.random.Generator,
    configs_per_batch: int = 8,
    views_per_config: int = 8,
) -> Batch:
    """`P` configuracoes x `K` vistas (o batch P x K de Hermans et al. 2017), cada
    vista com conteudo e arm sorteados de forma independente.

    Montado por classe porque o contrastivo so produz termo para ancoras com algum
    positivo no batch. Os conteudos de uma configuracao saem sem reposicao; os
    arms, com reposicao.
    """
    n_contents, n_configs, n_arms = index.shape
    if configs_per_batch > n_configs:
        raise ValueError(
            f"pedidas {configs_per_batch} configuracoes por batch, ha {n_configs}"
        )
    if views_per_config < 2:
        raise ValueError("sao precisas ao menos 2 vistas por configuracao para haver positivo")

    chosen = rng.choice(n_configs, size=configs_per_batch, replace=False)
    rows: List[int] = []
    for config in chosen:
        replace = n_contents < views_per_config
        contents = rng.choice(n_contents, size=views_per_config, replace=replace)
        arms = rng.integers(0, n_arms, size=views_per_config)
        for content, arm in zip(contents, arms):
            rows.append(index.row(int(content), int(config), int(arm)))

    rows_array = np.array(rows, dtype=np.int64)
    labels = index.labels(rows_array)
    return Batch(
        rows=rows_array,
        content=labels["content"],
        config=labels["config"],
        arm=labels["arm"],
    )
