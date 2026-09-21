"""Amostrador de tuplas controladas sobre a grade cruzada.

Alem de ancora e positivo (fase 1), cada tupla traz o doador de efeito e o alvo
da troca (fase 2). Como a grade e totalmente cruzada, o alvo
`x[conteudo(a), configuracao(b), implementacao(a)]` sempre existe em disco;
`GridIndex` recusa uma grade com buracos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ("content_id", "config_index", "arm")


@dataclass(frozen=True)
class SwapTuple:
    """Uma tupla controlada, em indices de linha do `GridIndex`.

    - `anchor`: doador de conteudo.
    - `positive`: mesma configuracao, outro conteudo e (havendo) outra implementacao.
    - `effect_donor`: doador de efeito, de conteudo diferente do anchor.
    - `swap_target`: conteudo e implementacao do anchor, configuracao do doador.
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
    """Sorteia `n` tuplas controladas, em indices de linha."""
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

        positive = index.row(_other(rng, n_contents, content), config, _other(rng, n_arms, arm))

        donor_content = _other(rng, n_contents, content)
        donor_config = int(rng.integers(0, n_configs))
        donor_arm = int(rng.integers(0, n_arms))

        out.append(
            SwapTuple(
                anchor=index.row(content, config, arm),
                positive=positive,
                effect_donor=index.row(donor_content, donor_config, donor_arm),
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
    """Um batch balanceado por configuracao, em indices de linha do `GridIndex`.

    `rows` sao as ancoras; `effect_donor` e `swap_target` as acompanham linha a
    linha, para a fase 2.
    """

    rows: np.ndarray
    content: np.ndarray
    config: np.ndarray
    arm: np.ndarray
    effect_donor: np.ndarray
    swap_target: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)


def class_balanced_batch(
    index: GridIndex,
    rng: np.random.Generator,
    configs_per_batch: int = 8,
    views_per_config: int = 8,
) -> Batch:
    """`P` configuracoes x `K` vistas, cada vista com conteudo e arm sorteados.

    Montado por classe porque o contrastivo so produz termo para ancoras com algum
    positivo no batch.
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
    donors: List[int] = []
    targets: List[int] = []
    for config in chosen:
        # Conteudos sem reposicao sempre que possivel.
        replace = n_contents < views_per_config
        contents = rng.choice(n_contents, size=views_per_config, replace=replace)
        arms = rng.integers(0, n_arms, size=views_per_config)
        for content, arm in zip(contents, arms):
            donor_content = _other(rng, n_contents, int(content))
            donor_config = int(rng.integers(0, n_configs))
            rows.append(index.row(int(content), int(config), int(arm)))
            donors.append(index.row(donor_content, donor_config, int(rng.integers(0, n_arms))))
            targets.append(index.row(int(content), donor_config, int(arm)))

    rows_array = np.array(rows, dtype=np.int64)
    labels = index.labels(rows_array)
    return Batch(
        rows=rows_array,
        content=labels["content"],
        config=labels["config"],
        arm=labels["arm"],
        effect_donor=np.array(donors, dtype=np.int64),
        swap_target=np.array(targets, dtype=np.int64),
    )


def batch_stream(
    index: GridIndex,
    steps: int,
    configs_per_batch: int = 8,
    views_per_config: int = 8,
    seed: int = 0,
):
    """`steps` batches balanceados. Gerador porque nada disto precisa existir junto."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield class_balanced_batch(index, rng, configs_per_batch, views_per_config)


# --- controle de permutacao ---------------------------------------------------
#: Permutadas juntas, para cada linha continuar coerente consigo mesma.
CONFIG_COLUMNS: Tuple[str, ...] = (
    "config_index", "config_key", "drive_level", "tone_level",
    "drive_knob", "tone_cutoff_hz", "drive_db_equivalente",
)


def permute_configs(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Embaralha a configuracao dentro de cada (conteudo, implementacao).

    Controle de permutacao: a grade continua cruzada e os batches balanceados, mas o
    agrupamento deixa de ter relacao com o audio.
    """
    frame = frame.reset_index(drop=True)
    faltando = [c for c in ("content_id", "arm") if c not in frame.columns]
    if faltando:
        raise ValueError(f"faltam colunas para agrupar: {faltando}")

    columns = [c for c in CONFIG_COLUMNS if c in frame.columns]
    if not columns:
        raise ValueError(f"nenhuma coluna de configuracao em {list(frame.columns)}")

    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, positions in frame.groupby(["content_id", "arm"], sort=True).indices.items():
        shuffled = rng.permutation(positions)
        out.loc[positions, columns] = frame.loc[shuffled, columns].to_numpy()
    return out
