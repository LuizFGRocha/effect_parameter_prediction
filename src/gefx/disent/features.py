"""Cache de features do dataset do POC II, com a extracao de `data/cache.py`.

A ordem do cache (alfabetica) nao e a do sidecar: as features sao sempre
casadas as linhas por nome de arquivo, e sobra ou falta e erro.

O cache e um `.npy` por arm, e nao o `.npz` do POC I: os caches nao cabem juntos
na RAM, e o `np.load(..., mmap_mode="r")` mapeia um `.npy` direto do disco.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.data.cache import build_cache, file_names_cache_file
from gefx.disent.sidecar import EFFECT_FOLDER, arm_dirs, read_sidecar


def arm_feature_folder(root: Path, arm: str) -> Path:
    return Path(root) / arm / EFFECT_FOLDER


def cache_file(root: Path, arm: str, feature_name: str) -> Path:
    return arm_feature_folder(root, arm) / f"{feature_name}.npy"


def ensure_arm_cache(
    root: Path, arm: str, feature_name: str, rebuild: bool = False
) -> Tuple[np.memmap, np.ndarray]:
    """(features em memmap, nomes) daquele arm, extraindo na primeira vez."""
    folder = arm_feature_folder(root, arm)
    path = cache_file(root, arm, feature_name)
    if rebuild or not path.exists() or not file_names_cache_file(folder).exists():
        features, names = build_cache(folder, feature_name)
        np.save(path, features)
        file_names_cache_file(folder).write_text(json.dumps(names.tolist(), indent=2),
                                                 encoding="utf-8")
    return open_cache_memmap(root, arm, feature_name)


def build_all_caches(
    root: Path,
    feature_name: str,
    arms: Optional[Sequence[str]] = None,
    rebuild: bool = False,
) -> Dict[str, int]:
    """Extrai o cache de todos os arms. Passo separado porque e caro e reusavel."""
    root = Path(root)
    wanted = list(arms) if arms else [path.name for path in arm_dirs(root)]
    if not wanted:
        raise FileNotFoundError(f"nenhum arm com sidecar em {root}")

    totals: Dict[str, int] = {}
    for arm in wanted:
        features, names = ensure_arm_cache(root, arm, feature_name, rebuild=rebuild)
        expected = len(read_sidecar(root / arm))
        if len(names) != expected:
            raise ValueError(
                f"{arm}: cache tem {len(names)} entradas e o sidecar tem {expected}"
            )
        totals[arm] = len(names)
        print(f"  {arm:16s} {features.shape}")
    return totals


# --- acesso sem carregar tudo -------------------------------------------------
def open_cache_memmap(
    root: Path, arm: str, feature_name: str
) -> Tuple[np.memmap, np.ndarray]:
    """(memmap das features, nomes) de um arm, sem trazer nada para a RAM.

    Exige o cache ja construido por `gefx disent cache`.
    """
    path = cache_file(root, arm, feature_name)
    names_file = file_names_cache_file(path.parent)
    if not path.exists() or not names_file.exists():
        raise FileNotFoundError(
            f"cache ausente em {path.parent}. Rode `gefx disent cache --feature {feature_name}`."
        )

    memmap = np.load(path, mmap_mode="r")
    shape = memmap.shape
    names = np.asarray(json.loads(names_file.read_text(encoding="utf-8")), dtype=object)
    if len(names) != shape[0]:
        raise ValueError(
            f"{path.parent}: cache com {shape[0]} linhas e file_names.json com {len(names)}"
        )
    return memmap, names


class FeatureStore:
    """Features de varios arms, alinhadas as linhas de um `frame`, sob demanda.

    Guarda um memmap por arm e a tabela linha-do-frame -> (arm, linha-do-cache).
    `take` materializa so as linhas pedidas, que e o tamanho do batch. O
    alinhamento e por nome de arquivo; sobra ou falta e erro.
    """

    def __init__(self, root: Path, frame: pd.DataFrame, feature_name: str = "Spec") -> None:
        self.root = Path(root)
        self.feature_name = feature_name
        self.frame = frame.reset_index(drop=True)

        self._memmaps: Dict[str, np.memmap] = {}
        self._source = np.empty(len(self.frame), dtype=np.int64)
        self._arm_of_row = np.empty(len(self.frame), dtype=np.int64)
        self.arms: list[str] = sorted(self.frame["arm"].unique())

        shape: Optional[Tuple[int, ...]] = None
        for arm_index, arm in enumerate(self.arms):
            memmap, names = open_cache_memmap(self.root, arm, feature_name)
            if shape is None:
                shape = memmap.shape[1:]
            elif memmap.shape[1:] != shape:
                raise ValueError(
                    f"{arm}: feature de shape {memmap.shape[1:]}, esperado {shape}"
                )
            self._memmaps[arm] = memmap

            position = {str(name): index for index, name in enumerate(names)}
            where = np.flatnonzero((self.frame["arm"] == arm).to_numpy())
            wanted = [str(name) for name in self.frame["file_name"].to_numpy()[where]]
            missing = [name for name in wanted if name not in position]
            if missing:
                raise ValueError(
                    f"{arm}: {len(missing)} linhas do sidecar sem feature em cache "
                    f"(ex.: {missing[:3]}). Reconstrua o cache."
                )
            self._source[where] = [position[name] for name in wanted]
            self._arm_of_row[where] = arm_index

        if shape is None:
            raise ValueError("frame vazio")
        self.feature_shape: Tuple[int, ...] = tuple(shape)
        self.dtype = next(iter(self._memmaps.values())).dtype

    def __len__(self) -> int:
        return len(self.frame)

    def take(self, rows) -> np.ndarray:
        """As features das linhas pedidas, na ordem pedida, ja em RAM."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), *self.feature_shape), dtype=self.dtype)
        for arm_index, arm in enumerate(self.arms):
            where = np.flatnonzero(self._arm_of_row[rows] == arm_index)
            if not len(where):
                continue
            source = self._source[rows[where]]
            # `argsort` porque leitura crescente num memmap e sequencial em disco.
            order = np.argsort(source, kind="stable")
            out[where[order]] = self._memmaps[arm][source[order]]
        return out

    def stream(self, rows, chunk: int = 256):
        """Itera `(linhas, features)` em blocos -- para estatisticas de uma passada."""
        rows = np.asarray(rows, dtype=np.int64)
        for start in range(0, len(rows), chunk):
            block = rows[start : start + chunk]
            yield block, self.take(block)


class PixelStandardizer:
    """Media e desvio por pixel do espectrograma, acumulados numa passada.

    A conta e a `StandardScaler` do sklearn, a mesma de `training/scaling.py`,
    alimentada por `partial_fit` em blocos para nao materializar o treino.
    Ajustado so no treino e persistido com a execucao, em `.npz`.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, n: int) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.n = int(n)

    @classmethod
    def fit(cls, store: "FeatureStore", rows=None, chunk: int = 256,
            floor: float = 1e-6) -> "PixelStandardizer":
        from sklearn.preprocessing import StandardScaler

        rows = np.arange(len(store)) if rows is None else np.asarray(rows, dtype=np.int64)
        if not len(rows):
            raise ValueError("nao ha linhas para ajustar a padronizacao")
        scaler = StandardScaler()
        for _, block in store.stream(rows, chunk=chunk):
            scaler.partial_fit(block.reshape(len(block), -1))
        shape = store.feature_shape
        # Pixels constantes (silencio nas bandas altas) teriam desvio zero.
        std = np.maximum(np.sqrt(scaler.var_), floor)
        return cls(scaler.mean_.reshape(shape), std.reshape(shape), len(rows))

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Padroniza e acrescenta o eixo de canal, como espera a Conv2D."""
        if features.shape[1:] != self.mean.shape:
            raise ValueError(
                f"shape {features.shape[1:]} incompativel com a padronizacao {self.mean.shape}"
            )
        scaled = (features.astype(np.float32) - self.mean) / self.std
        return scaled[..., None]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, n=np.array(self.n))

    @classmethod
    def load(cls, path: Path) -> "PixelStandardizer":
        with np.load(Path(path)) as data:
            return cls(data["mean"], data["std"], int(data["n"]))
