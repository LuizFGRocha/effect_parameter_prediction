"""Cache de features do dataset do POC II, sobre `data/cache.py`.

A ordem do cache (alfabetica) nao e a do sidecar: as features sao sempre
casadas as linhas por nome de arquivo, e sobra ou falta e erro.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.data.cache import ensure_feature_cache
from gefx.disent.sidecar import EFFECT_FOLDER, arm_dirs, read_sidecar


def arm_feature_folder(root: Path, arm: str) -> Path:
    return Path(root) / arm / EFFECT_FOLDER


def ensure_arm_cache(
    root: Path, arm: str, feature_name: str, rebuild: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """(features, nomes) daquele arm, extraindo na primeira vez."""
    return ensure_feature_cache(arm_feature_folder(root, arm), feature_name, force_rebuild=rebuild)


def align_to_frame(
    features: np.ndarray, file_names: Sequence[str], frame: pd.DataFrame
) -> np.ndarray:
    """Reordena as features para a ordem das linhas de `frame`, casando por nome de arquivo."""
    position = {str(name): index for index, name in enumerate(file_names)}
    wanted = [str(name) for name in frame["file_name"]]

    missing = [name for name in wanted if name not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} linhas do sidecar sem feature em cache "
            f"(ex.: {missing[:3]}). Reconstrua o cache."
        )
    return features[np.array([position[name] for name in wanted], dtype=np.int64)]


def load_features(
    root: Path,
    frame: pd.DataFrame,
    feature_name: str,
    rebuild: bool = False,
) -> np.ndarray:
    """Features alinhadas linha a linha com `frame`, que pode misturar arms, todas em RAM."""
    out: Optional[np.ndarray] = None
    for arm in sorted(frame["arm"].unique()):
        rows = frame["arm"] == arm
        features, names = ensure_arm_cache(root, arm, feature_name, rebuild=rebuild)
        aligned = align_to_frame(features, names, frame[rows])
        if out is None:
            out = np.empty((len(frame), *aligned.shape[1:]), dtype=aligned.dtype)
        out[np.flatnonzero(rows.to_numpy())] = aligned
    if out is None:
        raise ValueError("frame vazio")
    return out


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
# Os caches nao cabem juntos na RAM. Como `np.savez` nao comprime, o `.npy`
# dentro do zip esta cru em disco e pode ser mapeado direto.
NPZ_MEMBER = "arr_0.npy"
_LOCAL_HEADER = "<IHHHHHIIIHH"
_LOCAL_HEADER_SIZE = 30


def npy_member_offset(path: Path, member: str = NPZ_MEMBER):
    """(deslocamento, shape, dtype, ordem) do membro cru de um `.npz` sem compressao.

    O indice central aponta para o cabecalho local, e nao para os dados: entre
    os dois ficam o nome e o campo extra, cujos tamanhos so o cabecalho local tem.
    """
    import struct
    import zipfile

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"{path}: o membro {member!r} esta comprimido; o mapeamento de memoria "
            "so vale para `np.savez` (sem compressao)"
        )

    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        signature, *_, name_length, extra_length = struct.unpack(
            _LOCAL_HEADER, handle.read(_LOCAL_HEADER_SIZE))
        if signature != 0x04034B50:
            raise ValueError(f"{path}: cabecalho local do zip invalido")
        handle.seek(info.header_offset + _LOCAL_HEADER_SIZE + name_length + extra_length)
        version = np.lib.format.read_magic(handle)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError(f"{path}: versao de .npy nao suportada: {version}")
        shape, fortran, dtype = readers[version](handle)
        return handle.tell(), shape, dtype, fortran


def open_cache_memmap(
    root: Path, arm: str, feature_name: str
) -> Tuple[np.memmap, np.ndarray]:
    """(memmap das features, nomes) de um arm, sem trazer nada para a RAM.

    Exige o cache ja construido por `gefx disent cache`.
    """
    import json

    folder = arm_feature_folder(root, arm)
    npz = folder / f"{feature_name}.npz"
    names_file = folder / "file_names.json"
    if not npz.exists() or not names_file.exists():
        raise FileNotFoundError(
            f"cache ausente em {folder}. Rode `gefx disent cache --feature {feature_name}`."
        )

    offset, shape, dtype, fortran = npy_member_offset(npz)
    if fortran:
        raise ValueError(f"{npz}: array em ordem Fortran, nao suportado")
    memmap = np.memmap(npz, dtype=dtype, mode="r", offset=offset, shape=shape)

    names = np.asarray(json.loads(names_file.read_text(encoding="utf-8")), dtype=object)
    if len(names) != shape[0]:
        raise ValueError(
            f"{folder}: cache com {shape[0]} linhas e file_names.json com {len(names)}"
        )
    return memmap, names


class FeatureStore:
    """Features de varios arms, alinhadas as linhas de um `frame`, sob demanda.

    Guarda um memmap por arm e a tabela linha-do-frame -> (arm, linha-do-cache).
    `take` materializa so as linhas pedidas, que e o tamanho do batch. Substitui
    `load_features` quando o conjunto nao cabe na memoria; a semantica de
    alinhamento e a mesma -- casamento por nome de arquivo, sobra ou falta e erro.
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

    O equivalente de `training/scaling.py` sem materializar o treino. Ajustado
    so no treino e persistido com a execucao.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, n: int) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.n = int(n)

    @classmethod
    def fit(cls, store: "FeatureStore", rows=None, chunk: int = 256,
            floor: float = 1e-6) -> "PixelStandardizer":
        rows = np.arange(len(store)) if rows is None else np.asarray(rows, dtype=np.int64)
        if not len(rows):
            raise ValueError("nao ha linhas para ajustar a padronizacao")
        total = np.zeros(store.feature_shape, dtype=np.float64)
        total_sq = np.zeros(store.feature_shape, dtype=np.float64)
        for _, block in store.stream(rows, chunk=chunk):
            block = block.astype(np.float64)
            total += block.sum(axis=0)
            total_sq += (block ** 2).sum(axis=0)
        mean = total / len(rows)
        variance = np.maximum(total_sq / len(rows) - mean ** 2, 0.0)
        # Pixels constantes (silencio nas bandas altas) teriam desvio zero.
        std = np.maximum(np.sqrt(variance), floor)
        return cls(mean.astype(np.float32), std.astype(np.float32), len(rows))

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
