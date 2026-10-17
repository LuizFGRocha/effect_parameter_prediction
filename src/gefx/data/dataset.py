"""Leitura de um dataset: features em cache casadas com o sidecar.

`load_chain_dataset` le uma cadeia do POC I inteira para a RAM; `FeatureStore`
le, por demanda, os arms do POC II, que juntos nao cabem nela.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from gefx.data.cache import ensure_feature_cache
from gefx.data.features import FEATURE_NAMES
from gefx.data.grid import arm_folder
from gefx.data.metadata import chain_folder, read_metadata, target_lookup


def load_chain_dataset(
    dataset_root: str | Path,
    chain_key_value: str,
    feature_name: str = "MFCC40",
    force_rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (X, y, file_names) de uma cadeia.

    A ordem das linhas vem do `file_names.json` do cache; `y` e montado
    consultando o sidecar por nome de arquivo, e nao pela ordem do CSV.
    """
    if feature_name not in FEATURE_NAMES:
        raise ValueError(
            f"Unsupported feature_name={feature_name}. Expected one of {sorted(FEATURE_NAMES)}"
        )

    root = Path(dataset_root).resolve()
    metadata = read_metadata(root)
    features, file_names = ensure_feature_cache(
        chain_folder(root, chain_key_value),
        feature_name=feature_name,
        force_rebuild=force_rebuild_cache,
    )

    lookup = target_lookup(metadata, chain_key_value)
    missing = [name for name in file_names if name not in lookup]
    if missing:
        raise RuntimeError(
            "Metadata mismatch: some wav files have no target row. "
            f"Example missing file: {missing[0]}"
        )

    targets = np.asarray([lookup[name] for name in file_names], dtype=np.float32)
    return features, targets, file_names


class FeatureStore:
    """O `load_chain_dataset` do POC II: features de varios arms, alinhadas as
    linhas de um `frame` (um recorte do sidecar), lidas sob demanda.

    Guarda o cache de cada arm mapeado do disco e a tabela linha-do-frame ->
    (arm, linha-do-cache); o cache e extraido na primeira vez, como no POC I.
    `take` materializa so as linhas pedidas, que e o tamanho do batch. A ordem do
    cache (alfabetica) nao e a do sidecar: o alinhamento e por nome de arquivo, e
    sobra ou falta e erro.
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
            memmap, names = ensure_feature_cache(arm_folder(self.root, arm), feature_name,
                                                 mmap_mode="r")
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

    def batches(self, blocks, standardizer, labels: Optional[np.ndarray] = None):
        """Os lotes `blocks` (listas de linhas), padronizados, como um `PyDataset`
        do Keras: o `fit` e o `predict` leem o proximo numa thread enquanto a GPU
        roda o atual, na ordem de `blocks`. Com `labels`, cada lote leva os
        rotulos das suas linhas; com uma tupla deles, uma tupla, uma por saida."""
        import keras

        store = self

        class Batches(keras.utils.PyDataset):
            def __len__(self) -> int:
                return len(blocks)

            def __getitem__(self, index: int):
                rows = blocks[index]
                features = standardizer.transform(store.take(rows))
                if labels is None:
                    return features
                if isinstance(labels, tuple):
                    return features, tuple(label[rows] for label in labels)
                return features, labels[rows]

        # Duas threads: com uma so, o Keras le na thread principal.
        return Batches(workers=2, use_multiprocessing=False)
