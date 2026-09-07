"""Cache de features do dataset do POC II.

Reaproveita `data/cache.py` inteiro: ele ja opera sobre uma pasta de wavs, grava
`<feature>.npz` + `file_names.json` ao lado do audio e nao sabe nada sobre
cadeias. O que falta, e o que esta aqui, e o alinhamento.

O alinhamento e a mesma armadilha do POC I: `file_names.json` guarda a ordem em
que as features foram extraidas (ordem alfabetica dos wavs), que nao e a ordem
das linhas do sidecar nem a do `GridIndex`. Usar as duas trocadas nao levanta
erro nenhum -- so treina o modelo com os rotulos errados. Por isso o casamento e
sempre por nome de arquivo, e sobra ou falta vira excecao.
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
    """Reordena as features para a ordem das linhas de `frame`.

    Casa por nome de arquivo em vez de assumir que as duas ordens coincidem: elas
    nao coincidem, e o erro seria silencioso.
    """
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
    """Features alinhadas linha a linha com `frame`, que pode misturar arms.

    E esta a entrada do treino: o `GridIndex` e construido sobre o mesmo `frame`,
    entao os indices de linha das tuplas indexam diretamente este array.
    """
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
