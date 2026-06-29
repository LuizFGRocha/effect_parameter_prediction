from __future__ import annotations

import math

import numpy as np


def standard_error(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(len(array)))