"""Diferenca pareada entre duas execucoes, com IC95% por bootstrap sobre gravacoes.

A unidade e a gravacao, e nao a consulta: os trechos de uma execucao e os seus
renders em cada nivel e arm nao sao independentes. O bootstrap e o
`scipy.stats.bootstrap(..., paired=True)` sobre as medias por gravacao, valido
porque toda gravacao tem o mesmo numero de consultas na grade cruzada.

O IC nao inclui a variancia de treino: dois lotes identicos de 3 sementes do
SupCon diferiram -1,45 [-2,54; -0,31]. Leia-o como limite inferior da incerteza
e compare tambem semente a semente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

N_RESAMPLES = 10000


def recording_accuracy(run_dir: Path, split: str = "validacao") -> pd.Series:
    """Acerto exato do nivel por gravacao, das predicoes gravadas da execucao."""
    from gefx.disent.train import predictions_file

    predictions = pd.read_csv(Path(run_dir) / predictions_file(split))
    # `<gravacao>_s<k>`, o `content_id` do render.
    recording = predictions["query_content"].str.replace(r"_s\d+$", "", regex=True)
    hit = predictions["true_drive_level"] == predictions["pred_drive_level"]
    return hit.groupby(recording).mean()


def paired_difference(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
    """`(a - b, limite inferior, limite superior)` em pontos percentuais."""
    from scipy.stats import bootstrap

    if not a.index.equals(b.index):
        raise ValueError("as duas execucoes nao foram avaliadas nas mesmas gravacoes")
    result = bootstrap(
        (a.to_numpy(), b.to_numpy()),
        lambda x, y, axis: x.mean(axis) - y.mean(axis),
        paired=True, n_resamples=N_RESAMPLES, random_state=0, method="percentile",
    )
    low, high = result.confidence_interval
    return 100 * float(a.mean() - b.mean()), 100 * float(low), 100 * float(high)


def compare_runs(a_runs: Sequence[Path], b_runs: Sequence[Path],
                 split: str = "validacao") -> pd.DataFrame:
    """Uma linha por par de execucoes (`a_runs[i]` contra `b_runs[i]`, em geral a
    mesma semente) e, com mais de um par, uma com as medias por gravacao tiradas
    sobre as sementes antes do bootstrap."""
    if len(a_runs) != len(b_runs) or not a_runs:
        raise ValueError(f"pares desiguais: {len(a_runs)} execucoes A, {len(b_runs)} B")
    a = [recording_accuracy(run, split) for run in a_runs]
    b = [recording_accuracy(run, split) for run in b_runs]

    def row(name: str, x: pd.Series, y: pd.Series) -> Dict[str, object]:
        diff, low, high = paired_difference(x, y)
        return {"par": name, "a": 100 * float(x.mean()), "b": 100 * float(y.mean()),
                "a_menos_b": diff, "ic95_inf": low, "ic95_sup": high}

    rows: List[Dict[str, object]] = [
        row(f"{Path(x).name} - {Path(y).name}", ra, rb)
        for x, y, ra, rb in zip(a_runs, b_runs, a, b)
    ]
    if len(a) > 1:
        rows.append(row("juntas", pd.concat(a, axis=1).mean(axis=1),
                        pd.concat(b, axis=1).mean(axis=1)))
    return pd.DataFrame(rows).round(2)
