"""Oraculo sonico: qual ajuste de B soa como este ajuste de A.

A verdade fundamental do POC II. Implementacoes nao compartilham unidade de
ganho, entao a correspondencia e medida:

    q*(p) = argmin_q  media_s  D( x[A, p, s], x[B, q, s] )

sobre os mesmos conteudos `s`, com `D` = L1 sobre log-mel multirresolucao
(Steinmetz et al. 2022). O pareamento por conteudo cancela o conteudo.

O oraculo exige o sinal seco da referencia tocando o mesmo trecho, que nao
existe no uso real: ele e professor, nao produto.

O contraste (quanto a distancia varia entre o melhor e o pior nivel) e a
porteira de aceitacao de um arm: contraste baixo quer dizer que os niveis
daquele arm nao se distinguem, mesmo com o eixo monotono.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Janela curta resolve o transiente, janela longa o corpo harmonico.
FFT_SIZES: Tuple[int, ...] = (512, 1024, 2048, 4096)
N_MELS = 64
LOG_FLOOR = 1e-5

# O pior nivel tem de estar ao menos ao dobro da distancia do melhor.
CONTRAST_THRESHOLD = 1.0


def log_mel_stack(
    audio: np.ndarray,
    sr: int,
    fft_sizes: Sequence[int] = FFT_SIZES,
    n_mels: int = N_MELS,
) -> Tuple[np.ndarray, ...]:
    """Log-mel em varias resolucoes. Import de librosa adiado (custa ~1 s)."""
    import librosa

    x = np.asarray(audio, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError(f"esperado sinal mono, veio shape {np.shape(audio)}")
    return tuple(
        np.log(
            librosa.feature.melspectrogram(
                y=x, sr=sr, n_fft=n, hop_length=n // 4, n_mels=n_mels
            )
            + LOG_FLOOR
        )
        for n in fft_sizes
    )


def spectral_distance(
    left: Sequence[np.ndarray], right: Sequence[np.ndarray]
) -> float:
    """L1 media entre duas pilhas log-mel (L2 seria dominada pelas celulas de maior energia)."""
    if len(left) != len(right):
        raise ValueError(f"pilhas de tamanhos diferentes: {len(left)} e {len(right)}")
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(left, right)]))


def alignment_matrix(
    targets: Sequence[Sequence[Sequence[np.ndarray]]],
    candidates: Sequence[Sequence[Sequence[np.ndarray]]],
) -> np.ndarray:
    """Distancia media por conteudo entre cada nivel alvo e cada nivel candidato.

    `targets[p][s]` e a pilha do nivel `p` da referencia no conteudo `s`;
    `candidates[q][s]`, a do nivel `q` do arm. Devolve `(n_alvos, n_candidatos)`.
    """
    n_p, n_q = len(targets), len(candidates)
    if n_p == 0 or n_q == 0:
        raise ValueError("alvos e candidatos nao podem ser vazios")
    n_s = len(targets[0])
    if any(len(row) != n_s for row in targets) or any(len(row) != n_s for row in candidates):
        raise ValueError("todos os niveis precisam do mesmo numero de conteudos")

    out = np.empty((n_p, n_q), dtype=float)
    for p in range(n_p):
        for q in range(n_q):
            out[p, q] = float(
                np.mean([spectral_distance(targets[p][s], candidates[q][s]) for s in range(n_s)])
            )
    return out


def best_match(matrix: np.ndarray) -> np.ndarray:
    """Nivel do arm que casa cada nivel da referencia."""
    return np.asarray(matrix, dtype=float).argmin(axis=1)


def contrast(matrix: np.ndarray) -> np.ndarray:
    """(pior - melhor) / melhor, por linha: so a razao e comparavel entre arms.

    Nao e invariante ao piso: um arm que nunca chega perto do alvo tem o contraste
    diluido, e a porteira deve reparar nisso.
    """
    m = np.asarray(matrix, dtype=float)
    best = m.min(axis=1)
    # Distancia zero e a referencia contra si mesma: contraste infinito.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(best > 0.0, (m.max(axis=1) - best) / np.where(best > 0.0, best, 1.0),
                       np.inf)
    return out


def deviation(matrix: np.ndarray) -> np.ndarray:
    """`q*(p) - p`. Negativo = o arm distorce mais que o nominal."""
    picks = best_match(matrix)
    return picks - np.arange(len(picks))


def is_discriminative(matrix: np.ndarray, threshold: float = CONTRAST_THRESHOLD) -> bool:
    """O eixo desse arm distingue os proprios niveis? Cobra o pior nivel, nao a media."""
    return bool(contrast(matrix).min() >= threshold)


def summarize(matrix: np.ndarray, threshold: float = CONTRAST_THRESHOLD) -> Dict[str, object]:
    """Resumo de um arm para relatorio e para a porteira."""
    values = contrast(matrix)
    return {
        "best_match": best_match(matrix).tolist(),
        "deviation": deviation(matrix).tolist(),
        "mean_deviation": float(np.mean(deviation(matrix))),
        "contrast": values.tolist(),
        "min_contrast": float(values.min()),
        "discriminative": bool(values.min() >= threshold),
    }
