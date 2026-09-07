"""Oraculo sonico: qual ajuste de B soa como este ajuste de A.

E a verdade fundamental do POC II. Como implementacoes de distorcao nao
compartilham unidade de ganho, a correspondencia entre elas nao pode ser
declarada -- tem de ser medida. O oraculo mede:

    q*(p) = argmin_q  media_s  D( x[A, p, s], x[B, q, s] )

sobre os mesmos conteudos `s`, com `D` = L1 sobre log-mel multi-resolucao
(Steinmetz et al. 2022). O pareamento por conteudo e o que faz isso funcionar: o
conteudo cancela na diferenca e sobra a diferenca de tratamento.

**Por que isso nao dispensa a rede.** O oraculo exige o sinal seco da referencia
tocando o mesmo trecho, que nao existe no uso real -- la a referencia e outra
execucao, em outro equipamento. O oraculo e professor, nao produto: a rede
aprende uma versao dele invariante ao conteudo.

Alem do alinhamento, este modulo mede o **contraste**: quanto a distancia varia
entre o melhor e o pior nivel do arm. Contraste alto = existe uma resposta certa.
Contraste baixo = os niveis daquele arm nao se distinguem, e a verdade
fundamental fica mal definida antes de qualquer rede entrar em cena. Isso serve
de porteira de aceitacao e pega um defeito que a monotonicidade NAO pega: o
`byod-hotfuzz` tinha rho 1,000 e a maior faixa de THD do roster, e mesmo assim
contraste de 27% no topo (contra ~1000% do `lsp-tanh`) -- eixo perceptualmente
degenerado.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Multi-resolucao porque uma janela so nao serve: janela curta resolve o
# transiente da palhetada e borra o corpo harmonico; janela longa faz o
# contrario. Distorcao muda as duas coisas.
FFT_SIZES: Tuple[int, ...] = (512, 1024, 2048, 4096)
N_MELS = 64
LOG_FLOOR = 1e-5

# Piso de contraste para um arm ser aceito: 1,0 quer dizer "o nivel que pior casa
# tem de estar ao menos ao DOBRO da distancia do que melhor casa".
#
# Ancorado no medido, nao escolhido a esmo: lsp-tanh (identidade) da 9,6;
# byod-mxr 3,1; byod-bigmuff 1,6; byod-hotfuzz, reprovado, 0,27. O corte em 1,0
# fica com folga de 1,6x abaixo do pior aprovado e 3,7x acima do reprovado.
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
    """L1 media entre duas pilhas log-mel.

    L1 e nao L2 porque L2 e dominada pelas poucas celulas de maior energia, e o
    que distingue quantidade de distorcao esta espalhado nos harmonicos fracos.
    """
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
    """(pior - melhor) / melhor, por linha.

    Relativo ao melhor e nao absoluto porque a escala da distancia nao tem
    unidade: so a razao e comparavel entre arms.

    Isto NAO e invariante ao piso, e vale dizer: um arm de timbre muito proprio
    -- o `byod-bigmuff` nunca chega perto de zero porque soa como um Big Muff --
    tem o contraste diluido pelo piso alto, e por isso ele marca 1,6 contra 9,6
    do `lsp-tanh`. A dependencia e aceita de proposito: se nem o melhor nivel
    chega perto do alvo, a correspondencia e fraca tambem em termos absolutos, e
    faz sentido a porteira reparar nisso.
    """
    m = np.asarray(matrix, dtype=float)
    best = m.min(axis=1)
    # Distancia zero acontece de verdade e nao e entrada degenerada: e a
    # referencia comparada consigo mesma. O limite do contraste ali e infinito,
    # que e a resposta certa -- casamento perfeito e perfeitamente discriminado.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(best > 0.0, (m.max(axis=1) - best) / np.where(best > 0.0, best, 1.0),
                       np.inf)
    return out


def deviation(matrix: np.ndarray) -> np.ndarray:
    """`q*(p) - p`. Negativo = o arm distorce mais que o nominal."""
    picks = best_match(matrix)
    return picks - np.arange(len(picks))


def is_discriminative(matrix: np.ndarray, threshold: float = CONTRAST_THRESHOLD) -> bool:
    """O eixo desse arm distingue os proprios niveis?

    Cobra o **pior** nivel, nao a media: um eixo que colapsa so no topo ja tem
    verdade fundamental mal definida la, e e justamente no topo que ele colapsa
    (o `byod-hotfuzz` ia de 195% em p0 para 27% em p7).
    """
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
