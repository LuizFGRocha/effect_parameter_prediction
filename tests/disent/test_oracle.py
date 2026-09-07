"""Invariantes do oraculo sonico.

Sem plugin e sem dataset: as funcoes puras operam sobre pilhas de arrays, entao
da para construir os casos que importam a mao -- inclusive os patologicos que
motivaram o modulo.
"""
from __future__ import annotations

import numpy as np
import pytest

from gefx.disent.oracle import (
    CONTRAST_THRESHOLD,
    alignment_matrix,
    best_match,
    contrast,
    deviation,
    is_discriminative,
    log_mel_stack,
    spectral_distance,
    summarize,
)


def stack(value: float, shape=(4, 6)):
    """Pilha de duas resolucoes com valor constante -- distancia vira |dif|."""
    return (np.full(shape, value), np.full(shape, value))


def diagonal_matrix(n: int = 5, slope: float = 1.0):
    """Grade perfeitamente alinhada: minimo na diagonal, crescendo para os lados."""
    p = np.arange(n)[:, None]
    q = np.arange(n)[None, :]
    return slope * np.abs(p - q) + 0.1


# --- distancia ----------------------------------------------------------------
def test_distance_is_zero_for_identical_stacks():
    assert spectral_distance(stack(1.0), stack(1.0)) == 0.0


def test_distance_is_the_mean_absolute_difference():
    assert spectral_distance(stack(1.0), stack(3.0)) == pytest.approx(2.0)


def test_distance_is_symmetric():
    a, b = stack(0.5), stack(2.25)
    assert spectral_distance(a, b) == pytest.approx(spectral_distance(b, a))


def test_distance_rejects_mismatched_stacks():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        spectral_distance(stack(1.0), stack(1.0)[:1])


# --- matriz de alinhamento ----------------------------------------------------
def test_alignment_matrix_averages_over_contents():
    # dois conteudos, um nivel de cada lado: a distancia e a media dos dois.
    targets = [[stack(0.0), stack(0.0)]]
    candidates = [[stack(1.0), stack(3.0)]]
    assert alignment_matrix(targets, candidates)[0, 0] == pytest.approx(2.0)


def test_alignment_matrix_rejects_ragged_contents():
    targets = [[stack(0.0), stack(0.0)], [stack(0.0)]]
    candidates = [[stack(1.0), stack(1.0)], [stack(1.0), stack(1.0)]]
    with pytest.raises(ValueError, match="mesmo numero de conteudos"):
        alignment_matrix(targets, candidates)


# --- alinhamento e desvio -----------------------------------------------------
def test_perfect_grid_recovers_the_diagonal():
    # E a checagem de identidade: `lsp-tanh` tem a mesma curva e a mesma unidade
    # da referencia, entao o oraculo TEM de devolver a diagonal. Se nao devolver,
    # a metrica esta errada antes de qualquer conclusao apoiada nela.
    matrix = diagonal_matrix(8)
    assert best_match(matrix).tolist() == list(range(8))
    assert deviation(matrix).tolist() == [0] * 8


def test_deviation_is_negative_when_the_arm_distorts_more():
    # Arm deslocado: o nivel q casa o nivel q+1 da referencia, ou seja, ele chega
    # na mesma quantidade de distorcao com menos knob.
    n = 6
    p = np.arange(n)[:, None]
    q = np.arange(n)[None, :]
    matrix = np.abs(p - (q + 1)) + 0.1
    assert deviation(matrix)[1:-1].mean() < 0


# --- contraste, que e a porteira ---------------------------------------------
def test_contrast_is_high_for_a_sharp_minimum():
    assert contrast(diagonal_matrix(8, slope=1.0)).min() > CONTRAST_THRESHOLD


def test_contrast_collapses_for_a_flat_axis():
    # O caso `byod-hotfuzz`: os niveis do arm nao se distinguem, entao nao existe
    # resposta certa para "qual nivel casa" -- e nenhuma rede conserta isso.
    matrix = np.full((8, 8), 0.5) + np.random.default_rng(0).normal(0, 1e-3, (8, 8))
    assert contrast(matrix).max() < 0.05
    assert not is_discriminative(matrix)


def test_a_high_floor_dilutes_the_contrast_but_still_passes():
    # O caso `byod-bigmuff`: timbre proprio poe um piso alto, o contraste cai de
    # 9,6 para 1,6, e ainda assim ha um minimo destacado. A porteira tem de deixar
    # passar -- mas o teste registra que a diluicao existe, porque a medida NAO e
    # invariante ao piso.
    # Piso da ordem do alcance da curva, que e a proporcao medida no bigmuff
    # (piso 0,5-0,8 contra alcance ~1,5).
    limpo = diagonal_matrix(8, slope=1.0)
    com_piso = limpo + 2.0
    assert contrast(com_piso).min() < contrast(limpo).min()
    assert is_discriminative(com_piso)


def test_gate_cobra_o_pior_nivel_nao_a_media():
    # Eixo que so colapsa no topo: a media do contraste ainda passaria, o minimo
    # nao. E no topo que o hotfuzz colapsava.
    matrix = diagonal_matrix(8, slope=1.0)
    matrix[-1, :] = 0.5                      # ultima linha achatada
    assert contrast(matrix).mean() > CONTRAST_THRESHOLD
    assert not is_discriminative(matrix)


def test_zero_distance_gives_infinite_contrast():
    # E a referencia comparada consigo mesma: casamento exato, discriminacao
    # perfeita. Nao e entrada invalida.
    values = contrast(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.all(np.isinf(values))
    assert is_discriminative(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_summarize_reports_every_field_the_gate_uses():
    out = summarize(diagonal_matrix(5))
    assert out["best_match"] == [0, 1, 2, 3, 4]
    assert out["deviation"] == [0] * 5
    assert out["discriminative"] is True
    assert out["min_contrast"] == pytest.approx(min(out["contrast"]))


# --- log-mel ------------------------------------------------------------------
def test_log_mel_stack_has_one_array_per_resolution():
    sr = 22050
    t = np.arange(sr) / sr
    audio = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    stacks = log_mel_stack(audio, sr, fft_sizes=(512, 1024), n_mels=16)
    assert len(stacks) == 2
    assert all(s.shape[0] == 16 for s in stacks)
    # Janela maior -> menos quadros, porque o hop acompanha o n_fft.
    assert stacks[0].shape[1] > stacks[1].shape[1]


def test_log_mel_stack_rejects_stereo():
    with pytest.raises(ValueError, match="mono"):
        log_mel_stack(np.zeros((2, 1000)), 22050, fft_sizes=(512,))
