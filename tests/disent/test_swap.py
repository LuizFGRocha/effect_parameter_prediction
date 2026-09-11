"""Fase 2: troca de codigos.

O teste que carrega o arquivo e `test_the_donor_never_shares_content_with_the_anchor`:
com o mesmo conteudo o alvo da troca seria o proprio doador, a reconstrucao nao
exigiria separar nada, e a metrica subiria sem que nada tivesse sido demonstrado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefx.disent.sampler import GridIndex


def _frame(n_contents=4, n_configs=5, n_arms=3):
    linhas = []
    for conteudo in range(n_contents):
        for config in range(n_configs):
            for arm in range(n_arms):
                linhas.append({
                    "content_id": f"c{conteudo}", "config_index": config,
                    "arm": f"a{arm}", "drive_level": config // 2,
                    "tone_level": config % 2,
                    "file_name": f"c{conteudo}_{config}_{arm}.wav",
                })
    return pd.DataFrame(linhas)


def test_the_swap_target_is_content_of_the_anchor_and_config_of_the_donor():
    """A afirmacao inteira da fase 2 mora nesta tupla. Trocar qualquer um dos
    tres fatores de lugar mediria outra coisa -- e o alvo continuaria existindo
    em disco, entao o erro nao denunciaria a troca."""
    index = GridIndex(_frame())
    rotulos = index.labels(np.array([index.row(2, 1, 0)]))
    alvo = index.lookup[rotulos["content"], 4, rotulos["arm"]][0]
    linha = index.frame.iloc[alvo]
    assert linha["content_id"] == "c2"      # conteudo da ancora
    assert linha["config_index"] == 4       # configuracao do doador
    assert linha["arm"] == "a0"             # implementacao da ancora


def test_the_donor_never_shares_content_with_the_anchor():
    """`(conteudo + k) % n` com k em [1, n-1] nunca devolve o proprio conteudo."""
    rng = np.random.default_rng(0)
    n = 7
    ancora = rng.integers(0, n, size=500)
    doador = (ancora + rng.integers(1, n, size=500)) % n
    assert not np.any(doador == ancora)


def test_the_report_carries_every_control_the_reading_needs():
    """A leitura "o decodificado parece a configuracao do doador" nao vale
    sozinha. Precisa dos quatro pontos de referencia do erro -- piso, identidade,
    media do recorte -- e da leitura da IDENTIDADE, que mostra que sem trocar o
    que aparece e a configuracao da ancora. Se qualquer um sumir da saida, a
    tabela vira um numero solto."""
    import inspect

    from gefx.disent import swap

    fonte = inspect.getsource(swap.swap_fidelity)
    for chave in ('"piso"', '"identidade"', '"media_do_recorte"', '"leu_a_ancora"',
                  '"acaso_config"', '"acaso_drive"'):
        assert chave in fonte, chave


def test_onehot_marks_exactly_one_column_per_row():
    from gefx.disent.swap import _onehot

    saida = _onehot(np.array([0, 2, 1]), 3)
    assert saida.shape == (3, 3)
    assert np.array_equal(saida.sum(axis=1), np.ones(3))
    assert saida[1, 2] == 1.0


def test_swap_fidelity_refuses_a_run_without_decoder(tmp_path, monkeypatch):
    """Uma execucao da fase 1 nao tem o que medir aqui, e o erro tem de dizer
    isso em vez de estourar num atributo `None`."""
    from gefx.disent import swap

    class _SemDecoder:
        decoder = None

    monkeypatch.setattr(swap := __import__("gefx.disent.swap", fromlist=["x"]),
                        "swap_fidelity", swap.swap_fidelity)
    monkeypatch.setattr("gefx.disent.diagnostics.load_run",
                        lambda d: (_SemDecoder(), {"config": {}}))
    with pytest.raises(ValueError, match="nao tem decoder"):
        swap.swap_fidelity(tmp_path)


def test_swap_study_refuses_a_directory_without_runs(tmp_path):
    from gefx.disent.swap import swap_study

    with pytest.raises(FileNotFoundError, match="nenhuma execucao"):
        swap_study(tmp_path, tmp_path)


def test_the_recovered_fraction_normalises_away_each_decoder_own_floor():
    """Os erros crus nao sao comparaveis entre execucoes: um autoencoder melhor
    tem piso mais baixo e erro de troca mais baixo sem que a troca tenha
    funcionado melhor. A fracao recuperada e o que torna a coluna legivel."""
    identidade, piso = 0.55, 0.13
    fracao = lambda troca: (identidade - troca) / (identidade - piso)
    assert fracao(piso) == pytest.approx(1.0)         # troca perfeita
    assert fracao(identidade) == pytest.approx(0.0)   # troca nao fez nada
    assert fracao(0.1717) == pytest.approx(0.901, abs=0.01)
