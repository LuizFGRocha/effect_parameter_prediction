"""Leave-one-arm-out.

O teste que carrega o arquivo e `test_transfer_cost_compares_each_arm_with_itself`:
a comparacao errada -- contra a coluna `vistos`, que mistura 6 arms de
dificuldade muito diferente -- daria um custo de transferencia inflado, e e a
comparacao que um script rapido faria naturalmente.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gefx.disent.loo import STRATA, transfer_cost


def _resumo():
    return pd.DataFrame([
        {"arm_retirado": "lsp-tanh", "condicao": "transferencia",
         "drive_exact": 0.531, "mae_db": 2.77, "n": 800},
        {"arm_retirado": "lsp-tanh", "condicao": "vistos",
         "drive_exact": 0.395, "mae_db": 4.32, "n": 4800},
        {"arm_retirado": "byod-bigmuff", "condicao": "transferencia",
         "drive_exact": 0.139, "mae_db": 11.18, "n": 800},
        {"arm_retirado": "byod-bigmuff", "condicao": "vistos",
         "drive_exact": 0.533, "mae_db": 2.44, "n": 4800},
    ])


def _metrics(tmp_path):
    caminho = tmp_path / "metrics.json"
    caminho.write_text(json.dumps({"per_query_arm": {
        "lsp-tanh": {"drive_level": {"exact": 0.509}, "mae_db": 2.863},
        "byod-bigmuff": {"drive_level": {"exact": 0.228}, "mae_db": 7.189},
    }}), encoding="utf-8")
    return caminho


def test_transfer_cost_compares_each_arm_with_itself(tmp_path):
    """O custo sai de (arm inedito) menos (o MESMO arm na etapa 5), onde a
    pergunta e o catalogo sao identicos. Contra a coluna `vistos` o
    `byod-bigmuff` sairia com -39 pontos em vez de -8,9, so porque ele proprio
    puxa a media dos vistos para baixo quando esta dentro dela."""
    custo = transfer_cost(_resumo(), _metrics(tmp_path)).set_index("arm")
    assert custo.loc["lsp-tanh", "custo_pontos"] == pytest.approx(2.2, abs=0.05)
    assert custo.loc["byod-bigmuff", "custo_pontos"] == pytest.approx(-8.9, abs=0.05)


def test_transfer_cost_only_reads_the_transfer_rows(tmp_path):
    custo = transfer_cost(_resumo(), _metrics(tmp_path))
    assert len(custo) == 2
    assert set(custo["arm"]) == {"lsp-tanh", "byod-bigmuff"}


def test_transfer_cost_labels_the_stratum_even_without_the_column(tmp_path):
    """O resumo pode ter sido gravado por uma versao sem a coluna; o estrato e o
    eixo pelo qual o resultado se organiza e nao pode depender disso."""
    custo = transfer_cost(_resumo(), _metrics(tmp_path)).set_index("arm")
    assert custo.loc["lsp-tanh", "estrato"] == "S1"
    assert custo.loc["byod-bigmuff", "estrato"] == "S3"


def test_an_arm_missing_from_the_reference_is_skipped(tmp_path):
    resumo = _resumo()
    resumo.loc[len(resumo)] = {"arm_retirado": "arm-novo", "condicao": "transferencia",
                               "drive_exact": 0.4, "mae_db": 3.0, "n": 800}
    assert "arm-novo" not in set(transfer_cost(resumo, _metrics(tmp_path))["arm"])


def test_every_arm_of_the_roster_has_a_stratum():
    from gefx.disent.arms import arm_keys

    assert set(arm_keys()) == set(STRATA)
    assert set(STRATA.values()) == {"S1", "S2", "S3"}


def test_leave_one_out_needs_enough_arms(tmp_path):
    from gefx.disent.loo import leave_one_arm_out

    with pytest.raises(ValueError, match="ao menos 3"):
        leave_one_arm_out(tmp_path, tmp_path, arms=["a", "b"])
