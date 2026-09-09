"""Leave-one-arm-out.

O teste que carrega o arquivo e `test_transfer_cost_compares_each_arm_with_itself`:
a comparacao errada -- contra a coluna `vistos`, que mistura 6 arms de
dificuldade muito diferente -- daria um custo de transferencia inflado, e e a
comparacao que um script rapido faria naturalmente.
"""
from __future__ import annotations

import json
from pathlib import Path

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


# --- curva de diversidade (B2 e B3) -------------------------------------------
def test_the_diversity_order_crosses_the_three_strata_in_order():
    """A ordem e o que da sentido a cada ponto: os dois primeiros sao S1, os tres
    seguintes trazem o S2 e o ultimo traz o S3. Sorteada, a curva mediria
    quantidade e variedade misturadas e sem rotulo."""
    from gefx.disent.loo import DIVERSITY_HELD_OUT, DIVERSITY_ORDER

    estratos = [STRATA[arm] for arm in DIVERSITY_ORDER]
    assert estratos == ["S1", "S1", "S2", "S2", "S2", "S3"]
    assert DIVERSITY_HELD_OUT not in DIVERSITY_ORDER
    assert STRATA[DIVERSITY_HELD_OUT] == "S3"


def test_the_curve_reuses_the_leave_one_out_run_at_its_last_point(tmp_path, monkeypatch):
    """O ultimo ponto e, por construcao, a execucao do leave-one-out para o mesmo
    arm. Retreina-lo daria um numero levemente diferente do ja publicado na etapa
    7, e a curva deixaria de terminar onde a etapa 7 termina."""
    from gefx.disent import loo as modulo

    reuse = tmp_path / "loo" / "byod-mxr"
    reuse.mkdir(parents=True)
    (reuse / "run.json").write_text("{}", encoding="utf-8")

    treinados = []
    monkeypatch.setattr(modulo, "_evaluate_held_out",
                        lambda run_dir, *a, **k: [{"condicao": "transferencia",
                                                   "drive_exact": 0.0, "mae_db": 0.0,
                                                   "run_dir": str(run_dir)}])

    def _fake_train(config, verbose=True):
        treinados.append(tuple(config.arms))
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir) / "run.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("gefx.disent.train.train", _fake_train)
    tabela = modulo.arm_diversity_curve(
        tmp_path, tmp_path / "curva", order=("a", "b", "c"), held_out="byod-mxr",
        reuse=tmp_path / "loo", verbose=False,
    )
    assert [len(arms) for arms in treinados] == [1, 2]  # o ponto k=3 nao retreinou
    assert tabela.iloc[-1]["run_dir"] == str(reuse)


def test_the_curve_keeps_the_catalog_fixed_while_the_training_set_grows(tmp_path, monkeypatch):
    """Se o catalogo crescesse junto, cada ponto responderia a uma pergunta
    diferente e a curva nao mediria diversidade de treino."""
    from gefx.disent import loo as modulo

    vistos = []
    monkeypatch.setattr(
        modulo, "_evaluate_held_out",
        lambda run_dir, root, held_out, seen, catalog_arms=None, batch=64, extra=None:
            vistos.append((tuple(seen), tuple(catalog_arms or seen))) or [],
    )
    monkeypatch.setattr("gefx.disent.train.train",
                        lambda config, verbose=True: (
                            Path(config.output_dir).mkdir(parents=True, exist_ok=True),
                            (Path(config.output_dir) / "run.json").write_text("{}"),
                        ))
    modulo.arm_diversity_curve(tmp_path, tmp_path / "curva", order=("a", "b", "c"),
                               held_out="z", reuse=None, verbose=False)
    assert [treino for treino, _ in vistos] == [("a",), ("a", "b"), ("a", "b", "c")]
    assert {catalogo for _, catalogo in vistos} == {("a", "b", "c")}


def test_the_transfer_cost_averages_the_seeds_and_shows_the_spread(tmp_path):
    resumo = _resumo()
    resumo["seed"] = 1
    outra = _resumo()
    outra["seed"] = 2
    outra.loc[outra["condicao"] == "transferencia", "drive_exact"] = [0.551, 0.159]
    custo = transfer_cost(pd.concat([resumo, outra]), _metrics(tmp_path)).set_index("arm")
    assert custo.loc["lsp-tanh", "sementes"] == 2
    assert custo.loc["lsp-tanh", "inedito"] == pytest.approx(0.541)
    assert custo.loc["lsp-tanh", "amplitude_pontos"] == pytest.approx(2.0, abs=0.05)
