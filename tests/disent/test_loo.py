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

from gefx.disent.loo import transfer_cost


def _resumo():
    return pd.DataFrame([
        {"arm_retirado": "lsp-tanh", "estrato": "S1", "condicao": "transferencia",
         "drive_exact": 0.531, "mae_db": 2.77, "n": 800},
        {"arm_retirado": "lsp-tanh", "estrato": "S1", "condicao": "vistos",
         "drive_exact": 0.395, "mae_db": 4.32, "n": 4800},
        {"arm_retirado": "byod-bigmuff", "estrato": "S3", "condicao": "transferencia",
         "drive_exact": 0.139, "mae_db": 11.18, "n": 800},
        {"arm_retirado": "byod-bigmuff", "estrato": "S3", "condicao": "vistos",
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
    """O custo sai de (arm inedito) menos (o MESMO arm visto no treino), onde a
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


def test_transfer_cost_carries_the_stratum_of_each_arm(tmp_path):
    custo = transfer_cost(_resumo(), _metrics(tmp_path)).set_index("arm")
    assert custo.loc["lsp-tanh", "estrato"] == "S1"
    assert custo.loc["byod-bigmuff", "estrato"] == "S3"


def test_an_arm_missing_from_the_reference_is_skipped(tmp_path):
    resumo = _resumo()
    resumo.loc[len(resumo)] = {"arm_retirado": "arm-novo", "condicao": "transferencia",
                               "drive_exact": 0.4, "mae_db": 3.0, "n": 800,
                               "estrato": "S2"}
    assert "arm-novo" not in set(transfer_cost(resumo, _metrics(tmp_path))["arm"])


def test_leave_one_out_needs_enough_arms(tmp_path):
    from gefx.disent.loo import leave_one_arm_out

    with pytest.raises(ValueError, match="ao menos 3"):
        leave_one_arm_out(tmp_path, tmp_path, arms=["a", "b"])


# --- curva de diversidade (B2 e B3) -------------------------------------------
def test_without_an_order_the_training_grows_stratum_by_stratum(tmp_path, monkeypatch):
    """A ordem e o que da sentido a cada ponto: primeiro S1, depois S2, depois S3.
    Sorteada, a curva mediria quantidade e variedade misturadas e sem rotulo."""
    from gefx.disent import loo as modulo

    monkeypatch.setattr(modulo, "strata", lambda root: {
        "s3-a": "S3", "s1-b": "S1", "s2-a": "S2", "s1-a": "S1", "fora": "S3"})
    vistos = []
    monkeypatch.setattr(modulo, "_evaluate_held_out",
                        lambda run_dir, root, held_out, seen, **k:
                            vistos.append(tuple(seen)) or [])
    monkeypatch.setattr("gefx.disent.train.train",
                        lambda config, verbose=True: (
                            Path(config.output_dir).mkdir(parents=True, exist_ok=True),
                            (Path(config.output_dir) / "run.json").write_text("{}"),
                        ))
    modulo.arm_diversity_curve("fora", tmp_path, tmp_path / "curva", reuse=None,
                               verbose=False)
    assert vistos[-1] == ("s1-a", "s1-b", "s2-a", "s3-a")


def test_the_curve_reuses_the_leave_one_out_run_at_its_last_point(tmp_path, monkeypatch):
    """O ultimo ponto e, por construcao, a execucao do leave-one-out para o mesmo
    arm. Retreina-lo daria um numero levemente diferente do ja publicado no
    leave-one-out, e a curva deixaria de terminar onde ele termina."""
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
    monkeypatch.setattr(modulo, "strata", lambda root: dict.fromkeys("abc", "S1"))
    tabela = modulo.arm_diversity_curve(
        "byod-mxr", tmp_path, tmp_path / "curva", order=("a", "b", "c"),
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
    monkeypatch.setattr(modulo, "strata", lambda root: dict.fromkeys("abc", "S1"))
    modulo.arm_diversity_curve("z", tmp_path, tmp_path / "curva", order=("a", "b", "c"),
                               reuse=None, verbose=False)
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


def test_leave_one_out_trains_on_the_dataset_it_evaluates(tmp_path, monkeypatch):
    """Sem isto o treino caia no dataset padrao e a avaliacao no pedido."""
    from gefx.disent import loo as modulo

    for arm in ("a", "b", "c"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "metadata.csv").write_text("")
    raizes = []
    monkeypatch.setattr(modulo, "_evaluate_held_out", lambda *a, **k: [])
    monkeypatch.setattr("gefx.disent.train.train",
                        lambda config, verbose=True: (
                            raizes.append(config.dataset_root),
                            Path(config.output_dir).mkdir(parents=True, exist_ok=True)))
    modulo.leave_one_arm_out(tmp_path, tmp_path / "loo", verbose=False)
    assert raizes == [tmp_path] * 3


def test_transfer_cost_leaves_out_the_centered_search(tmp_path):
    """O custo e o da busca como esta; a centrada e outra pergunta."""
    resumo = _resumo().assign(centrado=False)
    centrada = resumo.assign(centrado=True, drive_exact=0.99)
    custo = transfer_cost(pd.concat([resumo, centrada]), _metrics(tmp_path)).set_index("arm")
    assert custo.loc["lsp-tanh", "custo_pontos"] == pytest.approx(2.2, abs=0.05)
    assert int(custo.loc["lsp-tanh", "sementes"]) == 1
