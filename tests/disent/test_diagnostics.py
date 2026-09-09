"""Sondas lineares sobre os codigos.

O modulo existe por causa de um resultado da etapa 5: o adversario de
implementacao ficou no acaso do primeiro passo ao ultimo e mesmo assim uma sonda
le a implementacao bem acima do acaso no mesmo `z_e`. Perda de adversario nao e
evidencia de remocao -- e a sonda que decide.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from gefx.disent.diagnostics import PROBE_FACTORS, linear_probes, probe_study


def _frame(n_arms=2, n_contents=4, n_drive=2, repeats=6):
    rows = []
    for arm, content, drive, _ in itertools.product(
        range(n_arms), range(n_contents), range(n_drive), range(repeats)
    ):
        rows.append({"arm": f"a{arm}", "content_id": f"c{content}", "drive_level": drive})
    return pd.DataFrame(rows)


def test_a_code_that_encodes_the_factor_is_read_at_the_ceiling():
    frame = _frame()
    codes = np.stack(
        [frame["drive_level"].to_numpy(dtype=float) * 10.0,
         np.zeros(len(frame))], axis=1
    )
    probes = linear_probes(codes, frame, factors=["drive_level"])
    assert probes["drive_level"]["accuracy"] == pytest.approx(1.0)
    assert probes["drive_level"]["above_chance"] == pytest.approx(1.0)


def test_a_code_that_carries_nothing_lands_at_chance():
    # Amostras de sobra: com poucas linhas por classe a sonda decora o ruido e
    # sai do acaso sozinha, que e um artefato do teste e nao do codigo.
    frame = _frame(repeats=40)
    codes = np.random.default_rng(0).normal(size=(len(frame), 3))
    probes = linear_probes(codes, frame, factors=["arm"])
    assert probes["arm"]["chance"] == pytest.approx(0.5)
    assert probes["arm"]["above_chance"] < 0.3


def test_above_chance_makes_factors_with_different_class_counts_comparable():
    """25% em 7 classes e 15% em 20 nao sao comparaveis crus; e a fracao do
    caminho entre acaso e acerto total que torna as duas linhas legiveis."""
    frame = _frame()
    codes = np.stack([frame["drive_level"].to_numpy(dtype=float), np.zeros(len(frame))], axis=1)
    probes = linear_probes(codes, frame, factors=["drive_level", "arm"])
    for numbers in probes.values():
        esperado = (numbers["accuracy"] - numbers["chance"]) / (1.0 - numbers["chance"])
        assert numbers["above_chance"] == pytest.approx(esperado)


def test_the_three_factors_cover_both_halves_of_the_claim():
    """Sondar so o que deve sair mediria metade da afirmacao: um codigo
    constante zera implementacao e conteudo e nao serve para nada."""
    assert set(PROBE_FACTORS) == {"arm", "content_id", "drive_level"}


def test_a_factor_missing_from_the_sidecar_is_refused():
    with pytest.raises(KeyError, match="tempo"):
        linear_probes(np.zeros((len(_frame()), 2)), _frame(), factors=["tempo"])


def test_codes_and_rows_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="codigos"):
        linear_probes(np.zeros((3, 2)), _frame())


def test_the_probe_is_deterministic_in_the_seed():
    frame = _frame()
    codes = np.random.default_rng(1).normal(size=(len(frame), 4))
    first = linear_probes(codes, frame, factors=["arm"], seed=5)
    second = linear_probes(codes, frame, factors=["arm"], seed=5)
    assert first == second


def test_probe_study_refuses_a_directory_without_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.json"):
        probe_study(tmp_path, tmp_path)


def test_probe_study_skips_techniques_that_were_not_run(tmp_path, monkeypatch):
    from gefx.disent import diagnostics

    (tmp_path / "contrastive").mkdir()
    (tmp_path / "contrastive" / "run.json").write_text(
        json.dumps({"config": {"technique": "contrastive"}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        diagnostics, "probe_run",
        lambda *args, **kwargs: {"probes": {"arm": {"accuracy": 0.3, "chance": 0.14,
                                                    "classes": 7, "above_chance": 0.2}}},
    )
    table = probe_study(tmp_path, tmp_path)
    assert list(table["technique"]) == ["contrastive"]
    assert list(table["factor"]) == ["arm"]


# --- ablacao da representacao -------------------------------------------------
def test_the_ablation_scorer_reports_the_three_numbers_the_table_needs():
    """`_score` e o unico pedaco da ablacao que faz conta; o resto e montagem.
    Ele passa pelo mesmo `retrieve_by_arm` do estudo, entao o que se testa aqui e
    que os tres numeros da tabela saem de la, e nao que a busca funciona."""
    from gefx.disent.diagnostics import _score

    arms = ["a0", "a1"]
    def tabela(contents, split):
        return pd.DataFrame([
            {"file_name": f"{c}_{d}_{a}.wav", "arm": a, "content_id": c, "split": split,
             "drive_level": d, "tone_level": 0, "drive_db_equivalente": 10.0 + 5 * d}
            for c, d, a in itertools.product(contents, (0, 1), arms)
        ])
    frames = {"query": tabela(["q0", "q1"], "query"),
              "catalog": tabela(["k0", "k1"], "catalog")}
    rng = np.random.default_rng(0)
    numeros = _score(frames,
                     rng.normal(size=(len(frames["query"]), 3)).astype(np.float32),
                     rng.normal(size=(len(frames["catalog"]), 3)).astype(np.float32))
    assert set(numeros) == {"drive_exact", "mae_db", "top_arm_share"}
    assert 0.0 <= numeros["drive_exact"] <= 1.0
    assert 0.0 < numeros["top_arm_share"] <= 1.0


def test_the_ablation_reduces_to_the_dimension_the_encoder_uses():
    """A ablacao so responde a pergunta se o degrau linear terminar na MESMA
    largura do `z_e`; em outra largura ela mediria outra coisa."""
    from gefx.disent.diagnostics import ABLATION_DIMS
    from gefx.disent.model import EncoderConfig

    assert ABLATION_DIMS == (EncoderConfig().effect_dim,)


# --- resolucao da grade -------------------------------------------------------
def test_the_level_subsets_all_keep_the_axis_ends_and_a_uniform_step():
    """Subconjunto com passo irregular mediria resolucao misturada com posicao no
    eixo, e o eixo nao e uniforme em dificuldade -- a metade de cima e mais
    discriminavel."""
    from gefx.disent.diagnostics import LEVEL_SUBSETS

    for nome, niveis in LEVEL_SUBSETS.items():
        passos = {b - a for a, b in zip(niveis, niveis[1:])}
        assert len(passos) == 1, f"{nome}: passos {passos}"
        assert set(niveis) <= set(range(8))


def test_there_are_two_subsets_of_the_same_size_shifted_by_one_level():
    """Pares e impares tem o mesmo passo e o mesmo tamanho e diferem so em onde
    o eixo comeca -- e o unico par que isola a POSICAO da resolucao."""
    from gefx.disent.diagnostics import LEVEL_SUBSETS

    pares = LEVEL_SUBSETS["4 niveis pares (8,3 dB)"]
    impares = LEVEL_SUBSETS["4 niveis impares (8,3 dB)"]
    assert len(pares) == len(impares)
    assert [b - a for a, b in zip(pares, pares[1:])] == [
        b - a for a, b in zip(impares, impares[1:])
    ]
    assert all(i - p == 1 for p, i in zip(pares, impares))


# --- DCI e MIG ----------------------------------------------------------------
def _structured_frame(repeats=8):
    """Grade cruzada com os quatro fatores das metricas de estrutura."""
    rows = []
    for arm, content, drive, tone, _ in itertools.product(
        range(3), range(4), range(4), range(2), range(repeats)
    ):
        rows.append({"arm": f"a{arm}", "content_id": f"c{content}",
                     "drive_level": drive, "tone_level": tone})
    return pd.DataFrame(rows)


def _planted(frame, seed=0):
    """Codigo com um fator por dimensao e ruido no resto: o caso em que as
    metricas de estrutura devem sair no teto."""
    rng = np.random.default_rng(seed)
    return np.stack([
        frame["drive_level"].to_numpy(dtype=float),
        frame["tone_level"].to_numpy(dtype=float),
        frame["arm"].astype("category").cat.codes.to_numpy(dtype=float),
        frame["content_id"].astype("category").cat.codes.to_numpy(dtype=float),
        rng.normal(size=len(frame)),
    ], axis=1)


def test_one_factor_per_dimension_reads_as_disentangled():
    from gefx.disent.diagnostics import dci_scores

    frame = _structured_frame()
    scores = dci_scores(_planted(frame), frame, trees=40)
    # Nao 1,0: a importancia de Gini vaza para a dimensao de ruido, que e
    # continua e de cardinalidade alta e por isso oferece muitos cortes. O teto
    # pratico com um codigo plantado perfeito fica na casa de 0,8 -- e o que
    # torna a leitura possivel e a distancia ate o codigo misturado, abaixo.
    assert scores["disentanglement"] > 0.75
    for factor, value in scores["completeness"].items():
        assert value > 0.6, factor
    for factor, numbers in scores["informativeness"].items():
        assert numbers["accuracy"] > 0.95, factor


def test_a_code_that_repeats_every_factor_everywhere_reads_as_entangled():
    """O contraste que da sentido ao numero anterior: mesma informacao, espalhada
    por todas as dimensoes. A informatividade continua no teto -- e por isso que
    ela sozinha nao mede desemaranhamento."""
    from gefx.disent.diagnostics import dci_scores

    frame = _structured_frame()
    planted = _planted(frame)
    rng = np.random.default_rng(1)
    mistura = planted @ rng.normal(size=(planted.shape[1], planted.shape[1]))
    scores = dci_scores(mistura, frame, trees=40)
    assert scores["disentanglement"] < 0.2
    assert scores["informativeness"]["drive_level"]["accuracy"] > 0.9


def test_the_block_mass_finds_the_block_that_carries_the_factor():
    """E esta a afirmacao do trabalho: nao que cada dimensao carregue um fator,
    e sim que os fatores de configuracao estejam em `z_e` e o resto em `z_c`."""
    from gefx.disent.diagnostics import dci_scores

    frame = _structured_frame()
    codes = _planted(frame)
    # Duas primeiras dimensoes = z_e (configuracao), o resto = z_c.
    blocks = {"z_e": range(2), "z_c": range(2, codes.shape[1])}
    scores = dci_scores(codes, frame, blocks=blocks, trees=40)
    assert scores["block_mass"]["drive_level"]["z_e"] > 0.85
    assert scores["block_mass"]["tone_level"]["z_e"] > 0.85
    assert scores["block_mass"]["arm"]["z_c"] > 0.9
    assert scores["block_mass"]["content_id"]["z_c"] > 0.9
    assert min(scores["block_completeness"].values()) > 0.4


def test_the_block_mass_of_each_factor_is_a_partition():
    from gefx.disent.diagnostics import dci_scores

    frame = _structured_frame()
    codes = _planted(frame)
    scores = dci_scores(codes, frame, blocks={"z_e": range(2), "z_c": range(2, 5)},
                        trees=40)
    for factor, shares in scores["block_mass"].items():
        assert sum(shares.values()) == pytest.approx(1.0), factor


def test_the_mig_points_at_the_dimension_that_holds_the_factor():
    from gefx.disent.diagnostics import mutual_information_gap

    frame = _structured_frame()
    gaps = mutual_information_gap(_planted(frame), frame)
    assert gaps["drive_level"]["top_latent"] == 0
    assert gaps["tone_level"]["top_latent"] == 1
    assert gaps["arm"]["top_latent"] == 2
    assert gaps["drive_level"]["mig"] > 0.5


def test_the_mig_collapses_when_the_factor_is_duplicated_across_dimensions():
    """MIG mede a **distancia** entre as duas melhores dimensoes: duplicar o
    fator zera o numero sem tirar informacao nenhuma do codigo. E a razao de ele
    entrar como descricao e nao como criterio."""
    from gefx.disent.diagnostics import mutual_information_gap

    frame = _structured_frame()
    codes = _planted(frame)
    duplicado = np.concatenate([codes, codes[:, :1]], axis=1)
    gaps = mutual_information_gap(duplicado, frame)
    assert gaps["drive_level"]["mig"] == pytest.approx(0.0, abs=1e-6)


def test_a_dead_dimension_does_not_break_the_binning():
    from gefx.disent.diagnostics import mutual_information_gap

    frame = _structured_frame()
    codes = np.concatenate([_planted(frame), np.zeros((len(frame), 1))], axis=1)
    gaps = mutual_information_gap(codes, frame)
    assert gaps["arm"]["top_latent"] == 2


def test_the_structure_metrics_refuse_a_mismatched_number_of_rows():
    from gefx.disent.diagnostics import dci_scores, mutual_information_gap

    frame = _structured_frame(repeats=1)
    codes = np.zeros((len(frame) + 1, 3))
    with pytest.raises(ValueError, match="codigos"):
        dci_scores(codes, frame, trees=10)
    with pytest.raises(ValueError, match="codigos"):
        mutual_information_gap(codes, frame)


def test_the_structure_factors_cover_what_each_block_should_hold():
    from gefx.disent.diagnostics import STRUCTURE_FACTORS

    assert set(STRUCTURE_FACTORS) == {"drive_level", "tone_level", "arm", "content_id"}


def test_the_forest_credits_the_noise_dimension_more_than_the_wrong_factor():
    """Uma ressalva a registrar, nao um defeito a esconder: a importancia de Gini
    prefere dimensoes continuas de cardinalidade alta, entao a dimensao de ruido
    recebe mais credito do que as dimensoes que carregam os outros fatores. Ler
    um `D` de 0,8 como "20% emaranhado" seria ler o vies do estimador."""
    from gefx.disent.diagnostics import dci_scores

    frame = _structured_frame()
    scores = dci_scores(_planted(frame), frame, trees=40)
    importancia = scores["importance"]
    coluna = list(scores["factors"]).index("drive_level")
    assert importancia[-1, coluna] > importancia[1, coluna]
