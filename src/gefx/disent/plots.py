"""Figuras do POC II, lidas dos CSV/JSON gravados: nada aqui recalcula resultado.

Os arms aparecem sempre por estrato e, dentro dele, por nome; estrato, acaso e
passo da grade saem das metricas gravadas, nao do roster.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: isto roda em terminal e em CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POC1_MAE_DB = 1.28  # erro do POC I dentro da propria implementacao

COLOR_B0 = "#8c8c8c"
COLOR_B1 = "#1f77b4"
COLOR_CHANCE = "#c44e52"


def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _pct(fraction: float) -> str:
    """Rotulo de porcentagem com virgula decimal: 0.21 -> '21,0%'."""
    return f"{fraction * 100:.1f}%".replace(".", ",")


def _db(value: float) -> str:
    return f"{value:.2f} dB".replace(".", ",")


def ordered_arms(present: Sequence[str], strata: Mapping[str, str]) -> List[str]:
    """Por estrato e, dentro dele, por nome: a mesma ordem em toda figura."""
    return sorted(set(present), key=lambda arm: (strata.get(arm, "~"), arm))


def grid_step_db(metrics: Mapping[str, object]) -> float:
    """Passo medio da escada de dB da referencia."""
    return float(np.mean(np.diff(metrics["drive_db_ladder"])))  # type: ignore[arg-type]


def drive_chance(metrics: Mapping[str, object]) -> float:
    return 1.0 / metrics["alphabet"]["drive_level"]  # type: ignore[index]


def _short(key: str) -> str:
    """Rotulo de eixo; mantem o prefixo para `lsp-tanh` nao se confundir com `pedalboard-tanh`."""
    return key.replace("pedalboard-", "pb-")


# --- 1. B0 x B1 por arm -------------------------------------------------------
def plot_baselines_by_arm(
    b0_metrics: Dict[str, object],
    b1_metrics: Dict[str, object],
    out_path: Path,
) -> None:
    """Acerto exato e erro em dB dos dois baselines, arm a arm, com o acaso e o passo da grade."""
    arms = ordered_arms(list(b0_metrics["per_query_arm"]), b0_metrics["strata"])
    passo = grid_step_db(b0_metrics)
    b0 = [b0_metrics["per_query_arm"][a]["drive_level"]["exact"] * 100 for a in arms]
    b1 = [b1_metrics["per_query_arm"][a]["drive_level"]["exact"] * 100 for a in arms]
    b0_db = [b0_metrics["per_query_arm"][a]["mae_db"] for a in arms]
    b1_db = [b1_metrics["per_query_arm"][a]["mae_db"] for a in arms]
    chance = b0_metrics["per_query_arm"][arms[0]]["drive_level"]["chance"] * 100

    positions = np.arange(len(arms))
    width = 0.38
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    top.bar(positions - width / 2, b0, width, label="B0 — vizinho mais próximo",
            color=COLOR_B0)
    top.bar(positions + width / 2, b1, width, label="B1 — regressor do POC I",
            color=COLOR_B1)
    top.axhline(chance, color=COLOR_CHANCE, linestyle="--", linewidth=1.2)
    top.annotate(f"acaso\n({chance:.1f}%)", xy=(-1.3, chance), xytext=(2, -6),
                 textcoords="offset points", ha="left", color=COLOR_CHANCE,
                 fontsize=9)
    # Margem a esquerda para as anotacoes das linhas de referencia.
    top.set_xlim(-1.35, len(arms) - 0.4)
    top.set_ylim(0, max(max(b0), max(b1)) * 1.18)
    top.set_ylabel("acerto exato do nível de drive (%)")
    top.set_title(f"Baselines por implementação — "
                  f"{b0_metrics['alphabet']['drive_level']} níveis de drive")
    top.legend(loc="upper left")

    bottom.bar(positions - width / 2, b0_db, width, color=COLOR_B0)
    bottom.bar(positions + width / 2, b1_db, width, color=COLOR_B1)
    bottom.axhline(passo, color=COLOR_CHANCE, linestyle="--", linewidth=1.2)
    bottom.annotate(f"um degrau\nda grade\n({passo:.2f} dB)",
                    xy=(-1.3, passo), xytext=(2, 5),
                    textcoords="offset points", ha="left",
                    color=COLOR_CHANCE, fontsize=9)
    bottom.axhline(POC1_MAE_DB, color="#555555", linestyle=":", linewidth=1.2)
    bottom.annotate(f"POC I, na própria\nimplementação\n({POC1_MAE_DB} dB)",
                    xy=(-1.3, POC1_MAE_DB), xytext=(2, 5),
                    textcoords="offset points", ha="left",
                    color="#555555", fontsize=9)
    bottom.set_ylim(0, max(b0_db) * 1.12)
    bottom.set_ylabel("erro médio (dB equivalentes)")
    bottom.set_xticks(positions)
    bottom.set_xticklabels([_short(a) for a in arms], rotation=25, ha="right")

    _save(fig, out_path)


# --- a escada ------------------------------------------------------------------
COLOR_LEARNED = "#4c72b0"
COLOR_RANDOM = "#937860"
COLOR_CEILING = "#55a868"

TECHNIQUE_LABEL = {
    "random_encoder": "encoder\nnão treinado",
    "bn_only": "não treinado\n+ BatchNorm calibrada",
    "supcon": "encoder treinado\n(SupCon)",
    "rnc": "encoder treinado\n(Rank-N-Contrast)",
}


def plot_ladder(tabela: pd.DataFrame, passo_db: float, out_path: Path) -> None:
    """Acerto e erro por tecnica, media das sementes com cada semente como ponto,
    e os baselines como linhas."""
    from gefx.disent.train import TECHNIQUES

    execucoes = tabela[tabela["technique"] != "baseline"]
    names = [name for name in TECHNIQUES if name in set(execucoes["technique"])]
    posicoes = np.arange(len(names))
    cores = [COLOR_LEARNED if name in ("rnc", "supcon") else COLOR_RANDOM for name in names]
    rotulos = [TECHNIQUE_LABEL.get(name, name) for name in names]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    for eixo, coluna, escala, formato in ((left, "drive_exact", 100, "{:.1f}"),
                                          (right, "mae_db", 1, "{:.2f}")):
        medias = []
        for x, name in enumerate(names):
            valores = execucoes.loc[execucoes["technique"] == name, coluna] * escala
            medias.append(float(valores.mean()))
            if len(valores) > 1:
                eixo.plot([x] * len(valores), valores, "o", color="k", markersize=4,
                          alpha=0.6, zorder=3)
        eixo.bar(posicoes, medias, color=cores)
        # Rotulo dentro da barra: por cima, cairia nas linhas de referencia.
        for x, valor in zip(posicoes, medias):
            eixo.annotate(formato.format(valor), xy=(x, valor), xytext=(0, -14),
                          textcoords="offset points", ha="center", fontsize=9,
                          color="white", fontweight="bold")
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(rotulos, fontsize=8)

    baselines = tabela[tabela["technique"] == "baseline"].set_index("run")
    linhas = (("chance", COLOR_CHANCE, ":", "acaso"),
              ("B0", COLOR_B0, "--", "B0 sem aprendizado"),
              ("B1", COLOR_B1, "--", "B1 regressor do POC I"))
    for chave, cor, estilo, texto in linhas:
        if chave in baselines.index:
            valor = float(baselines.loc[chave, "drive_exact"])
            left.axhline(valor * 100, color=cor, linestyle=estilo, linewidth=1.4,
                         label=f"{texto} ({_pct(valor)})")
    left.set_ylabel("acerto exato do nível de drive (%)")
    left.set_ylim(0, 75)
    left.set_title("Recuperação entre implementações")
    left.legend(fontsize=8, loc="upper left")

    for chave, cor, estilo, texto in linhas[1:]:
        if chave in baselines.index:
            valor = float(baselines.loc[chave, "mae_db"])
            right.axhline(valor, color=cor, linestyle=estilo, linewidth=1.4,
                          label=f"{texto} ({_db(valor)})")
    right.axhline(passo_db, color="k", linestyle=":", linewidth=1.2,
                  label=f"um degrau da grade ({passo_db:.2f} dB)")
    right.set_ylabel("erro médio (dB equivalentes)")
    right.set_ylim(0, max(tabela["mae_db"].max(), passo_db) * 1.15)
    right.set_title("Distância do ajuste certo")
    right.legend(fontsize=8)

    _save(fig, out_path)


def plot_by_arm(metrics: Dict[str, object], out_path: Path) -> None:
    """Acerto por implementacao consultada, com a media e o acaso."""
    por_arm = metrics["per_query_arm"]  # type: ignore[index]
    arms = ordered_arms(por_arm, metrics["strata"])  # type: ignore[arg-type]
    valores = [por_arm[arm]["drive_level"]["exact"] * 100 for arm in arms]
    acaso = 100.0 / metrics["alphabet"]["drive_level"]  # type: ignore[index]
    media = float(np.mean(valores))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([_short(arm) for arm in arms], valores, color=COLOR_LEARNED)
    for indice, valor in enumerate(valores):
        ax.annotate(f"{valor:.1f}", xy=(indice, valor), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.axhline(acaso, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label=f"acaso ({acaso:.1f}%)")
    ax.axhline(media, color="k", linestyle="--", linewidth=1.3,
               label=f"média ({media:.1f}%)")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, max(valores) * 1.3)
    ax.set_title("Por implementação consultada — encoder treinado")
    ax.legend(fontsize=8)

    _save(fig, out_path)


def build_all(results_dir: Path = Path("results/disent"),
              out_dir: Optional[Path] = None) -> List[Path]:
    """Todas as figuras que tiverem dado em disco.

    Baselines em `<results_dir>/b0.json` e `b1.json` (de `gefx disent retrieve`);
    o encoder em `<results_dir>/encoder/`.
    """
    from gefx.disent.train import compare

    results_dir = Path(results_dir)
    encoder_dir = results_dir / "encoder"
    out_dir = Path(out_dir or results_dir / "figuras")
    out_dir.mkdir(parents=True, exist_ok=True)

    def carrega(caminho: Path):
        return json.loads(caminho.read_text(encoding="utf-8"))

    escritos: List[Path] = []

    def grava(nome: str, desenha, *dados) -> None:
        alvo = out_dir / nome
        desenha(*dados, alvo)
        escritos.append(alvo)

    b0, b1 = results_dir / "b0.json", results_dir / "b1.json"
    if b0.exists() and b1.exists():
        grava("baselines_por_arm.png", plot_baselines_by_arm, carrega(b0), carrega(b1))

    treinado = encoder_dir / "rnc" / "metrics.json"
    if not treinado.exists():
        return escritos
    metricas = carrega(treinado)
    chance = drive_chance(metricas)

    grava("escada.png", plot_ladder, compare(encoder_dir, baselines_dir=results_dir),
          grid_step_db(metricas))
    grava("por_arm.png", plot_by_arm, metricas)

    custo = encoder_dir / "loo" / "custo_de_transferencia.csv"
    if custo.exists():
        grava("custo_de_transferencia.png", plot_transfer_cost, pd.read_csv(custo), chance)

    # Uma pasta por semente: `diversidade`, `diversidade_s2`, ...
    curvas = sorted(encoder_dir.glob("diversidade*/resumo.csv"))
    if curvas:
        dados = pd.concat([pd.read_csv(caminho) for caminho in curvas], ignore_index=True)
        arm = str(dados["arm_retirado"].iloc[0])
        por_arm = metricas["per_query_arm"]
        visto = float(por_arm[arm]["drive_level"]["exact"]) if arm in por_arm else None
        alvo = out_dir / "curva_de_diversidade.png"
        plot_diversity_curve(dados, chance, alvo, visto=visto)
        escritos.append(alvo)

    return escritos


# --- leave-one-arm-out ---------------------------------------------------------
def plot_transfer_cost(custo: pd.DataFrame, chance: float, out_path: Path) -> None:
    """Visto x inedito, arm por arm, agrupado pelo estrato do roster."""
    ordem = ordered_arms(custo["arm"], dict(zip(custo["arm"], custo["estrato"])))
    dados = custo.set_index("arm").loc[ordem]
    posicoes = np.arange(len(ordem))
    largura = 0.38

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5),
                                      gridspec_kw={"width_ratios": [2.2, 1]})
    left.bar(posicoes - largura / 2, dados["visto"] * 100, largura,
             color=COLOR_B1, label="implementação vista no treino")
    left.bar(posicoes + largura / 2, dados["inedito"] * 100, largura,
             color=COLOR_CEILING, label="implementação inédita")
    for x, (visto, inedito) in enumerate(zip(dados["visto"], dados["inedito"])):
        left.annotate(f"{(inedito - visto) * 100:+.1f}",
                      xy=(x, max(visto, inedito) * 100), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=9)
    left.axhline(chance * 100, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
                 label=f"acaso ({_pct(chance)})")
    left.set_xticks(posicoes)
    left.set_xticklabels([f"{_short(a)}\n{dados['estrato'][a]}" for a in ordem], fontsize=8)
    left.set_ylabel("acerto exato do nível de drive (%)")
    left.set_ylim(0, 65)
    left.set_title("Custo de nunca ter ouvido a implementação")
    left.legend(fontsize=8, loc="upper right")

    por_estrato = dados.groupby("estrato")["custo_pontos"].mean()
    cores = [COLOR_CEILING if v >= -1 else COLOR_B1 if v > -5 else COLOR_CHANCE
             for v in por_estrato]
    right.bar(por_estrato.index, por_estrato.values, color=cores)
    right.axhline(0, color="k", linewidth=1)
    for x, valor in enumerate(por_estrato.values):
        right.annotate(f"{valor:+.1f}", xy=(x, valor),
                       xytext=(0, 4 if valor >= 0 else -14),
                       textcoords="offset points", ha="center", fontsize=10)
    # Folga para os rotulos das barras extremas.
    right.set_ylim(min(por_estrato.min() * 1.35, -1.0), max(por_estrato.max() * 1.8, 1.0))
    right.set_ylabel("custo médio (pontos de acerto)")
    right.set_title("Por estrato do roster")
    right.set_xticks(range(len(por_estrato)))
    right.set_xticklabels(list(por_estrato.index), fontsize=9)

    _save(fig, out_path)


# --- B2/B3: quantas implementacoes o treino precisa ver -----------------------
def plot_diversity_curve(curva: pd.DataFrame, chance: float, out_path: Path,
                         visto: Optional[float] = None) -> None:
    """Transferencia para o arm retirado contra o numero de arms no treino.

    O catalogo e o mesmo em todos os pontos; a anotacao marca onde entra um estrato
    novo.
    """
    pontos = curva[curva["condicao"] == "transferencia"]
    transferencia = pontos.groupby(["k", "estratos"], as_index=False)["drive_exact"].mean()
    transferencia = transferencia.sort_values("k")
    vistos = curva[curva["condicao"] == "vistos"].groupby("k", as_index=False)[
        "drive_exact"].mean().sort_values("k")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # A dispersao entre sementes e da ordem da excursao da curva.
    sementes = pontos.groupby("k")["drive_exact"].nunique().max()
    if sementes and sementes > 1:
        ax.plot(pontos["k"], pontos["drive_exact"] * 100, "o", color=COLOR_CEILING,
                alpha=0.35, markersize=5,
                label=f"sementes individuais ({int(sementes)} por ponto)")
    ax.plot(transferencia["k"], transferencia["drive_exact"] * 100, "o-",
            color=COLOR_CEILING, linewidth=2, label="implementação inédita (transferência)")
    if not vistos.empty:
        ax.plot(vistos["k"], vistos["drive_exact"] * 100, "s--", color=COLOR_B1,
                linewidth=1.5, alpha=0.8, label="implementações do treino (controle)")
    ax.axhline(chance * 100, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label=f"acaso ({_pct(chance)})")
    if visto is not None:
        ax.axhline(visto * 100, color=COLOR_B0, linestyle="-.", linewidth=1.3,
                   label="o mesmo arm, visto no treino")

    for _, linha in transferencia.iterrows():
        ax.annotate(f"{linha['drive_exact'] * 100:.1f}%",
                    xy=(linha["k"], linha["drive_exact"] * 100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    rotulos = [
        f"{int(linha['k'])}\n{int(linha['estratos'])} estrato"
        + ("s" if int(linha["estratos"]) > 1 else "")
        for _, linha in transferencia.iterrows()
    ]
    ax.set_xticks(transferencia["k"])
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_xlabel("implementações no treino (catálogo fixo em todos os pontos)")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Diversidade de implementação no treino compra transferência?")
    # Embaixo: a metade inferior fica vazia (acaso de 12,5%).
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_path)
