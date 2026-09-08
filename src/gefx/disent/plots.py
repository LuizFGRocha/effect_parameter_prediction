"""Figuras dos baselines do POC II.

Le os CSV/JSON que `disent/retrieval.py` grava e nao recalcula nada: quem produz
numero e o modulo de recuperacao, quem desenha e este. Separar os dois e o que
permite refazer uma figura sem repetir dez minutos de comparacao.

A ordem dos arms nas figuras e sempre a do roster (`arms.arm_keys()`), com a
referencia primeiro e os estratos agrupados. Ordenar por resultado deixaria cada
figura com uma ordem diferente e impediria a leitura cruzada.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: isto roda em terminal e em CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gefx.disent.arms import DRIVE_LEVELS, TONE_LEVELS, arm_keys

# Passo medio da grade, em dB equivalentes. E a unidade de leitura de todo erro
# reportado: errar menos que isto e acertar o nivel.
GRID_STEP_DB = 4.15
# Erro do regressor do POC I dentro da propria implementacao, para referencia.
POC1_MAE_DB = 1.28

COLOR_B0 = "#8c8c8c"
COLOR_B1 = "#1f77b4"
COLOR_CHANCE = "#c44e52"
COLOR_EAR = "#2ca02c"


def ordered_arms(present: Sequence[str]) -> List[str]:
    """Arms do roster que aparecem no dado, na ordem do roster."""
    present = set(present)
    return [key for key in arm_keys() if key in present]


def _short(key: str) -> str:
    """Rotulo de eixo. Mantem o prefixo de backend de proposito.

    Encurtar `lsp-tanh` para `tanh` o tornaria indistinguivel do
    `pedalboard-tanh` justamente na figura em que os dois quase coincidem -- e
    essa coincidencia e o resultado.
    """
    return key.replace("pedalboard-", "pb-")


# --- 1. B0 x B1 por arm -------------------------------------------------------
def plot_baselines_by_arm(
    b0_metrics: Dict[str, object],
    b1_metrics: Dict[str, object],
    out_path: Path,
) -> None:
    """Acerto e erro dos dois baselines, arm a arm, com o acaso e o passo da grade.

    Duas escalas no mesmo assunto: acerto exato responde "acha o nivel certo?" e
    o erro em dB responde "quando erra, erra por quanto?". A segunda importa mais
    aqui, porque a grade e discreta e o acerto exato sozinho esconde a diferenca
    entre errar um degrau e errar quatro.
    """
    arms = ordered_arms(list(b0_metrics["per_query_arm"]))
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
    # Margem a esquerda para as anotacoes das linhas de referencia caberem
    # fora das barras, e nao por cima delas.
    top.set_xlim(-1.35, len(arms) - 0.4)
    top.set_ylim(0, max(max(b0), max(b1)) * 1.18)
    top.set_ylabel("acerto exato do nível de drive (%)")
    top.set_title(f"Baselines por implementação — {DRIVE_LEVELS} níveis de drive, "
                  f"{TONE_LEVELS} de tom")
    top.legend(loc="upper left")

    bottom.bar(positions - width / 2, b0_db, width, color=COLOR_B0)
    bottom.bar(positions + width / 2, b1_db, width, color=COLOR_B1)
    bottom.axhline(GRID_STEP_DB, color=COLOR_CHANCE, linestyle="--", linewidth=1.2)
    bottom.annotate(f"um degrau\nda grade\n({GRID_STEP_DB} dB)",
                    xy=(-1.3, GRID_STEP_DB), xytext=(2, 5),
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

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- 2. matriz par a par ------------------------------------------------------
def plot_pairwise(pairwise: pd.DataFrame, out_path: Path) -> None:
    """Acerto por par (consulta, catalogo), com catalogo de um arm so.

    A diagonal e o controle sem troca de implementacao. O que a figura tem de
    mostrar e que ela **nao** se destaca: e assim que se ve que a travessia entre
    implementacoes nao e o gargalo.
    """
    arms = ordered_arms(pairwise["query_arm"].unique())
    grid = (
        pairwise.pivot(index="query_arm", columns="catalog_arm", values="drive_exact")
        .loc[arms, arms]
        .to_numpy()
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(grid, cmap="viridis")
    for i in range(len(arms)):
        for j in range(len(arms)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.75 else "black",
                    fontsize=9)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="white", linewidth=2))
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_short(a) for a in arms], rotation=30, ha="right")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([_short(a) for a in arms])
    ax.set_xlabel("implementação do catálogo")
    ax.set_ylabel("implementação da consulta")
    diagonal = float(np.mean(np.diag(grid)))
    fora = float(grid[~np.eye(len(arms), dtype=bool)].mean())
    ax.set_title(
        "B0 par a par: acerto exato de drive (%)\n"
        f"diagonal {diagonal:.1f}%   fora dela {fora:.1f}%   acaso 12,5%"
    )
    fig.colorbar(image, ax=ax, shrink=0.8, label="acerto (%)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- 3. sorvedouro ------------------------------------------------------------
def plot_hubness(
    predictions: pd.DataFrame, catalog_size: int, out_path: Path
) -> None:
    """Concentracao das respostas: curva de ocupacao e de onde elas vem.

    A curva sozinha ja denuncia o *hub*, mas e a barra da direita que da o nome
    ao culpado -- e o que liga o defeito a uma implementacao concreta do roster.
    """
    counts = predictions["retrieved_file"].value_counts().to_numpy()
    occupancy = np.zeros(catalog_size)
    occupancy[: len(counts)] = counts
    ordered = np.sort(occupancy)[::-1]
    share = np.cumsum(ordered) / ordered.sum()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    left.plot(np.arange(1, catalog_size + 1) / catalog_size * 100, share * 100,
              color=COLOR_B1, linewidth=2)
    left.plot([0, 100], [0, 100], "k--", linewidth=1, label="catálogo uniforme")
    marca = share[max(0, catalog_size // 100 - 1)] * 100
    left.axvline(1.0, color=COLOR_CHANCE, linestyle=":", linewidth=1.2)
    left.annotate(f"1% do catálogo\natrai {marca:.0f}% das consultas",
                  xy=(1.0, marca), xytext=(12, -18), textcoords="offset points",
                  color=COLOR_CHANCE, fontsize=9,
                  arrowprops={"arrowstyle": "->", "color": COLOR_CHANCE})
    left.set_xlabel("itens do catálogo, do mais atraente ao menos (%)")
    left.set_ylabel("consultas acumuladas (%)")
    left.set_title(f"Concentração das respostas (máximo: {int(ordered[0])} consultas "
                   f"num item; esperado: {len(predictions)/catalog_size:.1f})")
    left.legend(loc="lower right")

    origem = predictions["retrieved_arm"].value_counts(normalize=True) * 100
    arms = ordered_arms(origem.index)
    valores = [origem[a] for a in arms]
    uniforme = 100.0 / len(arms)
    cores = [COLOR_CHANCE if v > 2 * uniforme else COLOR_B1 for v in valores]
    right.barh([_short(a) for a in arms], valores, color=cores)
    right.axvline(uniforme, color="k", linestyle="--", linewidth=1)
    right.annotate(f"uniforme\n({uniforme:.1f}%)", xy=(uniforme, 0),
                   xytext=(5, 0), textcoords="offset points", fontsize=9,
                   va="center")
    right.invert_yaxis()
    right.set_xlabel("fatia das consultas respondidas (%)")
    right.set_title("De qual implementação veio a resposta")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- 4. paridade do B1 --------------------------------------------------------
def plot_b1_parity(predictions: pd.DataFrame, out_path: Path) -> None:
    """Predito x verdadeiro do regressor do POC I, um painel por implementacao.

    Caixa por nivel em vez de dispersao porque o eixo verdadeiro e discreto: com
    700 pontos empilhados em 8 abscissas, a nuvem vira uma coluna solida e esconde
    exatamente o que interessa, que e o desvio da diagonal.
    """
    arms = ordered_arms(predictions["query_arm"].unique())
    ladder = np.array(
        sorted(predictions.groupby("true_drive_level")["true_drive_db"].first())
    )
    ncols = 4
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for index, arm in enumerate(arms):
        ax = axes[index // ncols][index % ncols]
        part = predictions[predictions["query_arm"] == arm]
        grupos = [part[part["true_drive_level"] == lvl]["pred_drive_db"].to_numpy()
                  for lvl in range(len(ladder))]
        ax.boxplot(grupos, positions=ladder, widths=2.4, showfliers=False,
                   medianprops={"color": COLOR_B1, "linewidth": 1.8},
                   manage_ticks=False)
        limites = [ladder[0] - 3, ladder[-1] + 3]
        ax.plot(limites, limites, "k--", linewidth=1)
        vies = float((part["pred_drive_db"] - part["true_drive_db"]).mean())
        ax.set_title(f"{_short(arm)}   viés {vies:+.2f} dB", fontsize=10)
        ax.set_xlim(*limites)
        ax.set_ylim(*limites)
        ax.set_xticks(np.round(ladder).astype(int))
        ax.tick_params(labelsize=8)

    for extra in range(len(arms), nrows * ncols):
        axes[extra // ncols][extra % ncols].axis("off")
        # `sharex` esconde os rotulos de quem tem painel abaixo. Nas colunas que
        # terminam antes da ultima linha, o painel de cima fica sem eixo nenhum.
        acima = axes[extra // ncols - 1][extra % ncols]
        acima.tick_params(labelbottom=True, labelsize=8)
    fig.supxlabel("drive verdadeiro (dB equivalentes)")
    fig.supylabel("drive predito pelo regressor do POC I (dB)")
    fig.suptitle("B1: a diagonal tracejada é o acerto perfeito", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- 5. convergencia dos vieses ----------------------------------------------
def plot_bias_convergence(
    predictions: pd.DataFrame,
    perceptual_bias: Dict[str, Dict[str, object]],
    out_path: Path,
    reference_arm: str = "pedalboard-tanh",
    step_db: float = GRID_STEP_DB,
) -> None:
    """Vies do B1 contra o vies medido no ouvido, na mesma unidade e no mesmo sinal.

    O vies do B1 e tomado **em relacao a referencia**, para descontar o efeito de
    teto do regressor -- ele satura antes do topo da escada e por isso subestima
    em todos os arms por igual. O que sobra e o que e proprio de cada
    implementacao, que e o que se compara com o ouvido.
    """
    arms = ordered_arms(predictions["query_arm"].unique())
    erro = predictions["pred_drive_db"] - predictions["true_drive_db"]
    por_arm = erro.groupby(predictions["query_arm"])
    base = float(por_arm.mean()[reference_arm])
    # Sinal invertido: o teste cego usa "negativo = soa mais distorcido", e mais
    # distorcido corresponde a um dB predito MAIOR.
    excedente = {arm: -(float(por_arm.mean()[arm]) - base) / step_db for arm in arms}
    erro_padrao = {
        arm: 1.96 * float(por_arm.std()[arm]) / np.sqrt(por_arm.size()[arm]) / step_db
        for arm in arms
    }

    positions = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions, [excedente[a] for a in arms],
           yerr=[erro_padrao[a] for a in arms], capsize=4, width=0.55,
           color=COLOR_B1, label="B1 — regressor do POC I (IC95)")

    marcados = [a for a in arms if a in perceptual_bias]
    ax.errorbar(
        [positions[arms.index(a)] for a in marcados],
        [perceptual_bias[a]["bias_levels"] for a in marcados],
        yerr=np.abs(
            np.array([[perceptual_bias[a]["ci95_levels"][0] for a in marcados],
                      [perceptual_bias[a]["ci95_levels"][1] for a in marcados]])
            - np.array([perceptual_bias[a]["bias_levels"] for a in marcados])
        ),
        fmt="D", color=COLOR_EAR, markersize=9, capsize=6, linewidth=2,
        label="teste cego de escuta (IC95)", zorder=5,
    )
    ax.axhline(0, color="k", linewidth=1)
    # Sem esta marca o arm de referencia parece um resultado: ele e zero porque
    # e dele que os outros sao subtraidos.
    if reference_arm in arms:
        ax.annotate("referência:\nzero por construção",
                    xy=(positions[arms.index(reference_arm)], 0),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color="#555555")
    ax.set_xticks(positions)
    ax.set_xticklabels([_short(a) for a in arms], rotation=20, ha="right")
    ax.set_ylabel("viés (níveis da grade)")
    ax.set_title("Viés residual: o regressor do POC I contra o ouvido\n"
                 "negativo = soa mais distorcido que o nível nominal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- 6. fidelidade da reducao -------------------------------------------------
def plot_fidelity(sweep: pd.DataFrame, chosen: int, out_path: Path) -> None:
    """Correlacao com a distancia integral do oraculo por agrupamento de tempo."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sweep["bands"], sweep["spearman"], "o-", color=COLOR_B1,
            label="Spearman (a ordem, que é o que a busca consome)")
    ax.plot(sweep["bands"], sweep["pearson"], "s--", color=COLOR_B0,
            label="Pearson (o valor)")
    escolhido = sweep[sweep["bands"] == chosen].iloc[0]
    ax.axvline(chosen, color=COLOR_CHANCE, linestyle=":", linewidth=1.5)
    ax.annotate(f"escolhido: {chosen} faixas\n{int(escolhido['dims'])} dimensões, "
                f"ρ = {escolhido['spearman']:.3f}",
                xy=(chosen, escolhido["spearman"]), xytext=(12, -42),
                textcoords="offset points", color=COLOR_CHANCE, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": COLOR_CHANCE})
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep["bands"])
    ax.set_xticklabels(sweep["bands"])
    ax.set_xlabel("faixas de tempo mantidas no descritor")
    ax.set_ylabel("correlação com a distância integral")
    ax.set_title("Preço da redução do descritor "
                 f"({int(escolhido['n_pairs'])} pares do split de consulta)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- driver -------------------------------------------------------------------
def build_all(results_dir: Path, out_dir: Optional[Path] = None) -> List[Path]:
    """Todas as figuras a partir do que a etapa 4 gravou."""
    from gefx.disent.arms import PERCEPTUAL_BIAS
    from gefx.disent.retrieval import MEL_TIME_POOL

    results_dir = Path(results_dir)
    out_dir = Path(out_dir or results_dir / "figuras")
    out_dir.mkdir(parents=True, exist_ok=True)

    def carrega(nome: str):
        return json.loads((results_dir / nome).read_text(encoding="utf-8"))

    b0_cross = pd.read_csv(results_dir / "b0_cross.csv")
    b1 = pd.read_csv(results_dir / "b1.csv")
    escritos: List[Path] = []

    alvo = out_dir / "baselines_por_arm.png"
    plot_baselines_by_arm(carrega("b0_cross.json"), carrega("b1.json"), alvo)
    escritos.append(alvo)

    alvo = out_dir / "b0_par_a_par.png"
    plot_pairwise(pd.read_csv(results_dir / "b0_pairwise.csv"), alvo)
    escritos.append(alvo)

    alvo = out_dir / "sorvedouro.png"
    metrics_b0 = carrega("b0_cross.json")
    if "catalog_size" not in metrics_b0:
        raise KeyError(
            "b0_cross.json nao tem `catalog_size`; refaca `gefx disent retrieve` "
            "-- a ocupacao esperada do diagnostico de sorvedouro sai dele"
        )
    plot_hubness(b0_cross, int(metrics_b0["catalog_size"]), alvo)
    escritos.append(alvo)

    alvo = out_dir / "b1_paridade.png"
    plot_b1_parity(b1, alvo)
    escritos.append(alvo)

    alvo = out_dir / "vies_convergencia.png"
    plot_bias_convergence(b1, PERCEPTUAL_BIAS, alvo)
    escritos.append(alvo)

    sweep_path = results_dir / "reduction_sweep.csv"
    if sweep_path.exists():
        alvo = out_dir / "fidelidade_reducao.png"
        plot_fidelity(pd.read_csv(sweep_path), MEL_TIME_POOL, alvo)
        escritos.append(alvo)

    return escritos
