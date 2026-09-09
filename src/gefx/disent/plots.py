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


# --- etapa 5: o estudo comparativo -------------------------------------------
COLOR_LEARNED = "#4c72b0"
COLOR_CONTROL = "#dd8452"
COLOR_RANDOM = "#937860"
COLOR_CEILING = "#55a868"

#: Ordem das tecnicas nas figuras: a escada de acumulo, depois os controles.
#: Ordenar por resultado esconderia justamente o que a escada mostra -- que cada
#: linha acrescenta uma ideia a anterior.
TECHNIQUE_ORDER = (
    "random_encoder", "beta_vae",
    "contrastive", "contrastive_aux", "grl", "full",
)
TECHNIQUE_LABEL = {
    "random_encoder": "encoder\nnão treinado",
    "beta_vae": "β-VAE\n(controle)",
    "contrastive": "contrastivo",
    "contrastive_aux": "+ regressão\nauxiliar",
    "grl": "+ GRL\nimpl. + conteúdo",
    "full": "+ GRL config.\n+ ortogonal.",
}


def ordered_techniques(present: Sequence[str]) -> List[str]:
    present = set(present)
    return [name for name in TECHNIQUE_ORDER if name in present]


def plot_technique_ladder(runs: Dict[str, Dict[str, object]], out_path: Path) -> None:
    """Acerto e erro por tecnica, com os baselines da etapa 4 como linhas.

    As linhas horizontais sao o ponto da figura: sem o B1 e sem o encoder nao
    treinado desenhados no mesmo eixo, uma barra de 44% parece um numero bom em
    vez de um numero **comparado**. O encoder nao treinado e o mais duro dos
    dois, porque separa o que o aprendizado trouxe do que a arquitetura ja dava.
    """
    from gefx.disent.train import BASELINES

    names = ordered_techniques(runs)
    exact = [runs[name]["drive_exact"] * 100 for name in names]
    erro = [runs[name]["mae_db"] for name in names]
    cores = [
        COLOR_CONTROL if name in ("beta_vae",) else
        COLOR_RANDOM if name == "random_encoder" else COLOR_LEARNED
        for name in names
    ]
    rotulos = [TECHNIQUE_LABEL.get(name, name) for name in names]
    posicoes = np.arange(len(names))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))

    left.bar(posicoes, exact, color=cores)
    # Rotulo dentro da barra: as linhas de referencia cruzam o topo delas, e um
    # rotulo por cima cairia em cima de uma linha justamente nos casos
    # interessantes -- os que empatam com um baseline.
    for x, valor in zip(posicoes, exact):
        left.annotate(f"{valor:.1f}", xy=(x, valor), xytext=(0, -14),
                      textcoords="offset points", ha="center", fontsize=9,
                      color="white", fontweight="bold")
    for chave, cor, estilo, texto in (
        ("chance", COLOR_CHANCE, ":", "acaso (12,5%)"),
        ("B0_marginal", COLOR_B0, "--", "B0 sem aprendizado (21,0%)"),
        ("B1_poc1_regressor", COLOR_B1, "--", "B1 regressor do POC I (31,7%)"),
        ("paired_content_ceiling", COLOR_CEILING, "-.", "teto pareado (69,6%)"),
    ):
        left.axhline(BASELINES[chave]["drive_exact"] * 100, color=cor,
                     linestyle=estilo, linewidth=1.4, label=texto)
    left.set_xticks(posicoes)
    left.set_xticklabels(rotulos, fontsize=8)
    left.set_ylabel("acerto exato do nível de drive (%)")
    left.set_ylim(0, max(75, max(exact) * 1.25))
    left.set_title("Recuperação entre implementações")
    left.legend(fontsize=8, loc="upper left")

    right.bar(posicoes, erro, color=cores)
    for x, valor in zip(posicoes, erro):
        right.annotate(f"{valor:.2f}", xy=(x, valor), xytext=(0, -14),
                       textcoords="offset points", ha="center", fontsize=9,
                       color="white", fontweight="bold")
    right.axhline(BASELINES["B1_poc1_regressor"]["mae_db"], color=COLOR_B1,
                  linestyle="--", linewidth=1.4, label="B1 (4,36 dB)")
    right.axhline(BASELINES["paired_content_ceiling"]["mae_db"], color=COLOR_CEILING,
                  linestyle="-.", linewidth=1.4, label="teto pareado (2,67 dB)")
    right.axhline(GRID_STEP_DB, color="k", linestyle=":", linewidth=1.2,
                  label=f"um degrau da grade ({GRID_STEP_DB:.2f} dB)")
    right.set_xticks(posicoes)
    right.set_xticklabels(rotulos, fontsize=8)
    right.set_ylabel("erro médio (dB equivalentes)")
    right.set_ylim(0, max(erro) * 1.3)
    right.set_title("Distância do ajuste certo")
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_training_curves(
    histories: Dict[str, pd.DataFrame],
    checkpoints: Dict[str, List[Dict[str, object]]],
    runs: Dict[str, Dict[str, object]],
    out_path: Path,
) -> None:
    """A esquerda, os termos da perda da tecnica completa; a direita, a
    recuperacao ao longo do treino, por tecnica.

    Os dois paineis respondem perguntas diferentes e nao intercambiaveis: perda
    caindo nao e recuperacao subindo, e e justamente onde os dois se descolam que
    a tecnica esta otimizando a coisa errada.
    """
    from gefx.disent.train import BASELINES

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    nome_completa = "full" if "full" in histories else next(iter(histories))
    completa = histories[nome_completa]
    termos = [c for c in completa.columns if c not in ("step", "lambda", "total")]
    for termo in sorted(termos):
        suave = completa[termo].rolling(50, min_periods=1).mean()
        left.plot(completa["step"], suave, linewidth=1.6, label=termo)
    # A rampa so e desenhada quando ha adversario: numa tecnica sem reversao ela
    # existe no historico mas nao multiplica gradiente nenhum, e mostra-la
    # sugeriria uma agenda que nao esta em vigor.
    eixo_lambda = None
    if any(termo.startswith("adversary") for termo in termos):
        eixo_lambda = left.twinx()
        eixo_lambda.plot(completa["step"], completa["lambda"], color="k",
                         linestyle=":", linewidth=1.2, label="λ")
        eixo_lambda.set_ylabel("λ da reversão de gradiente (linha pontilhada)",
                               fontsize=9)
        eixo_lambda.set_ylim(0, 1.05)
    left.set_xlabel("passo")
    left.set_ylabel("perda (média móvel de 50 passos)")
    left.set_title(f"Termos da perda — técnica «{nome_completa}»")
    # A legenda vai no eixo de cima (o do λ, quando existe): desenhada no de
    # baixo, a curva pontilhada passa por dentro dela.
    handles, labels = left.get_legend_handles_labels()
    (eixo_lambda if eixo_lambda is not None else left).legend(
        handles, labels, fontsize=8, loc="upper right", framealpha=0.95
    )

    for nome in ordered_techniques(checkpoints):
        pontos = list(checkpoints[nome])
        if not pontos:
            continue
        # O ultimo ponto e a avaliacao final, que nao esta em `checkpoints`. Sem
        # ele a curva termina antes do numero que a figura da escada reporta.
        passos = [int(p["step"]) for p in pontos] + [len(histories[nome])]
        acertos = [float(p["drive_exact"]) * 100 for p in pontos]
        acertos.append(float(runs[nome]["drive_exact"]) * 100)
        right.plot(passos, acertos, marker="o", linewidth=1.8,
                   label=TECHNIQUE_LABEL.get(nome, nome).replace("\n", " "))
    right.axhline(BASELINES["B1_poc1_regressor"]["drive_exact"] * 100,
                  color=COLOR_B1, linestyle="--", linewidth=1.4, label="B1")
    right.axhline(BASELINES["paired_content_ceiling"]["drive_exact"] * 100,
                  color=COLOR_CEILING, linestyle="-.", linewidth=1.4,
                  label="teto pareado")
    right.set_xlabel("passo")
    right.set_ylabel("acerto exato do nível de drive (%)")
    right.set_title("Recuperação ao longo do treino")
    # Fora dos eixos: as seis curvas ocupam a faixa inteira do painel e qualquer
    # canto que a legenda escolha cobre pontos de avaliacao.
    right.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_hub_by_technique(shares: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """De qual implementacao vem a resposta, por tecnica, contra o B0.

    O adversario de implementacao existe por causa desta figura: no B0 um unico
    arm respondia 64,6% das consultas, e uma acuracia agregada nao mostra isso.
    """
    from gefx.disent.train import BASELINES

    names = ordered_techniques(shares)
    arms = ordered_arms({arm for share in shares.values() for arm in share})
    uniforme = 100.0 / len(arms)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    largura = 0.8 / len(arms)
    posicoes = np.arange(len(names))
    for indice, arm in enumerate(arms):
        valores = [shares[name].get(arm, 0.0) * 100 for name in names]
        ax.bar(posicoes + indice * largura - 0.4 + largura / 2, valores,
               width=largura, label=_short(arm))
    ax.axhline(uniforme, color="k", linestyle="--", linewidth=1.2)
    ax.annotate(f"uniforme ({uniforme:.1f}%)", xy=(-0.45, uniforme),
                xytext=(0, 4), textcoords="offset points", ha="left", fontsize=9)
    ax.axhline(BASELINES["B0_marginal"]["top_arm_share"] * 100, color=COLOR_CHANCE,
               linestyle=":", linewidth=1.4)
    ax.annotate("pior caso do B0 (64,6%)",
                xy=(0, BASELINES["B0_marginal"]["top_arm_share"] * 100),
                xytext=(4, 4), textcoords="offset points", fontsize=9,
                color=COLOR_CHANCE)
    ax.set_xticks(posicoes)
    ax.set_xticklabels([TECHNIQUE_LABEL.get(n, n) for n in names], fontsize=8)
    ax.set_ylabel("fatia das consultas respondidas (%)")
    ax.set_title("Concentração das respostas por implementação")
    ax.legend(fontsize=8, ncol=4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_by_arm_with_and_without_bigmuff(
    metrics: Dict[str, object], technique: str, out_path: Path
) -> None:
    """Acerto por implementacao consultada, com o agregado nos dois recortes.

    O agregado sem o `byod-bigmuff` acompanha todo agregado deste trabalho por
    decisao registrada: o arm reprova a porteira de contraste no audio
    renderizado e carrega ruido de rotulo conhecido. Mostrar so o agregado cheio
    esconderia quanto do resultado e ele.
    """
    por_arm = metrics["per_query_arm"]  # type: ignore[index]
    arms = ordered_arms(por_arm)
    valores = [por_arm[arm]["drive_level"]["exact"] * 100 for arm in arms]
    acaso = 100.0 / metrics["alphabet"]["drive_level"]  # type: ignore[index]

    cheio = float(np.mean(valores))
    sem = float(np.mean([v for a, v in zip(arms, valores) if a != "byod-bigmuff"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    cores = [COLOR_CHANCE if arm == "byod-bigmuff" else COLOR_LEARNED for arm in arms]
    ax.bar([_short(arm) for arm in arms], valores, color=cores)
    for indice, valor in enumerate(valores):
        ax.annotate(f"{valor:.1f}", xy=(indice, valor), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.axhline(acaso, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label=f"acaso ({acaso:.1f}%)")
    ax.axhline(cheio, color="k", linestyle="--", linewidth=1.3,
               label=f"média dos 7 arms ({cheio:.1f}%)")
    ax.axhline(sem, color=COLOR_CEILING, linestyle="-.", linewidth=1.3,
               label=f"média sem o byod-bigmuff ({sem:.1f}%)")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, max(valores) * 1.3)
    ax.set_title(f"Por implementação consultada — técnica «{technique}»")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def build_etapa5(results_dir: Path, out_dir: Optional[Path] = None) -> List[Path]:
    """Figuras do estudo comparativo, a partir dos `run.json` ja gravados."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir or results_dir / "figuras")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[str, Dict[str, object]] = {}
    checkpoints: Dict[str, List[Dict[str, object]]] = {}
    histories: Dict[str, pd.DataFrame] = {}
    shares: Dict[str, Dict[str, float]] = {}
    metrics: Dict[str, Dict[str, object]] = {}

    for pasta in sorted(results_dir.iterdir()):
        manifesto = pasta / "run.json"
        if not pasta.is_dir() or not manifesto.exists():
            continue
        dados = json.loads(manifesto.read_text(encoding="utf-8"))
        runs[pasta.name] = dados["decision"]["measured"]
        checkpoints[pasta.name] = dados.get("checkpoints", [])
        metrics[pasta.name] = json.loads(
            (pasta / "metrics.json").read_text(encoding="utf-8")
        )
        shares[pasta.name] = metrics[pasta.name]["hubness"]["by_retrieved_arm"]
        historico = json.loads((pasta / "history.json").read_text(encoding="utf-8"))
        if historico:
            histories[pasta.name] = pd.DataFrame(historico)

    if not runs:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")

    escritos: List[Path] = []

    alvo = out_dir / "escada_de_tecnicas.png"
    plot_technique_ladder(runs, alvo)
    escritos.append(alvo)

    if histories:
        alvo = out_dir / "curvas_de_treino.png"
        plot_training_curves(histories, checkpoints, runs, alvo)
        escritos.append(alvo)

    alvo = out_dir / "sorvedouro_por_tecnica.png"
    plot_hub_by_technique(shares, alvo)
    escritos.append(alvo)

    custo = results_dir / "loo" / "custo_de_transferencia.csv"
    if custo.exists():
        alvo = out_dir / "custo_de_transferencia.png"
        plot_transfer_cost(pd.read_csv(custo), alvo)
        escritos.append(alvo)

    curva = results_dir / "diversidade" / "resumo.csv"
    if curva.exists():
        alvo = out_dir / "curva_de_diversidade.png"
        referencia = None
        melhores = results_dir / "contrastive_aux" / "metrics.json"
        if melhores.exists():
            por_arm = json.loads(melhores.read_text(encoding="utf-8"))["per_query_arm"]
            dados = pd.read_csv(curva)
            arm = str(dados["arm_retirado"].iloc[0])
            if arm in por_arm:
                referencia = float(por_arm[arm]["drive_level"]["exact"])
        plot_diversity_curve(pd.read_csv(curva), alvo, visto=referencia)
        escritos.append(alvo)

    estrutura = results_dir / "estrutura.csv"
    if estrutura.exists():
        alvo = out_dir / "estrutura_por_bloco.png"
        plot_structure_blocks(pd.read_csv(estrutura), alvo)
        escritos.append(alvo)

    melhor = max(runs, key=lambda nome: runs[nome]["drive_exact"])
    alvo = out_dir / "por_arm_melhor_tecnica.png"
    plot_by_arm_with_and_without_bigmuff(metrics[melhor], melhor, alvo)
    escritos.append(alvo)

    return escritos


# --- etapa 7: transferencia para implementacao inedita ------------------------
def plot_transfer_cost(custo: pd.DataFrame, out_path: Path) -> None:
    """Visto x inedito, arm por arm, agrupado pelo estrato do roster.

    E a figura que sustenta ou derruba a tese do trabalho. Os numeros da etapa 5
    sao medidos com todas as implementacoes no treino; so esta comparacao diz o
    que acontece com uma implementacao que a rede nunca ouviu. Agrupar por
    estrato e o ponto: o custo nao e uniforme, ele segue exatamente o eixo que o
    desenho do roster previu.
    """
    ordem = ordered_arms(custo["arm"])
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
    left.axhline(100 / 8, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
                 label="acaso (12,5%)")
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
    # Folga nas duas pontas: sem ela o rotulo da barra mais negativa cai fora
    # do eixo, que e justamente a barra que carrega o resultado.
    right.set_ylim(min(por_estrato.min() * 1.35, -1.0), max(por_estrato.max() * 1.8, 1.0))
    right.set_ylabel("custo médio (pontos de acerto)")
    right.set_title("Por estrato do roster")
    right.set_xticks(range(len(por_estrato)))
    right.set_xticklabels(
        [f"{e}\n{d}" for e, d in zip(por_estrato.index,
                                     ("mesma forma,\nmesma unidade",
                                      "mesma unidade,\nforma diferente",
                                      "unidade e topologia\ndiferentes"))],
        fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- B2/B3: quantas implementacoes o treino precisa ver -----------------------
def plot_diversity_curve(curva: pd.DataFrame, out_path: Path,
                         visto: Optional[float] = None) -> None:
    """Transferencia para o arm retirado contra o numero de arms no treino.

    O catalogo e o mesmo em todos os pontos, entao a unica coisa que muda ao
    longo do eixo x e quantas implementacoes o encoder ouviu. A anotacao do
    estrato importa mais que o numero de arms: se a curva subir nos degraus em
    que um estrato **novo** entra e ficar plana nos outros, o que compra
    transferencia e variedade, nao quantidade.
    """
    transferencia = curva[curva["condicao"] == "transferencia"].sort_values("k")
    vistos = curva[curva["condicao"] == "vistos"].sort_values("k")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(transferencia["k"], transferencia["drive_exact"] * 100, "o-",
            color=COLOR_CEILING, linewidth=2, label="implementação inédita (transferência)")
    if not vistos.empty:
        ax.plot(vistos["k"], vistos["drive_exact"] * 100, "s--", color=COLOR_B1,
                linewidth=1.5, alpha=0.8, label="implementações do treino (controle)")
    ax.axhline(100 / 8, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label="acaso (12,5%)")
    if visto is not None:
        ax.axhline(visto * 100, color=COLOR_B0, linestyle="-.", linewidth=1.3,
                   label="o mesmo arm, visto no treino (etapa 5)")

    for _, linha in transferencia.iterrows():
        ax.annotate(f"{linha['drive_exact'] * 100:.1f}%",
                    xy=(linha["k"], linha["drive_exact"] * 100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    rotulos = [
        f"{int(linha['k'])}\n{linha['estratos']} estrato"
        + ("s" if int(linha["estratos"]) > 1 else "")
        for _, linha in transferencia.iterrows()
    ]
    ax.set_xticks(transferencia["k"])
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_xlabel("implementações no treino (catálogo fixo em todos os pontos)")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Diversidade de implementação no treino compra transferência?")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- etapa 6: onde cada fator esta escrito ------------------------------------
def plot_structure_blocks(estrutura: pd.DataFrame, out_path: Path) -> None:
    """Fracao da importancia de cada fator que cai em `z_e`, por tecnica.

    E a afirmacao do desemaranhamento na forma em que ela foi feita: em blocos.
    A linha de 50% e a fronteira -- acima dela o fator esta escrito em `z_e`,
    abaixo em `z_c`. As duas primeiras barras de cada grupo deveriam ficar em
    cima e as duas ultimas embaixo.
    """
    tecnicas = ordered_techniques(estrutura["technique"].unique())
    fatores = ["drive_level", "tone_level", "arm", "content_id"]
    nomes = {"drive_level": "drive\n(deve ficar)", "tone_level": "tom\n(deve ficar)",
             "arm": "implementação\n(deve sair)", "content_id": "conteúdo\n(deve sair)"}
    tabela = estrutura.pivot_table(index="technique", columns="factor", values="massa_z_e")

    posicoes = np.arange(len(fatores))
    largura = 0.8 / max(len(tecnicas), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for indice, tecnica in enumerate(tecnicas):
        deslocamento = (indice - (len(tecnicas) - 1) / 2) * largura
        cor = (COLOR_RANDOM if tecnica == "random_encoder"
               else COLOR_CONTROL if tecnica == "beta_vae" else COLOR_LEARNED)
        ax.bar(posicoes + deslocamento,
               [tabela.loc[tecnica, fator] * 100 for fator in fatores], largura,
               color=cor, alpha=0.55 + 0.45 * indice / max(len(tecnicas) - 1, 1),
               label=TECHNIQUE_LABEL.get(tecnica, tecnica).replace("\n", " "))
    ax.axhline(50, color="k", linestyle="--", linewidth=1.2)
    ax.annotate("fronteira: metade da importância em cada bloco", xy=(len(fatores) - 0.5, 51),
                ha="right", fontsize=8)
    ax.set_xticks(posicoes)
    ax.set_xticklabels([nomes[f] for f in fatores], fontsize=9)
    ax.set_ylabel("importância do fator que cai em $z_e$ (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Onde cada fator está escrito: $z_e$ (efeito) contra $z_c$ (conteúdo)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
