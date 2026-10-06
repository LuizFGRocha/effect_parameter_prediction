"""Graficos: as primitivas compartilhadas, as figuras do POC I e as do POC II.

As do POC II sao lidas dos CSV/JSON gravados: nada aqui recalcula resultado. Os
arms aparecem sempre por estrato e, dentro dele, por nome; estrato, acaso e passo
da grade saem das metricas gravadas, nao do roster.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # sem display: isto roda em terminal e em CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCATTER_COLOR = "#1f77b4"
ERROR_BAR_STYLE = {"elinewidth": 1, "ecolor": "#444444"}


def safe_filename(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def grouped_bar_axes(
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    left_label: str,
    right_label: str,
    left_errors: Optional[Sequence[float]] = None,
    right_errors: Optional[Sequence[float]] = None,
    width: float = 0.35,
    figsize=(10, 5),
):
    """Duas series de barras lado a lado, com barras de erro opcionais: `(fig, ax)`."""
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        positions - width / 2,
        left_values,
        width,
        label=left_label,
        yerr=left_errors,
        capsize=4 if left_errors is not None else 0,
        error_kw=ERROR_BAR_STYLE,
    )
    ax.bar(
        positions + width / 2,
        right_values,
        width,
        label=right_label,
        yerr=right_errors,
        capsize=4 if right_errors is not None else 0,
        error_kw=ERROR_BAR_STYLE,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    return fig, ax


def grouped_bars(
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    left_label: str,
    right_label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    left_errors: Optional[Sequence[float]] = None,
    right_errors: Optional[Sequence[float]] = None,
    width: float = 0.35,
) -> None:
    """`grouped_bar_axes` com titulo, eixos e legenda, gravado em `out_path`."""
    fig, ax = grouped_bar_axes(labels, left_values, right_values, left_label, right_label,
                               left_errors, right_errors, width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parity_axis(axis, real: np.ndarray, estimated: np.ndarray, title: str) -> None:
    """Dispersao estimado-vs-real com a diagonal ideal e eixos quadrados iguais."""
    min_value = float(min(real.min(), estimated.min()))
    max_value = float(max(real.max(), estimated.max()))
    padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
    axis_min, axis_max = min_value - padding, max_value + padding

    axis.scatter(real, estimated, s=18, alpha=0.65, color=SCATTER_COLOR)
    axis.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1, label="Ideal")
    axis.set_title(title)
    axis.set_xlabel("Real value")
    axis.set_ylabel("Estimated value")
    axis.set_xlim(axis_min, axis_max)
    axis.set_ylim(axis_min, axis_max)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")


def parity_grid(panels: Sequence[tuple], out_path: Path, suptitle: str, ncols_cap: int = 2) -> None:
    """Grade de paineis de paridade. Cada painel e (titulo, real, estimated)."""
    if not panels:
        raise ValueError("parity_grid precisa de pelo menos um painel.")

    ncols = min(ncols_cap, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for axis, (title, real, estimated) in zip(axes.flat, panels):
        parity_axis(axis, real, estimated, title)
    for axis in axes.flat[len(panels) :]:
        fig.delaxes(axis)

    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_chain_baseline(chain_frame: pd.DataFrame, out_path: Path) -> None:
    grouped = chain_frame.sort_values("chain_key")
    grouped_bars(
        labels=grouped["chain_key"].tolist(),
        left_values=grouped["mae"],
        right_values=grouped["mse"],
        left_errors=grouped["mae_sem"] if "mae_sem" in grouped else None,
        right_errors=grouped["mse_sem"] if "mse_sem" in grouped else None,
        left_label="Mean MAE",
        right_label="Mean MSE",
        title="Fixed-order chain comparison",
        xlabel="Chain",
        ylabel="Error",
        out_path=out_path,
    )


def plot_estimated_vs_real(plot_frame: pd.DataFrame, out_dir: Path) -> None:
    """Um png por parametro, com um painel de paridade por cadeia."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for param in sorted(plot_frame["parameter"].unique()):
        param_frame = plot_frame[plot_frame["parameter"] == param]
        panels = [
            (
                chain_key_value,
                param_frame.loc[param_frame["chain_key"] == chain_key_value, "real"].to_numpy(np.float64),
                param_frame.loc[param_frame["chain_key"] == chain_key_value, "estimated"].to_numpy(np.float64),
            )
            for chain_key_value in sorted(param_frame["chain_key"].unique())
        ]
        parity_grid(panels, out_dir / f"{safe_filename(param)}.png", f"Estimated vs real | {param}")


def plot_best_worst(plot_frame: pd.DataFrame, out_path: Path, top_n: int = 3) -> None:
    """Os `top_n` pares (cadeia, parametro) de menor e de maior MAE."""
    per_pair = (
        plot_frame.assign(abs_error=(plot_frame["estimated"] - plot_frame["real"]).abs())
        .groupby(["chain_key", "parameter"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    selected = pd.concat(
        [per_pair.nsmallest(top_n, "mae"), per_pair.nlargest(top_n, "mae")], ignore_index=True
    )
    if selected.empty:
        raise ValueError("Nenhuma predicao para o grafico de melhores/piores.")

    panels = []
    for row in selected.itertuples(index=False):
        pair = plot_frame[
            (plot_frame["chain_key"] == row.chain_key) & (plot_frame["parameter"] == row.parameter)
        ]
        panels.append(
            (
                f"{row.chain_key} | {row.parameter}\nMAE={row.mae:.4f}",
                pair["real"].to_numpy(np.float64),
                pair["estimated"].to_numpy(np.float64),
            )
        )

    parity_grid(panels, out_path, "Best 3 and worst 3 parameter estimates", ncols_cap=3)


# --- POC II -------------------------------------------------------------------
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


# --- a escada ------------------------------------------------------------------
COLOR_LEARNED = "#4c72b0"
COLOR_RANDOM = "#937860"
COLOR_CEILING = "#55a868"

TECHNIQUE_LABEL = {
    "random_encoder": "encoder\nnão treinado",
    "bn_only": "não treinado\n+ BatchNorm calibrada",
    "supcon": "encoder treinado\n(SupCon)",
    "regressao": "mesmo tronco,\nsaída escalar (MSE)",
}


def plot_ladder(tabela: pd.DataFrame, passo_db: float, out_path: Path) -> None:
    """Acerto e erro por tecnica, media das sementes com cada semente como ponto,
    e os baselines como linhas."""
    from gefx.training.architecture import TECHNIQUES

    execucoes = tabela[tabela["technique"] != "baseline"]
    names = [name for name in TECHNIQUES if name in set(execucoes["technique"])]
    posicoes = np.arange(len(names))
    cores = [COLOR_LEARNED if name == "supcon" else COLOR_RANDOM for name in names]
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
        # Rotulo no pe da barra: no topo, cairia nas sementes e nas linhas de referencia.
        for x, valor in zip(posicoes, medias):
            eixo.annotate(formato.format(valor).replace(".", ","), xy=(x, 0), xytext=(0, 6),
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
    left.set_ylim(0, max(execucoes["drive_exact"].max() * 100 * 1.15, 50))
    left.set_title("Recuperação entre implementações")
    left.legend(fontsize=8, loc="upper left")

    for chave, cor, estilo, texto in linhas[1:]:
        if chave in baselines.index:
            valor = float(baselines.loc[chave, "mae_db"])
            right.axhline(valor, color=cor, linestyle=estilo, linewidth=1.4,
                          label=f"{texto} ({_db(valor)})")
    right.axhline(passo_db, color="k", linestyle=":", linewidth=1.2,
                  label=f"um degrau da grade ({_db(passo_db)})")
    right.set_ylabel("erro médio (dB equivalentes)")
    right.set_ylim(0, max(tabela["mae_db"].max(), passo_db) * 1.15)
    right.set_title("Distância do ajuste certo")
    right.legend(fontsize=8)

    _save(fig, out_path)


def build_all(results_dir: Path = Path("results/disent/v2/validacao"),
              out_dir: Optional[Path] = None, split: str = "validacao") -> List[Path]:
    """As tres figuras do relatorio, as que tiverem dado em disco: a escada, o
    custo de transferencia (leave-one-arm-out) e a curva de diversidade.

    Baselines em `<results_dir>/b0_<split>.json` e `b1_<split>.json` (de `gefx
    disent retrieve`); o encoder em `<results_dir>/encoder/`. Uma pasta de figuras
    por particao: o teste so tem as suas depois da passada final.
    """
    from gefx.evaluation.loo import cost_file, summary_file
    from gefx.evaluation.retrieval import metrics_file
    from gefx.evaluation.runner import compare

    results_dir = Path(results_dir)
    encoder_dir = results_dir / "encoder"
    out_dir = Path(out_dir or results_dir / f"figuras_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def carrega(caminho: Path):
        return json.loads(caminho.read_text(encoding="utf-8"))

    escritos: List[Path] = []

    def grava(nome: str, desenha, *dados) -> None:
        alvo = out_dir / nome
        desenha(*dados, alvo)
        escritos.append(alvo)

    treinado = encoder_dir / "supcon" / metrics_file(split)
    if not treinado.exists():
        return escritos
    metricas = carrega(treinado)
    chance = drive_chance(metricas)

    grava("escada.png", plot_ladder, compare(encoder_dir, baselines_dir=results_dir, split=split),
          grid_step_db(metricas))

    custo = encoder_dir / "loo" / cost_file(split)
    if custo.exists():
        grava("custo_de_transferencia.png", plot_transfer_cost, pd.read_csv(custo), chance)

    # Uma pasta por semente: `diversidade`, `diversidade_s2`, ...
    curvas = sorted(encoder_dir.glob(f"diversidade*/{summary_file(split)}"))
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
    """Visto x inedito, arm por arm, com o custo em pontos em cima de cada par."""
    ordem = ordered_arms(custo["arm"], dict(zip(custo["arm"], custo["estrato"])))
    dados = custo.set_index("arm").loc[ordem]

    fig, ax = grouped_bar_axes([_short(a) for a in ordem], dados["visto"] * 100,
                               dados["inedito"] * 100, "implementação vista no treino",
                               "implementação inédita", width=0.38, figsize=(10, 5.5))
    for x, (visto, inedito) in enumerate(zip(dados["visto"], dados["inedito"])):
        ax.annotate(f"{(inedito - visto) * 100:+.1f}".replace(".", ","),
                    xy=(x, max(visto, inedito) * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(chance * 100, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label=f"acaso ({_pct(chance)})")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, dados[["visto", "inedito"]].to_numpy().max() * 100 * 1.2)
    ax.set_title("Custo de nunca ter ouvido a implementação")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_path)


# --- B2/B3: quantas implementacoes o treino precisa ver -----------------------
def plot_diversity_curve(curva: pd.DataFrame, chance: float, out_path: Path,
                         visto: Optional[float] = None) -> None:
    """Transferencia para o arm retirado contra o numero de arms no treino.

    O catalogo e o mesmo em todos os pontos; a anotacao marca onde entra um estrato
    novo.
    """
    curva = curva[~curva["centrado"].astype(bool)]  # a curva e a da busca como esta
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
        ax.annotate(_pct(linha["drive_exact"]),
                    xy=(linha["k"], linha["drive_exact"] * 100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    # O arm que entra em cada ponto: e ele, e nao a contagem, que explica os saltos.
    entra = {int(k): str(arms).split("|")[-1] for k, arms
             in pontos.groupby("k")["arms_treinados"].first().items()} \
        if "arms_treinados" in pontos.columns else {}
    rotulos = [f"{int(linha['k'])}\n+ {_short(entra[int(linha['k'])])}"
               if int(linha["k"]) in entra else str(int(linha["k"]))
               for _, linha in transferencia.iterrows()]
    ax.set_xticks(transferencia["k"])
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_xlabel("implementações no treino, e a que entra em cada ponto (catálogo fixo)")
    ax.set_ylabel("acerto exato do nível de drive (%)")
    ax.set_ylim(0, max(curva["drive_exact"].max() * 100 * 1.15, 50))
    ax.set_title("Diversidade de implementação no treino compra transferência?")
    # Embaixo: a metade inferior fica vazia (acaso de 12,5%).
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_path)
