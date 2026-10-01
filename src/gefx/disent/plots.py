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
    from gefx.disent.train import TECHNIQUES

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
              out_dir: Optional[Path] = None) -> List[Path]:
    """As tres figuras do relatorio, as que tiverem dado em disco: a escada, o
    custo de transferencia (leave-one-arm-out) e a curva de diversidade.

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

    treinado = encoder_dir / "supcon" / "metrics.json"
    if not treinado.exists():
        return escritos
    metricas = carrega(treinado)
    chance = drive_chance(metricas)

    grava("escada.png", plot_ladder, compare(encoder_dir, baselines_dir=results_dir),
          grid_step_db(metricas))

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
    """Visto x inedito, arm por arm, com o custo em pontos em cima de cada par."""
    ordem = ordered_arms(custo["arm"], dict(zip(custo["arm"], custo["estrato"])))
    dados = custo.set_index("arm").loc[ordem]
    posicoes = np.arange(len(ordem))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(posicoes - largura / 2, dados["visto"] * 100, largura,
           color=COLOR_B1, label="implementação vista no treino")
    ax.bar(posicoes + largura / 2, dados["inedito"] * 100, largura,
           color=COLOR_CEILING, label="implementação inédita")
    for x, (visto, inedito) in enumerate(zip(dados["visto"], dados["inedito"])):
        ax.annotate(f"{(inedito - visto) * 100:+.1f}".replace(".", ","),
                    xy=(x, max(visto, inedito) * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(chance * 100, color=COLOR_CHANCE, linestyle=":", linewidth=1.3,
               label=f"acaso ({_pct(chance)})")
    ax.set_xticks(posicoes)
    ax.set_xticklabels([_short(a) for a in ordem], rotation=25, ha="right", fontsize=8)
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
    from gefx.disent.loo import centered_rows

    curva = curva[~centered_rows(curva)]  # a curva e a da busca como esta
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
