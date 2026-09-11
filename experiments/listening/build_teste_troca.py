"""Teste de escuta 4: a troca de codigos da fase 2 e audivel?

A AFIRMACAO TESTADA, e so ela: a troca moveu o efeito **na direcao do doador e
para longe da ancora**. E a metade da afirmacao da fase 2 que um erro quadratico
nao verifica, e e a que tem os numeros mais fortes (le o doador a 4,2x o acaso,
le a ancora NO acaso).

O ENSAIO
  referencia = a gravacao DOADORA         (frase B, config Y, arm_d)
  candidato S = o que a troca recuperou   (frase A, config Y_hat, arm_a)
  candidato N = a propria ancora          (frase A, config X,     arm_a)
Pergunta: qual dos dois tem a quantidade de distorcao mais proxima da
referencia? Previsao da fase 2: o candidato S. Acaso: 50%.

POR QUE ESTE DESENHO E MAIS LIMPO QUE OS TRES ANTERIORES
Os dois candidatos estao no **mesmo arm e com o mesmo conteudo** por construcao
-- sao ambos `x[conteudo(a), *, arm(a)]`. Entao o nivel nominal entre eles e
comparavel (o problema que invalidou metade do teste 3 nao alcanca aqui) e a
frase e identica nos dois, o que remove o conteudo como pista.

FILTROS, TODOS PRE-DECLARADOS (ver a memoria `como-montar-teste-de-escuta`)
  F1  Y_hat != X: os dois candidatos precisam ser arquivos diferentes.
  F2  |drive(Y_hat) - drive(X)| >= 2, no MESMO arm, entao nominal vale aqui.
  F3  o nivel do doador mapeado para arm_a pelo ORACULO difere de drive(X) por
      >= 2 -- sem isto nao ha movimento a detectar. Atravessa arms, entao usa o
      oraculo e nao o rotulo nominal.
  F4  tone(Y_hat) == tone(X): os dois candidatos diferem SO em drive. Sem isto a
      pergunta "quanta distorcao" fica confundida pelo filtro de tom.
  F5  drive dos dois candidatos na metade suja (>= 3): um degrau e audivel 11/12
      la e 6/12 na metade limpa.

Nenhum filtro pergunta se o modelo acertou -- so se o ensaio e respondivel.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path("datasets/disent")
SAIDA = Path("datasets/teste_troca")
EXECUCAO = Path("results/disent/fase2/swap")
ALINHAMENTO = Path("results/disent/etapa5/oracle_alignment.npz")
N_ENSAIOS = 24
SEMENTE = 20260911


def respostas_da_troca(pares: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Uma linha por par (ancora, doador), com o que a troca recuperou."""
    from gefx.disent.diagnostics import load_run
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.sampler import GridIndex
    from gefx.disent.swap import _onehot
    from gefx.disent.train import _mean_spectra, embed_blocks, split_frames

    modelo, manifesto = load_run(EXECUCAO)
    frame = split_frames(RAIZ, manifesto["config"].get("arms"))["query"]
    index = GridIndex(frame)
    loja = FeatureStore(RAIZ, index.frame, "Spec")
    padrao = PixelStandardizer.load(EXECUCAO / "standardizer.npz")
    z_e, z_c = embed_blocks(modelo, loja, padrao, 256)
    espectros = _mean_spectra(loja, padrao, 256)

    n_conteudos, n_configs, n_arms = index.shape
    rng = np.random.default_rng(seed)
    ancoras = rng.integers(0, len(index.frame), size=pares)
    rot = index.labels(ancoras)
    conteudo_doador = (rot["content"] + rng.integers(1, n_conteudos, size=pares)) % n_conteudos
    config_doador = rng.integers(0, n_configs, size=pares)
    arm_doador = rng.integers(0, n_arms, size=pares)
    doadores = index.lookup[conteudo_doador, config_doador, arm_doador]

    predicao = np.asarray(modelo.decode(z_e[doadores], z_c[ancoras],
                                        _onehot(rot["arm"], n_arms), training=False))
    candidatos = index.lookup[rot["content"][:, None], np.arange(n_configs)[None, :],
                              rot["arm"][:, None]]
    distancias = np.einsum("pcf,pcf->pc",
                           (d := espectros[candidatos] - predicao[:, None, :]), d)
    escolhida = distancias.argmin(axis=1)

    linhas = index.frame
    return pd.DataFrame({
        "ancora": ancoras, "doador": doadores,
        "recuperada": candidatos[np.arange(pares), escolhida],
        "arm_ancora": linhas["arm"].to_numpy()[ancoras],
        "arm_doador": linhas["arm"].to_numpy()[doadores],
        "conteudo_ancora": linhas["content_id"].to_numpy()[ancoras],
        "drive_ancora": linhas["drive_level"].to_numpy()[ancoras],
        "tone_ancora": linhas["tone_level"].to_numpy()[ancoras],
        "drive_doador": linhas["drive_level"].to_numpy()[doadores],
        "drive_recuperada": linhas["drive_level"].to_numpy()[
            candidatos[np.arange(pares), escolhida]],
        "tone_recuperada": linhas["tone_level"].to_numpy()[
            candidatos[np.arange(pares), escolhida]],
    })


def aplica_filtros(tabela: pd.DataFrame) -> pd.DataFrame:
    """Os cinco filtros pre-declarados. Nenhum olha se o modelo acertou."""
    alinhamento = np.load(ALINHAMENTO, allow_pickle=True)
    arms = list(alinhamento["arms"])
    corresp = alinhamento["correspondence"]  # [arm_i, arm_j, nivel_i] -> nivel_j
    indice = {nome: i for i, nome in enumerate(arms)}

    tabela = tabela.copy()
    tabela["doador_no_arm_da_ancora"] = [
        int(corresp[indice[d], indice[a], nivel])
        for d, a, nivel in zip(tabela["arm_doador"], tabela["arm_ancora"],
                               tabela["drive_doador"])
    ]
    f1 = tabela["recuperada"] != tabela["ancora"]
    f2 = (tabela["drive_recuperada"] - tabela["drive_ancora"]).abs() >= 2
    f3 = (tabela["doador_no_arm_da_ancora"] - tabela["drive_ancora"]).abs() >= 2
    f4 = tabela["tone_recuperada"] == tabela["tone_ancora"]
    f5 = (tabela["drive_recuperada"] >= 3) & (tabela["drive_ancora"] >= 3)
    tabela["passa"] = f1 & f2 & f3 & f4 & f5
    for nome, mascara in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4), ("F5", f5)):
        print(f"  {nome}: {int(mascara.sum()):5d} de {len(tabela)}")
    print(f"  todos: {int(tabela['passa'].sum())}")
    return tabela


def monta(tabela: pd.DataFrame, n: int = N_ENSAIOS, seed: int = SEMENTE) -> pd.DataFrame:
    """Sorteia os ensaios ESTRATIFICADOS pelo veredito do oraculo.

    Sem estratificar, os filtros entregam 270 ensaios em que o oraculo da razao a
    troca e 17 em que ele da razao a ancora -- e o teste deixaria de medir o
    modelo. A causa e que o eixo e limitado (0-7): o F2 poe a recuperada longe da
    ancora e o F3 poe o doador longe da ancora, e numa escala limitada "os dois
    longe" quase sempre quer dizer "os dois do mesmo lado", logo perto um do
    outro. Os filtros que existiam para garantir boa-postura acabavam
    selecionando os casos em que o modelo acertou.

    Metade e metade, entao, e o teste passa a ter poder: no estrato discordante o
    ouvido PRECISA discordar do modelo se estiver seguindo o som.
    """
    from gefx.disent.train import split_frames

    frame = split_frames(RAIZ)["query"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    elegiveis = tabela[tabela["passa"]].copy()
    elegiveis["oraculo_da_razao_a_troca"] = (
        (elegiveis["doador_no_arm_da_ancora"] - elegiveis["drive_recuperada"]).abs()
        < (elegiveis["doador_no_arm_da_ancora"] - elegiveis["drive_ancora"]).abs()
    )
    metade = n // 2
    partes = []
    for veredito in (True, False):
        estrato = elegiveis[elegiveis["oraculo_da_razao_a_troca"] == veredito]
        # Variedade de plugin dentro de cada estrato.
        parte = (estrato.sample(frac=1.0, random_state=seed)
                 .groupby("arm_ancora").head(3).head(metade))
        if len(parte) < metade:
            raise SystemExit(
                f"estrato oraculo={veredito}: so {len(parte)} de {metade} pedidos"
            )
        partes.append(parte)
    escolhidos = pd.concat(partes).sample(frac=1.0, random_state=seed + 1)

    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for numero, (_, r) in enumerate(escolhidos.iterrows(), start=1):
        pasta = SAIDA / f"ensaio{numero:02d}"
        pasta.mkdir(exist_ok=True)
        # O audio mora em <arm>/<chain_key>/<arquivo>; `chain_key` e `distortion`
        # neste dataset. E a linha abaixo confere que o indice guardado na tabela
        # aponta para a MESMA linha do sidecar -- desalinhamento de indice ja
        # custou caro neste projeto.
        def caminho(i: int) -> Path:
            return RAIZ / frame.loc[i, "arm"] / "distortion" / frame.loc[i, "file_name"]

        assert frame.loc[int(r["ancora"]), "arm"] == r["arm_ancora"]
        assert frame.loc[int(r["ancora"]), "content_id"] == r["conteudo_ancora"]
        assert frame.loc[int(r["doador"]), "arm"] == r["arm_doador"]
        posicao_da_troca = int(rng.integers(1, 3))   # 1 ou 2, sorteada
        arquivos = {posicao_da_troca: int(r["recuperada"]),
                    3 - posicao_da_troca: int(r["ancora"])}
        shutil.copy(caminho(int(r["doador"])), pasta / "referencia.wav")
        for posicao, linha in arquivos.items():
            shutil.copy(caminho(linha), pasta / f"{posicao}.wav")
        # O arbitro do oraculo, gravado ANTES das respostas: ele diz qual
        # candidato esta mais perto do doador na escala do arm da ancora. Metade
        # dos ensaios tem este campo FALSO de proposito -- sao os casos em que o
        # modelo errou, e sao eles que permitem ver se o ouvido segue o som ou o
        # modelo.
        perto_da_troca = bool(r["oraculo_da_razao_a_troca"])
        linhas.append({
            "ensaio": numero, "posicao_da_troca": posicao_da_troca,
            "arm_ancora": r["arm_ancora"], "arm_doador": r["arm_doador"],
            "conteudo_ancora": r["conteudo_ancora"],
            "drive_ancora": int(r["drive_ancora"]),
            "drive_recuperada": int(r["drive_recuperada"]),
            "drive_doador": int(r["drive_doador"]),
            "doador_no_arm_da_ancora": int(r["doador_no_arm_da_ancora"]),
            "tone_dos_candidatos": int(r["tone_ancora"]),
            "oraculo_da_razao_a_troca": bool(perto_da_troca),
        })
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print("calculando as respostas da troca...")
    # Tres sorteios: o estrato discordante e raro (~6% dos elegiveis) e um
    # sorteio so nao entrega ensaios suficientes dele.
    bruta = pd.concat([respostas_da_troca(seed=s) for s in (0, 1, 2)],
                      ignore_index=True)
    print("filtros:")
    filtrada = aplica_filtros(bruta)
    gabarito = monta(filtrada)
    destino = Path("experiments/listening/teste_troca_gabarito.csv")
    gabarito.to_csv(destino, index=False)
    respostas = Path("datasets/teste_troca/respostas.csv")
    respostas.write_text(
        "ensaio,escolha  # 1 ou 2; acrescente ' # chute' se nao houve base\n"
        + "".join(f"{n},\n" for n in gabarito["ensaio"]), encoding="utf-8")
    print(f"\ngabarito em {destino}")
    print(f"respostas em {respostas}")
    print(gabarito.to_string(index=False))
