"""Fase 2: a troca de codigos vale? Avaliacao **nao circular**.

A tentacao seria encodar o espectro decodificado e perguntar ao proprio encoder
de que configuracao ele parece. Isso mediria o encoder concordando consigo
mesmo. Aqui nada e pontuado pelo encoder: as duas medidas comparam o espectro
decodificado com **gravacoes reais em disco**, e so.

O que a grade totalmente cruzada permite, e que e o presente do desenho: para
qualquer ancora `a` e doador `b`, o alvo da troca

    t = x[conteudo(a), configuracao(b), implementacao(a)]

existe. Entao da para perguntar duas coisas de forma direta:

1. **Erro contra o alvo**, com os tres pontos de referencia que dao escala a ele:
   o piso (decodificar o proprio alvo), a identidade (nao trocar nada -- e o que
   se obtem se o `z_e` nao carregar efeito) e a media do recorte (o que um
   decoder que ignora tudo entrega).
2. **A que configuracao o espectro decodificado se parece**, comparando-o com as
   40 gravacoes reais `x[conteudo(a), *, implementacao(a)]`. Se a troca funciona,
   a resposta e a configuracao do doador. O acaso e 1/40 para a configuracao
   exata e 1/8 para o nivel de drive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: Quantos pares (ancora, doador) sorteados. 2.000 ja deixa o erro padrao de uma
#: proporcao abaixo de 1,2 ponto, e o custo e dominado pela decodificacao.
DEFAULT_PAIRS = 2000


def _onehot(indices: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(indices), width), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def swap_fidelity(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "query",
    feature: str = "Spec",
    pairs: int = DEFAULT_PAIRS,
    seed: int = 0,
    batch: int = 256,
) -> Dict[str, object]:
    """Mede a troca de codigos numa execucao ja treinada, em conteudo inedito."""
    from gefx.disent.diagnostics import load_run
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.sampler import GridIndex
    from gefx.disent.train import _mean_spectra, embed_blocks, split_frames

    run_dir = Path(run_dir)
    model, manifest = load_run(run_dir)
    if model.decoder is None:
        raise ValueError(f"{run_dir} nao tem decoder: nada a medir na fase 2")

    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    index = GridIndex(frame)
    store = FeatureStore(dataset_root, index.frame, feature)
    standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
    z_e, z_c = embed_blocks(model, store, standardizer, batch)
    espectros = _mean_spectra(store, standardizer, batch)

    n_contents, n_configs, n_arms = index.shape
    rng = np.random.default_rng(seed)
    ancoras = rng.integers(0, len(index.frame), size=pairs)
    rotulos = index.labels(ancoras)
    # O doador tem de ter conteudo diferente: com o mesmo conteudo o alvo da troca
    # seria o proprio doador e a reconstrucao nao exigiria separar nada.
    conteudo_doador = (rotulos["content"]
                       + rng.integers(1, n_contents, size=pairs)) % n_contents
    config_doador = rng.integers(0, n_configs, size=pairs)
    doadores = index.lookup[conteudo_doador, config_doador,
                            rng.integers(0, n_arms, size=pairs)]
    alvos = index.lookup[rotulos["content"], config_doador, rotulos["arm"]]

    arm_oh = _onehot(rotulos["arm"], n_arms)
    decodifica = lambda e, c, a: np.asarray(model.decode(e, c, a, training=False))
    troca = decodifica(z_e[doadores], z_c[ancoras], arm_oh)
    identidade = decodifica(z_e[ancoras], z_c[ancoras], arm_oh)
    piso = decodifica(z_e[alvos], z_c[alvos], _onehot(index.labels(alvos)["arm"], n_arms))

    alvo_spec = espectros[alvos]
    erro = lambda p: float(np.mean((p - alvo_spec) ** 2))
    media_do_recorte = np.repeat(espectros.mean(axis=0, keepdims=True), pairs, axis=0)

    # --- a que configuracao o espectro decodificado se parece ------------------
    # Comparacao com gravacoes REAIS: nenhuma nota aqui passa pelo encoder.
    candidatos = index.lookup[rotulos["content"][:, None],
                              np.arange(n_configs)[None, :],
                              rotulos["arm"][:, None]]            # (pares, 40)
    def config_mais_proxima(predicao: np.ndarray) -> np.ndarray:
        distancias = np.einsum(
            "pcf,pcf->pc",
            (d := espectros[candidatos] - predicao[:, None, :]), d,
        )
        return distancias.argmin(axis=1)

    grade = index.frame.drop_duplicates("config_index").set_index("config_index")
    drive = grade["drive_level"].reindex(index.configs).to_numpy()
    tone = grade["tone_level"].reindex(index.configs).to_numpy()

    saida: Dict[str, object] = {
        "run_dir": str(run_dir), "split": split, "pares": int(pairs),
        "erro": {"troca": erro(troca), "identidade": erro(identidade),
                 "piso": erro(piso), "media_do_recorte": erro(media_do_recorte)},
    }
    for rotulo, predicao, esperado in (("troca", troca, config_doador),
                                       ("identidade", identidade, rotulos["config"])):
        achada = config_mais_proxima(predicao)
        saida[f"leitura_{rotulo}"] = {
            "config_exata": float(np.mean(achada == esperado)),
            "acaso_config": 1.0 / n_configs,
            "drive_exato": float(np.mean(drive[achada] == drive[esperado])),
            "acaso_drive": float(1.0 / len(set(drive))),
            "tone_exato": float(np.mean(tone[achada] == tone[esperado])),
            "acaso_tone": float(1.0 / len(set(tone))),
            # Le como a configuracao da ANCORA: se a troca nao fez nada, e isto
            # que sobe, e e o controle que impede ler sorte como sucesso.
            "leu_a_ancora": float(np.mean(achada == rotulos["config"])),
        }
    return saida


def swap_study(
    results_dir: Path = Path("results/disent/fase2"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Optional[List[str]] = None,
    split: str = "query",
    pairs: int = DEFAULT_PAIRS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por execucao com decoder encontrada sob `results_dir`."""
    results_dir = Path(results_dir)
    nomes = techniques or sorted(
        p.name for p in results_dir.iterdir()
        if p.is_dir() and (p / "run.json").exists()
    )
    linhas: List[Dict[str, object]] = []
    for nome in nomes:
        caminho = results_dir / nome
        try:
            relatorio = swap_fidelity(caminho, dataset_root, split=split,
                                      pairs=pairs, seed=seed)
        except ValueError:
            continue  # execucao sem decoder: nao e da fase 2
        linha: Dict[str, object] = {"technique": nome}
        linha.update({f"erro_{k}": v for k, v in relatorio["erro"].items()})
        for rotulo in ("troca", "identidade"):
            linha.update({f"{rotulo}_{k}": v
                          for k, v in relatorio[f"leitura_{rotulo}"].items()
                          if not k.startswith("acaso")})
        linhas.append(linha)
    if not linhas:
        raise FileNotFoundError(f"nenhuma execucao com decoder em {results_dir}")
    return pd.DataFrame(linhas)
