"""Treino do encoder desemaranhado e o estudo comparativo de tecnicas.

A perda total nao esta escrita aqui: e a soma ponderada das entradas de
`losses.LOSS_REGISTRY`, escolhidas por `TECHNIQUES`. Essa indirecao e a terceira
costura do compromisso da fase 2 -- `swap_recon` entra como mais um peso, sem
ramo condicional novo no laco -- e e tambem o que torna o estudo comparativo
honesto: as tecnicas comparadas rodam pelo *mesmo* codigo, mesma arquitetura,
mesmo amostrador, mesma semente. A unica coisa que muda entre elas e o
dicionario de pesos.

**Os criterios estao declarados antes de rodar** (`CRITERIA`), com os numeros dos
baselines ja medidos na etapa 4. Declarar depois seria escolher a metrica que o
resultado favorece; e o baseline B1 (31,7% e 4,36 dB) e um alvo desconfortavel de
proposito, porque um encoder que nao o supere nao justifica a mudanca de
formulacao de regressao para recuperacao.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gefx.config import git_revision
from gefx.disent.features import FeatureStore, PixelStandardizer
from gefx.disent.losses import DEFAULT_TEMPERATURE, RAMP_GAMMA, lambda_ramp, total_loss
from gefx.disent.model import BetaVAE, DisentModel, EncoderConfig, HeadConfig
from gefx.disent.retrieval import hubness, retrieve_by_arm
from gefx.disent.sampler import (
    GridIndex,
    build_index,
    class_balanced_batch,
    permute_configs,
)
from gefx.disent.sidecar import read_dataset

#: As tecnicas do estudo comparativo, em ordem de acumulo. Cada uma acrescenta
#: **uma** ideia a anterior, para que a diferenca entre duas linhas vizinhas seja
#: atribuivel. `beta_vae` esta fora dessa escada: e controle negativo.
TECHNIQUES: Dict[str, Dict[str, float]] = {
    # So o contrastivo supervisionado: configuracao junta, o resto que se vire.
    "contrastive": {"contrastive": 1.0},
    # Acrescenta a ordem dos eixos, que o contrastivo nao ve (40 classes sem
    # vizinhanca).
    "contrastive_aux": {"contrastive": 1.0, "aux_regression": 1.0},
    # Acrescenta a remocao ativa de implementacao e conteudo de `z_e`.
    "grl": {
        "contrastive": 1.0,
        "aux_regression": 1.0,
        "adversary_arm": 0.3,
        "adversary_content": 0.3,
    },
    # Acrescenta o lado simetrico (configuracao fora de `z_c`) e a independencia.
    "full": {
        "contrastive": 1.0,
        "aux_regression": 1.0,
        "adversary_arm": 0.3,
        "adversary_content": 0.3,
        "adversary_config": 0.3,
        "orthogonality": 1.0,
    },
    # Controle negativo: nao le rotulo nenhum.
    "beta_vae": {},
    # Controle de arquitetura: os mesmos pesos, sem um passo de treino. Mede o
    # que a forma da rede da de graca -- e ela ja da alguma coisa, porque reduzir
    # 44.288 dimensoes a 32 destroi o efeito de sorvedouro que domina o B0. Sem
    # esta linha, todo o ganho sobre o B0 seria creditado ao aprendizado.
    "random_encoder": {},
}

#: Decomposicao do unico resultado que sobreviveu ao IC agrupado por conteudo:
#: `full` e pior que `grl` por 4,1 pontos. Nao sao tecnicas do estudo -- ficam
#: fora de `TECHNIQUES` de proposito, para nao diluir a escada pre-declarada nem
#: entrar nas figuras da etapa 5. Existem porque `full` acrescenta **dois**
#: termos de uma vez, e a escada so atribui um efeito a uma ideia quando as
#: linhas vizinhas diferem por uma. Sem isto, "a ortogonalidade piora" e uma
#: conclusao que os dados nao sustentam -- pode ser o adversario de configuracao,
#: e pode ser o peso e nao o termo.
WEIGHT_VARIANTS: Dict[str, Dict[str, float]] = {
    # `grl` + o terceiro adversario, sozinho.
    "grl_config": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "adversary_config": 0.3,
    },
    # `grl` + a ortogonalidade, sozinha.
    "grl_orth": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "orthogonality": 1.0,
    },
    # `full` com a ortogonalidade dez vezes mais leve: separa "o termo atrapalha"
    # de "este peso atrapalha". O peso 1,0 nunca foi ajustado.
    "full_light_orth": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "adversary_config": 0.3,
        "orthogonality": 0.1,
    },
}


#: Fase 2: reconstrucao com troca de codigos. Fora de `TECHNIQUES` pela mesma
#: razao que as variantes de peso: a escada da etapa 5 foi pre-declarada e as
#: figuras dela nao podem ser diluidas por uma tecnica que responde outra
#: pergunta. `swap` parte da melhor configuracao medida (contrastivo + auxiliar)
#: e acrescenta os dois termos do decoder.
PHASE2_TECHNIQUES: Dict[str, Dict[str, float]] = {
    # So o autoencoder, sem troca: controle. Mede quanto do espectro o decoder
    # acerta sem nunca ter sido cobrado a separar os blocos.
    "recon_only": {"contrastive": 1.0, "aux_regression": 1.0, "recon": 1.0},
    # A fase 2 propriamente dita.
    "swap": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "recon": 1.0, "swap_recon": 1.0,
    },
}

#: Termos que exigem decoder. Quem define se ha decoder e a PERDA, nao a linha de
#: comando: uma tecnica com `swap_recon` precisa de decoder por definicao, e
#: deixar isso para uma bandeira convidaria a execucao silenciosamente sem ele.
DECODER_TERMS: Tuple[str, ...] = ("recon", "swap_recon")

#: Larguras do decoder quando a tecnica precisa de um e nada foi pedido.
DEFAULT_DECODER_UNITS: Tuple[int, ...] = (256, 256)


def needs_decoder(weights: Mapping[str, float]) -> bool:
    return any(weights.get(term) for term in DECODER_TERMS)


def technique_weights(name: str) -> Dict[str, float]:
    """Pesos de uma tecnica do estudo, de uma variante de peso ou da fase 2."""
    known = {**TECHNIQUES, **WEIGHT_VARIANTS, **PHASE2_TECHNIQUES}
    if name not in known:
        raise KeyError(f"tecnica desconhecida: {name!r}. Ha {sorted(known)}")
    return dict(known[name])


#: Tecnicas que nao executam passo de otimizacao.
UNTRAINED: Tuple[str, ...] = ("random_encoder",)

#: Ordem de execucao do estudo. Nao e a ordem de `TECHNIQUES` nem a de exibicao:
#: e a ordem em que as **referencias** ficam disponiveis. O `random_encoder`
#: precisa vir antes de todos (e o criterio mais duro) e o `beta_vae` depois do
#: `contrastive` (e contra ele que o controle e julgado). Rodar fora desta ordem
#: nao quebra nada -- so deixa criterios de fora, silenciosamente.
STUDY_ORDER: Tuple[str, ...] = (
    "random_encoder", "contrastive", "contrastive_aux", "grl", "full", "beta_vae",
)

#: Baselines da etapa 4, na tarefa entre implementacoes (consulta x catalogo,
#: sem o proprio arm no catalogo). Ficam aqui porque sao os limiares dos
#: criterios, e um criterio sem numero nao e criterio.
BASELINES: Dict[str, Dict[str, float]] = {
    "B0_marginal": {"drive_exact": 0.210, "mae_db": 8.15, "top_arm_share": 0.646},
    "B1_poc1_regressor": {"drive_exact": 0.317, "mae_db": 4.36},
    # O acaso nao tem erro em dB: sortear um nivel nao e uma estimativa de nivel.
    # A chave simplesmente nao existe, em vez de existir como NaN -- NaN nao e
    # JSON valido e nao sobrevive a um ida-e-volta por disco.
    "chance": {"drive_exact": 0.125},
    "paired_content_ceiling": {"drive_exact": 0.696, "mae_db": 2.67},
}

#: Arms que saem do agregado alternativo. O `byod-bigmuff` reprova a porteira de
#: contraste quando medida no audio renderizado (minimo 0,51-0,65 contra o limiar
#: de 1,0) e carrega ruido de rotulo conhecido; a decisao registrada foi mante-lo
#: e **reportar todo agregado tambem sem ele**. Ver `disent/arms.py`.
EXCLUDED_FROM_AGGREGATE: Tuple[str, ...] = ("byod-bigmuff",)

#: Criterios pre-declarados. Avaliados por `decide()` sobre as metricas finais.
CRITERIA: Dict[str, str] = {
    "supera_b1_acerto": "acerto exato de drive > 0,317 (B1, o regressor do POC I)",
    "supera_b1_erro": "erro medio < 4,36 dB (B1)",
    "supera_b0_acerto": "acerto exato de drive > 0,210 (B0, sem aprendizado)",
    "reduz_sorvedouro": (
        "fracao das respostas vinda do arm mais atrator < 0,646 (B0) -- o "
        "adversario de implementacao existe para isso"
    ),
    "abaixo_do_teto": (
        "acerto <= 0,696 (teto com conteudo pareado). Ultrapassar seria sinal de "
        "vazamento, nao de sucesso"
    ),
    "supera_encoder_aleatorio": (
        "acerto > o do encoder nao treinado. E o criterio mais duro: separa o que "
        "o aprendizado trouxe do que a arquitetura ja dava de graca"
    ),
    "beta_vae_nao_ganha": (
        "o controle nao supervisionado nao supera o contrastivo; se superar, e a "
        "supervisao que esta mal empregada (Locatello et al. 2019)"
    ),
}


@dataclass
class TrainConfig:
    """Tudo o que define uma execucao. Vai inteiro para o `run.json`."""

    dataset_root: Path = Path("datasets/disent")
    feature: str = "Spec"
    technique: str = "full"
    arms: Optional[Tuple[str, ...]] = None
    steps: int = 4000
    configs_per_batch: int = 8
    views_per_config: int = 8
    learning_rate: float = 1e-3
    temperature: float = DEFAULT_TEMPERATURE
    ramp_gamma: float = RAMP_GAMMA
    beta: float = 4.0
    eval_every: int = 500
    embed_batch: int = 128
    seed: int = 20260908
    #: Nucleos deterministicos do TensorFlow. Sem isto **a semente nao reproduz a
    #: execucao**: ela fixa a inicializacao bit a bit e nada alem -- duas
    #: execucoes identicas divergem em dezenas de passos, na GPU e tambem na CPU,
    #: e 4.000 passos depois sao dois modelos que diferem em ate 7 pontos numa
    #: metrica de 800 consultas. Medido, e medido tambem o conserto: com a
    #: bandeira ligada, 34 de 34 tensores identicos e perda igual bit a bit.
    #: Custa ~20% de velocidade. O padrao e desligado so porque as execucoes ja
    #: publicadas rodaram assim; para qualquer medida nova, ligar.
    deterministic: bool = False
    #: Avaliacao final. Desligar so faz sentido quando quem mede e outra coisa:
    #: na curva de diversidade o ponto de um arm so nao tem tarefa entre
    #: implementacoes para medir por dentro, e quem pontua e o arm retirado.
    evaluate_at_end: bool = True
    #: Controle de permutacao: embaralha a configuracao dentro de cada
    #: (conteudo, implementacao). Nao e uma tecnica -- e a checagem de que o
    #: ganho vem do rotulo e nao do procedimento de treino.
    permute_labels: bool = False
    output_dir: Optional[Path] = None
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    weights: Optional[Dict[str, float]] = None

    def resolved_weights(self) -> Dict[str, float]:
        if self.weights is not None:
            return dict(self.weights)
        return technique_weights(self.technique)

    def resolved_output(self) -> Path:
        nome = f"{self.technique}_permutado" if self.permute_labels else self.technique
        return Path(self.output_dir or Path("results/disent/etapa5") / nome)

    def as_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["dataset_root"] = str(self.dataset_root)
        out["output_dir"] = str(self.resolved_output())
        out["arms"] = list(self.arms) if self.arms else None
        out["encoder"] = self.encoder.as_dict()
        out["weights"] = self.resolved_weights()
        return out


# --- dados --------------------------------------------------------------------
def split_frames(
    root: Path, arms: Optional[Sequence[str]] = None
) -> Dict[str, pd.DataFrame]:
    """As tres particoes por conteudo, ja restritas aos arms pedidos."""
    data = read_dataset(Path(root))
    if arms is not None:
        data = data[data["arm"].isin(list(arms))]
    frames = {
        name: data[data["split"] == name].reset_index(drop=True)
        for name in ("train", "catalog", "query")
    }
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(f"particoes vazias: {empty}")
    return frames


def aux_targets(frame: pd.DataFrame) -> np.ndarray:
    """Niveis de drive e tone normalizados em [0, 1], na ordem das linhas.

    Normalizados pelo numero de niveis **do recorte**, nao da grade cheia, pela
    mesma razao que o acaso em `retrieval.alphabet` sai do dado: um recorte
    parcial tem outra escala, e usar a da grade cheia mudaria silenciosamente o
    alvo da regressao auxiliar.
    """
    columns = []
    for axis in ("drive_level", "tone_level"):
        values = frame[axis].to_numpy(dtype=np.float32)
        span = max(float(values.max()), 1.0)
        columns.append(values / span)
    return np.stack(columns, axis=1)


# --- passo de treino ----------------------------------------------------------
def _mean_spectra(store: FeatureStore, standardizer: PixelStandardizer,
                  batch: int = 128) -> np.ndarray:
    """Espectro medio padronizado de todas as linhas do `store`, na ordem delas."""
    rows = np.arange(len(store))
    out: Optional[np.ndarray] = None
    for block, features in store.stream(rows, chunk=batch):
        media = mean_spectrum(standardizer.transform(features))
        if out is None:
            out = np.empty((len(store), media.shape[1]), dtype=np.float32)
        out[block] = media
    if out is None:
        raise ValueError("store vazio")
    return out


def mean_spectrum(features):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia).

    E o alvo do decoder da fase 2, e e o que o codigo pode conter: o tronco ja
    faz esta mesma media antes do gargalo (`model._mean_over_time`).
    """
    return features[..., 0].mean(axis=2)


def make_step(model: DisentModel, optimizer, weights: Mapping[str, float],
              temperature: float, n_arms: int = 0):
    """Fecha o passo sobre os pesos: eles sao estaticos dentro do `tf.function`.

    `n_arms` so e usado quando a tecnica tem decoder: a implementacao entra nele
    como one-hot. Com decoder, `donor_x` e `swap_spec` carregam o doador de efeito
    e o espectro do alvo da troca; sem decoder o laco passa a propria ancora nos
    dois e nada os consome -- a bifurcacao e em Python, resolvida no tracamento.
    """
    import tensorflow as tf

    variables = model.trainable_variables
    decode = needs_decoder(weights)

    @tf.function(reduce_retracing=True)
    def step(x, config_label, content_label, arm_label, aux_target, lam,
             donor_x, swap_spec):
        with tf.GradientTape() as tape:
            context = model(x, lam=lam, training=True)
            context = dict(context)
            context.update(
                {
                    "config_label": config_label,
                    "content_label": content_label,
                    "arm_label": arm_label,
                    "aux_target": aux_target,
                    "temperature": temperature,
                }
            )
            if decode:
                arm_onehot = tf.one_hot(arm_label, n_arms)
                z_e_donor, _ = model.encode(donor_x, training=True)
                context.update({
                    # A reconstrucao da ancora sai do proprio `x`: a media no
                    # tempo e de graca e nao custa uma leitura de disco.
                    "recon_target": tf.reduce_mean(x[..., 0], axis=2),
                    "recon_prediction": model.decode(
                        context["z_e"], context["z_c"], arm_onehot, training=True),
                    "swap_target_spec": swap_spec,
                    # A troca: efeito do doador, conteudo da ancora, implementacao
                    # da ancora. E esta linha que a fase 2 inteira afirma.
                    "swap_prediction": model.decode(
                        z_e_donor, context["z_c"], arm_onehot, training=True),
                })
            parts = total_loss(context, weights)
        gradients = tape.gradient(parts["total"], variables)
        # Pesos de cabecas fora da tecnica escolhida nao recebem gradiente. Passar
        # `None` para o otimizador e erro; descarta-los e o comportamento certo.
        pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        optimizer.apply_gradients(pairs)
        return parts

    return step


def make_vae_step(model: BetaVAE, optimizer):
    import tensorflow as tf

    variables = model.trainable_variables

    @tf.function(reduce_retracing=True)
    def step(x):
        with tf.GradientTape() as tape:
            parts = model.losses(x, training=True)
        gradients = tape.gradient(parts["total"], variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients, variables) if g is not None]
        )
        return parts

    return step


# --- avaliacao ----------------------------------------------------------------
def embed_blocks(model, store: FeatureStore, standardizer: PixelStandardizer,
                 batch: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """`(z_e, z_c)` de todas as linhas do `store`, na ordem do `frame`.

    A busca so usa `z_e` (ver `embed`), mas as metricas de estrutura da etapa 6
    precisam dos dois blocos: a afirmacao do desemaranhamento e sobre **onde**
    cada fator esta legivel, e um bloco so nao responde isso.
    """
    rows = np.arange(len(store))
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for block, features in store.stream(rows, chunk=batch):
        z_e, z_c = model.encode(standardizer.transform(features), training=False)
        z_e, z_c = np.asarray(z_e), np.asarray(z_c)
        if out is None:
            out = (np.empty((len(store), z_e.shape[1]), dtype=np.float32),
                   np.empty((len(store), z_c.shape[1]), dtype=np.float32))
        out[0][block], out[1][block] = z_e, z_c
    if out is None:
        raise ValueError("store vazio")
    return out


def embed(model, store: FeatureStore, standardizer: PixelStandardizer,
          batch: int = 128) -> np.ndarray:
    """`z_e` de todas as linhas do `store`, na ordem do `frame`.

    So `z_e` sai daqui: `z_c` nao entra no catalogo por construcao. Se entrasse,
    a busca voltaria a responder por conteudo, que e exatamente o que a etapa 4
    mostrou ser o gargalo.
    """
    return embed_blocks(model, store, standardizer, batch)[0]


def evaluate(
    model,
    frames: Mapping[str, pd.DataFrame],
    stores: Mapping[str, FeatureStore],
    standardizer: PixelStandardizer,
    batch: int = 128,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """A tarefa do POC II sobre o codigo aprendido, no protocolo da etapa 4.

    Passa por `retrieve_by_arm`, o mesmo caminho do B0: mesma exclusao do proprio
    arm do catalogo, mesmo alfabeto, mesmo denominador de sorvedouro. E a metrica
    e cosseno, que e onde o contrastivo trabalha.
    """
    arms = set(frames["catalog"]["arm"].unique())
    if len(arms) < 2:
        # Com uma implementacao so no recorte, "excluir o proprio arm do
        # catalogo" nao deixa candidato nenhum, e a tarefa entre implementacoes
        # simplesmente nao existe. E o caso do B2 na curva de diversidade, onde
        # quem mede e o arm retirado -- ver `TrainConfig.evaluate_at_end`.
        raise ValueError(
            f"a tarefa entre implementacoes precisa de ao menos 2 arms no "
            f"catalogo, ha {sorted(arms)}. Use evaluate_at_end=False e meca "
            f"por fora."
        )
    z_query = embed(model, stores["query"], standardizer, batch)
    z_catalog = embed(model, stores["catalog"], standardizer, batch)

    cross = retrieve_by_arm(
        frames["query"], frames["catalog"], z_query, z_catalog,
        same_arm=False, metric="cosine",
    )
    same = retrieve_by_arm(
        frames["query"], frames["catalog"], z_query, z_catalog,
        same_arm=True, metric="cosine",
    )
    metrics: Dict[str, object] = dict(cross.metrics)
    metrics["hubness"] = hubness(cross.predictions, int(cross.metrics["catalog_size"]))
    # O controle sem travessia: quanto do erro e a travessia entre
    # implementacoes e quanto e a tarefa em si.
    metrics["same_arm_control"] = same.metrics["overall"]
    return cross.predictions, metrics


def aggregate_without(
    metrics: Mapping[str, object], excluded: Sequence[str] = EXCLUDED_FROM_AGGREGATE
) -> Dict[str, float]:
    """Agregado sobre os arms restantes, para acompanhar todo agregado dos sete.

    E media das taxas por arm, e nao das linhas: da no mesmo aqui porque o split
    de consulta e balanceado (800 linhas por arm), e falha ruidosamente se
    algum dia deixar de ser -- o `n` de cada arm vai junto no resultado.
    """
    por_arm = dict(metrics["per_query_arm"])  # type: ignore[index]
    mantidos = [arm for arm in por_arm if arm not in set(excluded)]
    if not mantidos:
        raise ValueError(f"a exclusao de {list(excluded)} esvazia o agregado")
    tamanhos = {int(por_arm[arm]["n"]) for arm in mantidos}
    if len(tamanhos) != 1:
        raise ValueError(
            f"arms com numeros de consultas diferentes ({sorted(tamanhos)}): a media "
            "das taxas por arm deixaria de ser o agregado das linhas"
        )
    return {
        "arms": len(mantidos),
        "excluded": list(excluded),
        "drive_exact": float(
            np.mean([por_arm[arm]["drive_level"]["exact"] for arm in mantidos])
        ),
        "mae_db": float(np.mean([por_arm[arm]["mae_db"] for arm in mantidos])),
    }


def decide(
    metrics: Mapping[str, object],
    references: Optional[Mapping[str, Mapping[str, float]]] = None,
    technique: Optional[str] = None,
) -> Dict[str, object]:
    """Aplica `CRITERIA` as metricas finais. Sem interpretacao, so o veredito.

    Os criterios que dependem de outra execucao so aparecem quando a referencia
    e passada; ausencia de referencia deixa o criterio **de fora**, e nao
    aprovado por omissao. E `beta_vae_nao_ganha` so vale para o proprio controle:
    aplicado a uma tecnica supervisionada ele compararia duas linhas da escada e
    reportaria "falha" por uma delas ser melhor que a outra, que e o contrario do
    que a escada quer mostrar.
    """
    overall = metrics["overall"]  # type: ignore[index]
    hub = metrics.get("hubness", {})
    top_share = max(hub.get("by_retrieved_arm", {"_": 0.0}).values()) if hub else float("nan")
    drive = overall["drive_level"]["exact"]  # type: ignore[index]
    mae_db = overall["mae_db"]  # type: ignore[index]

    verdicts = {
        "supera_b1_acerto": bool(drive > BASELINES["B1_poc1_regressor"]["drive_exact"]),
        "supera_b1_erro": bool(mae_db < BASELINES["B1_poc1_regressor"]["mae_db"]),
        "supera_b0_acerto": bool(drive > BASELINES["B0_marginal"]["drive_exact"]),
        "reduz_sorvedouro": bool(top_share < BASELINES["B0_marginal"]["top_arm_share"]),
        "abaixo_do_teto": bool(drive <= BASELINES["paired_content_ceiling"]["drive_exact"]),
    }
    references = references or {}
    if "random_encoder" in references and technique != "random_encoder":
        verdicts["supera_encoder_aleatorio"] = bool(
            drive > references["random_encoder"]["drive_exact"]
        )
    if technique == "beta_vae" and "contrastive" in references:
        verdicts["beta_vae_nao_ganha"] = bool(
            references["contrastive"]["drive_exact"] >= drive
        )

    measured = {"drive_exact": drive, "mae_db": mae_db, "top_arm_share": top_share}
    if "per_query_arm" in metrics:
        measured["sem_bigmuff"] = aggregate_without(metrics)
    return {
        "criteria": CRITERIA,
        "technique": technique,
        "measured": measured,
        "baselines": BASELINES,
        "verdicts": verdicts,
    }


def rescore(results_dir: Path = Path("results/disent/etapa5")) -> Dict[str, Dict[str, object]]:
    """Reaplica `CRITERIA` sobre as metricas ja gravadas e regrava os `run.json`.

    Existe porque criterio e resultado sao coisas separadas: corrigir a regra nao
    pode custar horas de GPU, e recomputar o veredito a partir de `metrics.json`
    da o mesmo numero que a execucao daria. A ordem de visita e a da escada, que
    e o que garante que as referencias (`random_encoder`, `contrastive`) ja
    existam quando os criterios que dependem delas forem avaliados.
    """
    results_dir = Path(results_dir)
    references: Dict[str, Mapping[str, float]] = {}
    out: Dict[str, Dict[str, object]] = {}
    for name in STUDY_ORDER:
        if not (results_dir / name / "metrics.json").exists():
            continue
        pasta = results_dir / name
        metrics = json.loads((pasta / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((pasta / "run.json").read_text(encoding="utf-8"))
        manifest["decision"] = decide(metrics, references, technique=name)
        (pasta / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        references[name] = manifest["decision"]["measured"]
        out[name] = manifest["decision"]
    return out


# --- laco ---------------------------------------------------------------------
def train(
    config: TrainConfig,
    verbose: bool = True,
    references: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    """Treina uma tecnica e grava tudo o que a torna reproduzivel e auditavel."""
    import keras
    import tensorflow as tf

    if config.deterministic:
        tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(config.seed)

    root = Path(config.dataset_root)
    frames = split_frames(root, config.arms)
    stores = {
        name: FeatureStore(root, frame, config.feature)
        for name, frame in frames.items()
    }
    treino = frames["train"]
    if config.permute_labels:
        treino = permute_configs(treino, seed=config.seed)
    index: GridIndex = build_index(treino, arms=config.arms)
    # O laco indexa o `FeatureStore` com linhas do `GridIndex`. As duas tabelas
    # sao a mesma, mas isso e invariante e nao garantia: um recorte a mais em
    # qualquer um dos dois lados desalinharia features e rotulos em silencio --
    # a mesma armadilha que `features.align_to_frame` existe para evitar.
    if not index.frame["file_name"].equals(stores["train"].frame["file_name"]):
        raise ValueError(
            "o indice da grade e o cache de features estao em ordens diferentes"
        )

    standardizer = PixelStandardizer.fit(stores["train"])
    aux = aux_targets(index.frame)

    # `replace` e nao uma construcao campo a campo: a versao anterior listava os
    # campos um a um e **descartava em silencio** qualquer campo novo do
    # `EncoderConfig`, enquanto o `run.json` gravava a config PEDIDA. O manifesto
    # passava a mentir sobre o modelo que rodou, e uma execucao inteira saia
    # identica a de controle sem nenhum aviso. So o `input_shape` e sobrescrito,
    # porque quem o sabe e a loja de features.
    encoder_config = replace(
        config.encoder, input_shape=(*stores["train"].feature_shape, 1)
    )
    weights = config.resolved_weights()
    if needs_decoder(weights) and not encoder_config.decoder_units:
        encoder_config = replace(encoder_config, decoder_units=DEFAULT_DECODER_UNITS)
    # O manifesto tem de gravar a config que RODOU, e nao a que foi pedida. Sem
    # esta linha, `run.json` dizia `decoder_units: []` numa execucao com decoder,
    # e `load_run` reconstruia o modelo sem ele -- a fase 2 ficava inavaliavel.
    # E a mesma armadilha que o `replace` acima ja tinha consertado do lado do
    # modelo; faltava o lado do registro.
    config = replace(config, encoder=encoder_config)
    heads = HeadConfig(
        n_arms=len(index.arms), n_contents=len(index.contents), n_configs=len(index.configs)
    )

    is_vae = config.technique == "beta_vae"
    steps = 0 if config.technique in UNTRAINED else config.steps
    model = (
        BetaVAE(encoder_config, beta=config.beta)
        if is_vae
        else DisentModel(encoder_config, heads)
    )
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    step = (
        make_vae_step(model, optimizer)
        if is_vae
        else make_step(model, optimizer, weights, config.temperature,
                       n_arms=len(index.arms))
    ) if steps else None

    # O alvo da troca e o espectro MEDIO do alvo, e ele nao depende do treino:
    # calcula-lo uma vez troca uma leitura de disco por passo por 17 MB de RAM.
    # Sem isto cada passo leria tres vezes mais espectrogramas do que a fase 1.
    espectros = (
        _mean_spectra(stores["train"], standardizer, config.embed_batch)
        if steps and not is_vae and needs_decoder(weights) else None
    )

    rng = np.random.default_rng(config.seed)
    history: List[Dict[str, float]] = []
    checkpoints: List[Dict[str, object]] = []
    started = time.time()

    for number in range(1, steps + 1):
        batch = class_balanced_batch(
            index, rng, config.configs_per_batch, config.views_per_config
        )
        features = standardizer.transform(stores["train"].take(batch.rows))
        lam = lambda_ramp(number / steps, config.ramp_gamma)

        if is_vae:
            parts = step(tf.constant(features))
        else:
            if espectros is None:
                # Sem decoder nada consome estes dois; passar a propria ancora
                # mantem uma assinatura so e nao le disco a mais.
                doador, alvo = features, mean_spectrum(features)
            else:
                doador = standardizer.transform(
                    stores["train"].take(batch.effect_donor))
                alvo = espectros[batch.swap_target]
            parts = step(
                tf.constant(features),
                tf.constant(batch.config, dtype=tf.int32),
                tf.constant(batch.content, dtype=tf.int32),
                tf.constant(batch.arm, dtype=tf.int32),
                tf.constant(aux[batch.rows], dtype=tf.float32),
                tf.constant(lam, dtype=tf.float32),
                tf.constant(doador),
                tf.constant(alvo, dtype=tf.float32),
            )
        record = {name: float(value) for name, value in parts.items()}
        record.update({"step": number, "lambda": float(lam)})
        history.append(record)

        if verbose and (number == 1 or number % 100 == 0):
            terms = " ".join(
                f"{name}={record[name]:.4f}" for name in sorted(record)
                if name not in ("step", "lambda", "total")
            )
            print(
                f"  passo {number:5d}/{steps}  total={record['total']:.4f}  "
                f"lambda={lam:.3f}  {terms}",
                flush=True,
            )

        if config.eval_every and number % config.eval_every == 0 and number < steps:
            _, partial = evaluate(model, frames, stores, standardizer, config.embed_batch)
            drive = partial["overall"]["drive_level"]["exact"]  # type: ignore[index]
            checkpoints.append({"step": number, "drive_exact": drive,
                                "mae_db": partial["overall"]["mae_db"]})  # type: ignore[index]
            if verbose:
                print(f"  [avaliacao no passo {number}] drive_exact={drive:.4f} "
                      f"mae_db={partial['overall']['mae_db']:.2f}", flush=True)

    if config.evaluate_at_end:
        predictions, metrics = evaluate(model, frames, stores, standardizer,
                                        config.embed_batch)
    else:
        predictions, metrics = None, None
    elapsed = time.time() - started

    out_dir = config.resolved_output()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(out_dir / "weights")
    standardizer.save(out_dir / "standardizer.npz")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if metrics is not None:
        predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                              encoding="utf-8")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "elapsed_seconds": round(elapsed, 1),
        "config": config.as_dict(),
        "heads": heads.as_dict(),
        "splits": {name: int(len(frame)) for name, frame in frames.items()},
        "steps_executed": steps,
        "checkpoints": checkpoints,
        "decision": (decide(metrics, references, technique=config.technique)
                     if metrics is not None else None),
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(f"  gravado em {out_dir}")
    return manifest


def compare(
    results_dir: Path = Path("results/disent/etapa5"),
    techniques: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Junta os `run.json` das tecnicas numa tabela, com os baselines por cima.

    Le so o que ja esta em disco -- nunca retreina -- pela mesma razao de
    `plots.py`: uma tabela que recalcula pode discordar do artefato que ela
    resume, e ai nao se sabe qual dos dois esta no relatorio.
    """
    results_dir = Path(results_dir)
    wanted = list(techniques) if techniques else sorted(TECHNIQUES)
    rows: List[Dict[str, object]] = []
    for name, numbers in BASELINES.items():
        rows.append({"technique": name, "kind": "baseline", **numbers})
    for name in wanted:
        manifest_path = results_dir / name / "run.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Uma execucao sem avaliacao final (`evaluate_at_end=False`) nao tem
        # veredito para comparar; ela e medida por fora.
        if not manifest.get("decision"):
            continue
        measured = dict(manifest["decision"]["measured"])
        sem = measured.pop("sem_bigmuff", None)
        rows.append(
            {
                "technique": name,
                "kind": "control" if name == "beta_vae" else "learned",
                **measured,
                # O agregado sem o arm de rotulo ruidoso acompanha o dos sete por
                # decisao registrada: sem ele nao da para saber quanto do numero
                # e o `byod-bigmuff`.
                "drive_exact_sem_bigmuff": sem["drive_exact"] if sem else None,
                "mae_db_sem_bigmuff": sem["mae_db"] if sem else None,
                # "n de m": o numero de criterios varia por tecnica (os que
                # dependem de referencia so existem para quem tem referencia),
                # entao a contagem crua nao e comparavel entre linhas.
                "verdicts": "{}/{}".format(
                    sum(manifest["decision"]["verdicts"].values()),
                    len(manifest["decision"]["verdicts"]),
                ),
                "steps": manifest.get("steps_executed", manifest["config"]["steps"]),
            }
        )
    return pd.DataFrame(rows)
