"""Treino do encoder desemaranhado e o estudo comparativo de tecnicas.

Uma tecnica e so um dicionario de pesos sobre `losses.LOSS_REGISTRY`: todas
rodam pelo mesmo codigo, arquitetura, amostrador e semente, e a unica diferenca
entre duas linhas do estudo e o dicionario.

Os criterios (`CRITERIA`) foram declarados antes de rodar, com os baselines da
etapa 4 como limiares.
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

#: A escada do estudo: cada tecnica acrescenta uma ideia a anterior, para que a
#: diferenca entre linhas vizinhas seja atribuivel. `beta_vae` e
#: `random_encoder` sao controles fora da escada.
TECHNIQUES: Dict[str, Dict[str, float]] = {
    "contrastive": {"contrastive": 1.0},
    # O contrastivo ve 40 classes sem vizinhanca; o auxiliar da a ordem dos eixos.
    "contrastive_aux": {"contrastive": 1.0, "aux_regression": 1.0},
    "grl": {
        "contrastive": 1.0,
        "aux_regression": 1.0,
        "adversary_arm": 0.3,
        "adversary_content": 0.3,
    },
    "full": {
        "contrastive": 1.0,
        "aux_regression": 1.0,
        "adversary_arm": 0.3,
        "adversary_content": 0.3,
        "adversary_config": 0.3,
        "orthogonality": 1.0,
    },
    # Nao supervisionado: nao le rotulo nenhum.
    "beta_vae": {},
    # Sem treino: mede o que a arquitetura sozinha ja entrega.
    "random_encoder": {},
}

#: Separam os dois termos que `full` acrescenta a `grl` de uma vez. Ficam fora
#: de `TECHNIQUES` para nao entrar na escada pre-declarada.
WEIGHT_VARIANTS: Dict[str, Dict[str, float]] = {
    "grl_config": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "adversary_config": 0.3,
    },
    "grl_orth": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "orthogonality": 1.0,
    },
    # Separa "o termo atrapalha" de "este peso atrapalha".
    "full_light_orth": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "adversary_arm": 0.3, "adversary_content": 0.3, "adversary_config": 0.3,
        "orthogonality": 0.1,
    },
}


#: Fase 2: `contrastive_aux` mais o decoder. `recon_only` e o controle que
#: reconstroi sem nunca ser cobrado a trocar os blocos.
PHASE2_TECHNIQUES: Dict[str, Dict[str, float]] = {
    "recon_only": {"contrastive": 1.0, "aux_regression": 1.0, "recon": 1.0},
    "swap": {
        "contrastive": 1.0, "aux_regression": 1.0,
        "recon": 1.0, "swap_recon": 1.0,
    },
}

#: Termos que exigem decoder: e a perda, nao uma bandeira, que decide se ha um.
DECODER_TERMS: Tuple[str, ...] = ("recon", "swap_recon")

DEFAULT_DECODER_UNITS: Tuple[int, ...] = (256, 256)


def needs_decoder(weights: Mapping[str, float]) -> bool:
    return any(weights.get(term) for term in DECODER_TERMS)


KNOWN_TECHNIQUES: Dict[str, Dict[str, float]] = {
    **TECHNIQUES, **WEIGHT_VARIANTS, **PHASE2_TECHNIQUES,
}


def technique_weights(name: str) -> Dict[str, float]:
    """Pesos de uma tecnica do estudo, de uma variante de peso ou da fase 2."""
    if name not in KNOWN_TECHNIQUES:
        raise KeyError(f"tecnica desconhecida: {name!r}. Ha {sorted(KNOWN_TECHNIQUES)}")
    return dict(KNOWN_TECHNIQUES[name])


UNTRAINED: Tuple[str, ...] = ("random_encoder",)

#: Ordem em que as referencias dos criterios ficam disponiveis: todos sao
#: julgados contra o `random_encoder`, e o `beta_vae` contra o `contrastive`.
STUDY_ORDER: Tuple[str, ...] = (
    "random_encoder", "contrastive", "contrastive_aux", "grl", "full", "beta_vae",
)

#: Baselines da etapa 4 na tarefa entre implementacoes; limiares de `CRITERIA`.
BASELINES: Dict[str, Dict[str, float]] = {
    "B0_marginal": {"drive_exact": 0.210, "mae_db": 8.15, "top_arm_share": 0.646},
    "B1_poc1_regressor": {"drive_exact": 0.317, "mae_db": 4.36},
    # Sem `mae_db`: o acaso nao estima nivel, e NaN nao e JSON valido.
    "chance": {"drive_exact": 0.125},
    "paired_content_ceiling": {"drive_exact": 0.696, "mae_db": 2.67},
}

#: Todo agregado e reportado tambem sem estes arms (rotulo ruidoso; ver `arms.py`).
EXCLUDED_FROM_AGGREGATE: Tuple[str, ...] = ("byod-bigmuff",)

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
    #: Sem isto a semente so fixa a inicializacao, e duas execucoes divergem.
    #: Desligado por padrao so porque as execucoes publicadas rodaram assim.
    deterministic: bool = False
    #: Desligar quando quem mede e outra coisa (curva de diversidade).
    evaluate_at_end: bool = True
    #: Controle: embaralha a configuracao dentro de cada (conteudo, implementacao).
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
    """Niveis de drive e tone em [0, 1], na escala do recorte e nao da grade cheia."""
    columns = []
    for axis in ("drive_level", "tone_level"):
        values = frame[axis].to_numpy(dtype=np.float32)
        span = max(float(values.max()), 1.0)
        columns.append(values / span)
    return np.stack(columns, axis=1)


# --- passo de treino ----------------------------------------------------------
def map_store(store: FeatureStore, standardizer: PixelStandardizer, fn,
              batch: int = 128) -> Tuple[np.ndarray, ...]:
    """Aplica `fn` a cada bloco padronizado e junta as saidas, na ordem do `frame`.

    `fn` devolve uma tupla de arrays com uma linha por amostra.
    """
    out: Optional[Tuple[np.ndarray, ...]] = None
    for block, features in store.stream(np.arange(len(store)), chunk=batch):
        partes = tuple(np.asarray(parte) for parte in fn(standardizer.transform(features)))
        if out is None:
            out = tuple(np.empty((len(store), *parte.shape[1:]), dtype=np.float32)
                        for parte in partes)
        for destino, parte in zip(out, partes):
            destino[block] = parte
    if out is None:
        raise ValueError("store vazio")
    return out


def _mean_spectra(store: FeatureStore, standardizer: PixelStandardizer,
                  batch: int = 128) -> np.ndarray:
    """Espectro medio padronizado de todas as linhas do `store`, na ordem delas."""
    return map_store(store, standardizer, lambda t: (mean_spectrum(t),), batch)[0]


def mean_spectrum(features):
    """(batch, frequencia, tempo, canal) -> (batch, frequencia).

    Alvo do decoder: o tronco faz a mesma media antes do gargalo, entao e o
    maximo que o codigo pode conter.
    """
    return features[..., 0].mean(axis=2)


def make_step(model: DisentModel, optimizer, weights: Mapping[str, float],
              temperature: float, n_arms: int = 0):
    """Passo de treino com os pesos estaticos dentro do `tf.function`.

    `donor_x`, `swap_spec` e `n_arms` so sao usados quando a tecnica tem decoder.
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
                    "recon_target": tf.reduce_mean(x[..., 0], axis=2),  # mean_spectrum
                    "recon_prediction": model.decode(
                        context["z_e"], context["z_c"], arm_onehot, training=True),
                    "swap_target_spec": swap_spec,
                    # Efeito do doador; conteudo e implementacao da ancora.
                    "swap_prediction": model.decode(
                        z_e_donor, context["z_c"], arm_onehot, training=True),
                })
            parts = total_loss(context, weights)
        gradients = tape.gradient(parts["total"], variables)
        # Cabecas fora da tecnica nao recebem gradiente.
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
    """`(z_e, z_c)` de todas as linhas do `store`, na ordem do `frame`."""
    z_e, z_c = map_store(store, standardizer,
                         lambda t: model.encode(t, training=False), batch)
    return z_e, z_c


def embed(model, store: FeatureStore, standardizer: PixelStandardizer,
          batch: int = 128) -> np.ndarray:
    """`z_e` de todas as linhas do `store`: so ele entra na busca."""
    return embed_blocks(model, store, standardizer, batch)[0]


def evaluate(
    model,
    frames: Mapping[str, pd.DataFrame],
    stores: Mapping[str, FeatureStore],
    standardizer: PixelStandardizer,
    batch: int = 128,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """A tarefa do POC II sobre o codigo aprendido, pelo mesmo caminho do B0."""
    arms = set(frames["catalog"]["arm"].unique())
    if len(arms) < 2:
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
    metrics["same_arm_control"] = same.metrics["overall"]
    return cross.predictions, metrics


def aggregate_without(
    metrics: Mapping[str, object], excluded: Sequence[str] = EXCLUDED_FROM_AGGREGATE
) -> Dict[str, float]:
    """Agregado sobre os arms restantes.

    Media das taxas por arm, que so e o agregado das linhas porque o split de
    consulta e balanceado -- por isso a checagem dos tamanhos.
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
    """Aplica `CRITERIA` as metricas finais.

    Criterio que depende de uma referencia ausente fica de fora, e nao aprovado
    por omissao. `beta_vae_nao_ganha` so se aplica ao proprio `beta_vae`.
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
    """Reaplica `CRITERIA` sobre os `metrics.json` gravados e regrava os `run.json`."""
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
    if not index.frame["file_name"].equals(stores["train"].frame["file_name"]):
        raise ValueError(
            "o indice da grade e o cache de features estao em ordens diferentes"
        )

    standardizer = PixelStandardizer.fit(stores["train"])
    aux = aux_targets(index.frame)

    encoder_config = replace(
        config.encoder, input_shape=(*stores["train"].feature_shape, 1)
    )
    weights = config.resolved_weights()
    if needs_decoder(weights) and not encoder_config.decoder_units:
        encoder_config = replace(encoder_config, decoder_units=DEFAULT_DECODER_UNITS)
    # O manifesto grava a config que rodou, nao a pedida: `load_run` depende disso.
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

    # Os alvos da troca nao dependem do treino: calculados uma vez, em RAM.
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
                # Sem decoder nada consome estes dois; a ancora preenche a assinatura.
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
    """Junta os `run.json` das tecnicas numa tabela, com os baselines por cima."""
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
        if not manifest.get("decision"):  # `evaluate_at_end=False`
            continue
        measured = dict(manifest["decision"]["measured"])
        sem = measured.pop("sem_bigmuff", None)
        rows.append(
            {
                "technique": name,
                "kind": "control" if name == "beta_vae" else "learned",
                **measured,
                "drive_exact_sem_bigmuff": sem["drive_exact"] if sem else None,
                "mae_db_sem_bigmuff": sem["mae_db"] if sem else None,
                # "n de m": o numero de criterios varia por tecnica.
                "verdicts": "{}/{}".format(
                    sum(manifest["decision"]["verdicts"].values()),
                    len(manifest["decision"]["verdicts"]),
                ),
                "steps": manifest.get("steps_executed", manifest["config"]["steps"]),
            }
        )
    return pd.DataFrame(rows)
