"""Treino do encoder e os dois controles que separam aprendizado de arquitetura.

- `supcon`: o encoder, treinado com o SupCon (`losses.py`) sobre `z_e`;
- `regressao`: o mesmo tronco com uma saida so, o drive em dB normalizado, por
  MSE. Controla se `z_e` precisa de mais de uma dimensao para o nivel;
- `random_encoder`: a mesma rede sem nenhum passo, o que a arquitetura ja entrega;
- `bn_only`: a rede sem treino com so as estatisticas moveis da BatchNorm
  calibradas, sem gradiente. Com zero passos elas ficam na inicializacao (0, 1),
  sem relacao com as ativacoes reais, o que deixa o `random_encoder` em
  desvantagem artificial.

Os quatro rodam pelo mesmo codigo, arquitetura, amostrador e semente.

Protocolo: ate `steps` passos, com o erro de validacao (a mesma busca do teste,
em dB, sobre `val_query` e `val_catalog`) medido a cada `eval_every`. O treino
para quando ele passa `patience` avaliacoes sem descer, e os pesos do minimo sao
os que ficam -- a semantica de `keras.callbacks.EarlyStopping` com
`restore_best_weights=True` (Goodfellow et al. 2016, alg. 7.1; a paciencia longa
segue Prechelt 1998, em que criterios mais lentos generalizam um pouco melhor).

A taxa de aprendizado decai em cosseno ao longo de `steps`, de `learning_rate` a
`learning_rate * lr_min_fraction`, como no `SupConLoss` oficial (`--cosine` em
HobbitLong/SupContrast: ate `lr * 0.1 ** 3`). E o `keras.optimizers.schedules.
CosineDecay`. O POC I tinha taxa fixa; com ela a validacao oscilava o bastante
(0,08 dB no SupCon, 0,32 dB na regressao) para o minimo ser um vale isolado.

O treino so le a validacao: as metricas que grava (`metrics_validacao.json`) sao
dela. O teste fica para `evaluate_runs(split="teste")`, uma passada no fim com os
modelos finais.
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
from gefx.disent.losses import DEFAULT_TEMPERATURE, sup_con_loss
from gefx.disent.model import REGRESSION, WEIGHTS_FILE, EncoderConfig, build_model
from gefx.disent.retrieval import alphabet, nearest_level, retrieve_by_arm, score
from gefx.disent.grid import EVAL_SPLITS, VALIDATION_RECORDINGS
from gefx.disent.sampler import GridIndex, build_index, class_balanced_batch
from gefx.disent.sidecar import split_frames

TECHNIQUES: Tuple[str, ...] = ("random_encoder", "bn_only", "supcon", REGRESSION)

RESULTS_ROOT = Path("results/disent/v2/validacao/encoder")

#: Arquivos de `gefx disent retrieve`, ao lado de `RESULTS_ROOT`: `<nome>_<particao>.json`.
BASELINES: Dict[str, str] = {"B0": "b0", "B1": "b1"}


def metrics_file(split: str) -> str:
    return f"metrics_{split}.json"


def predictions_file(split: str) -> str:
    return f"predictions_{split}.csv"


@dataclass
class TrainConfig:
    """Tudo o que define uma execucao. Vai inteiro para o `run.json`."""

    dataset_root: Path = Path("datasets/disent_v2")
    feature: str = "Spec"
    technique: str = "supcon"
    arms: Optional[Tuple[str, ...]] = None
    #: O teto. Acima do ponto em que as curvas de 16.000 passos se achataram (~10k).
    steps: int = 20000
    configs_per_batch: int = 8
    views_per_config: int = 8
    learning_rate: float = 1e-3
    #: "cosine" (o SupCon oficial) ou "constant" (o POC I).
    lr_schedule: str = "cosine"
    #: O fim do cosseno, em fracao da taxa inicial: `0.1 ** 3`, como no original.
    lr_min_fraction: float = 1e-3
    temperature: float = DEFAULT_TEMPERATURE
    eval_every: int = 500
    #: Avaliacoes sem descer o erro de validacao antes de parar; 0 nao para.
    patience: int = 16
    #: Gravacoes de cada lado da validacao, tiradas do treino.
    validation_recordings: int = VALIDATION_RECORDINGS
    embed_batch: int = 128
    seed: int = 20260908
    #: Sem isto a semente so fixa a inicializacao, e duas execucoes divergem.
    deterministic: bool = False
    #: Desligar quando quem mede e outra coisa (curva de diversidade).
    evaluate_at_end: bool = True
    output_dir: Optional[Path] = None
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def resolved_output(self) -> Path:
        return Path(self.output_dir or RESULTS_ROOT / self.technique)

    def as_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["dataset_root"] = str(self.dataset_root)
        out["output_dir"] = str(self.resolved_output())
        out["arms"] = list(self.arms) if self.arms else None
        out["encoder"] = self.encoder.as_dict()
        return out


# --- passo de treino ----------------------------------------------------------
def learning_rate(config: TrainConfig):
    """A taxa de `config`: fixa, ou o `CosineDecay` do Keras ao longo de `steps`."""
    import keras

    if config.lr_schedule == "constant":
        return config.learning_rate
    if config.lr_schedule == "cosine":
        return keras.optimizers.schedules.CosineDecay(
            config.learning_rate, decay_steps=config.steps, alpha=config.lr_min_fraction)
    raise ValueError(f"lr_schedule desconhecido: {config.lr_schedule!r}")


def make_optimizer(config: TrainConfig):
    import keras

    return keras.optimizers.Adam(learning_rate=learning_rate(config))


def make_step(model, optimizer, temperature: float):
    import tensorflow as tf

    variables = model.trainable_variables

    @tf.function(reduce_retracing=True)
    def step(x, config_label):
        with tf.GradientTape() as tape:
            z_e = model(x, training=True)
            # Uma vista por linha: [bsz, n_views=1, dim], como o SupConLoss pede.
            # O rotulo e o nivel de drive (`config_index == drive_level`).
            loss = sup_con_loss(z_e[:, None, :], config_label, temperature)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    return step


def make_regression_step(model, optimizer):
    """MSE entre a saida sigmoide e o drive em [0, 1], como o regressor do POC I."""
    import tensorflow as tf

    variables = model.trainable_variables

    @tf.function(reduce_retracing=True)
    def step(x, drive):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True) - drive))
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    return step


def drive_range(frame: pd.DataFrame) -> Tuple[float, float]:
    """Os extremos da escada em dB: o regressor preve dentro deles, em [0, 1]."""
    values = frame["drive_db_equivalente"]
    return float(values.min()), float(values.max())


def make_bn_step(model):
    """Passo para frente em modo de treino: so as medias moveis da BatchNorm mudam."""
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def step(x):
        model(x, training=True)

    return step


# --- avaliacao ----------------------------------------------------------------
def embed(model, store: FeatureStore, standardizer: PixelStandardizer,
          batch: int = 128) -> np.ndarray:
    """`z_e` de todas as linhas do `store`, na ordem do `frame`."""
    out = np.empty((len(store), model.output_shape[-1]), dtype=np.float32)
    for block, features in store.stream(np.arange(len(store)), chunk=batch):
        out[block] = np.asarray(
            model(standardizer.transform(features), training=False))
    return out


def evaluate(
    model,
    frames: Mapping[str, pd.DataFrame],
    stores: Mapping[str, FeatureStore],
    standardizer: PixelStandardizer,
    batch: int = 128,
    split: str = "validacao",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """A tarefa do POC II sobre o codigo aprendido, pelo mesmo caminho do B0."""
    query_key, catalog_key = EVAL_SPLITS[split]
    arms = set(frames[catalog_key]["arm"].unique())
    if len(arms) < 2:
        raise ValueError(
            f"a tarefa entre implementacoes precisa de ao menos 2 arms no "
            f"catalogo, ha {sorted(arms)}. Use evaluate_at_end=False e meca "
            f"por fora."
        )
    queries, catalog = frames[query_key], frames[catalog_key]
    z_query = embed(model, stores[query_key], standardizer, batch)
    z_catalog = embed(model, stores[catalog_key], standardizer, batch)
    scalar = z_query.shape[1] == 1

    metric = "euclidean" if scalar else "cosine"
    cross = retrieve_by_arm(queries, catalog, z_query, z_catalog,
                            same_arm=False, metric=metric)
    same = retrieve_by_arm(queries, catalog, z_query, z_catalog,
                           same_arm=True, metric=metric)
    metrics: Dict[str, object] = dict(cross.metrics)
    metrics["split"] = split
    metrics["same_arm_control"] = same.metrics["overall"]
    if scalar:
        metrics["direct_readout"] = direct_readout(queries, frames["train"], z_query[:, 0])
    return cross.predictions, metrics


def validation_error(
    model,
    frames: Mapping[str, pd.DataFrame],
    stores: Mapping[str, FeatureStore],
    standardizer: PixelStandardizer,
    batch: int = 128,
) -> Dict[str, float]:
    """A busca do teste sobre a validacao: o erro em dB que decide a parada.

    Entre implementacoes, como o teste; com um arm so no treino (a curva de
    diversidade), dentro dele, que e a unica busca possivel sem olhar o retirado.
    """
    queries, catalog = frames["val_query"], frames["val_catalog"]
    z_query = embed(model, stores["val_query"], standardizer, batch)
    z_catalog = embed(model, stores["val_catalog"], standardizer, batch)
    same_arm = catalog["arm"].nunique() < 2
    result = retrieve_by_arm(
        queries, catalog, z_query, z_catalog, same_arm=same_arm,
        metric="euclidean" if z_query.shape[1] == 1 else "cosine",
    )
    overall = result.metrics["overall"]
    return {"mae_db": float(overall["mae_db"]),
            "drive_exact": float(overall["drive_level"]["exact"]),
            "same_arm": bool(same_arm)}


def direct_readout(queries: pd.DataFrame, train_frame: pd.DataFrame,
                   predicted: np.ndarray) -> Dict[str, object]:
    """O regressor lido sem catalogo: o degrau da escada mais proximo, como no B1."""
    lo, hi = drive_range(train_frame)
    drive_db = lo + predicted * (hi - lo)
    ladder = np.array(sorted(queries.groupby("drive_level")["drive_db_equivalente"].first()))
    out = pd.DataFrame({
        "query_arm": queries["arm"].to_numpy(),
        "retrieved_arm": "(leitura direta)",
        "true_drive_level": queries["drive_level"].to_numpy(),
        "pred_drive_level": nearest_level(drive_db, ladder),
        "true_drive_db": queries["drive_db_equivalente"].to_numpy(),
        "pred_drive_db": drive_db,
    })
    return score(out, alphabet(queries))


# --- laco ---------------------------------------------------------------------
def train(config: TrainConfig, verbose: bool = True) -> Dict[str, object]:
    """Treina (ou so calibra) uma tecnica e grava o que a torna reproduzivel."""
    import keras
    import tensorflow as tf

    if config.technique not in TECHNIQUES:
        raise KeyError(f"tecnica desconhecida: {config.technique!r}. Ha {list(TECHNIQUES)}")
    if config.deterministic:
        tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(config.seed)

    root = Path(config.dataset_root)
    frames = split_frames(root, config.arms, config.validation_recordings)
    stores = {
        name: FeatureStore(root, frame, config.feature)
        for name, frame in frames.items()
    }
    index: GridIndex = build_index(frames["train"], arms=config.arms)
    if not index.frame["file_name"].equals(stores["train"].frame["file_name"]):
        raise ValueError(
            "o indice da grade e o cache de features estao em ordens diferentes"
        )

    standardizer = PixelStandardizer.fit(stores["train"])

    # O manifesto grava a config que rodou, nao a pedida: `load_run` depende disso.
    config = replace(
        config,
        encoder=replace(config.encoder, input_shape=(*stores["train"].feature_shape, 1)),
    )
    model = build_model(config.encoder, config.technique)

    steps = 0 if config.technique == "random_encoder" else config.steps
    if config.technique == "bn_only":
        step = make_bn_step(model)
    elif config.technique == REGRESSION:
        step = make_regression_step(model, make_optimizer(config))
        lo, hi = drive_range(index.frame)
        drive = ((index.frame["drive_db_equivalente"].to_numpy() - lo) / (hi - lo))
    else:
        step = make_step(model, make_optimizer(config), config.temperature)

    rng = np.random.default_rng(config.seed)
    history: List[Dict[str, float]] = []
    checkpoints: List[Dict[str, object]] = []
    best: Dict[str, object] = {"mae_db": float("inf"), "step": 0, "weights": None}
    waited, executed = 0, 0
    started = time.time()

    for number in range(1, steps + 1):
        executed = number
        batch = class_balanced_batch(
            index, rng, config.configs_per_batch, config.views_per_config
        )
        features = tf.constant(standardizer.transform(stores["train"].take(batch.rows)))
        if config.technique == "bn_only":
            step(features)
        else:
            label = (tf.constant(drive[batch.rows][:, None], dtype=tf.float32)
                     if config.technique == REGRESSION
                     else tf.constant(batch.config, dtype=tf.int32))
            loss = float(step(features, label))
            history.append({"loss": loss, "step": number})
            if verbose and (number == 1 or number % 100 == 0):
                print(f"  passo {number:5d}/{steps}  {config.technique}={loss:.4f}", flush=True)

        if config.eval_every and number % config.eval_every == 0:
            validation = validation_error(model, frames, stores, standardizer,
                                          config.embed_batch)
            checkpoints.append({"step": number, **validation})
            if verbose:
                print(f"  [validacao no passo {number}] "
                      f"drive_exact={validation['drive_exact']:.4f} "
                      f"mae_db={validation['mae_db']:.3f}", flush=True)
            if validation["mae_db"] < best["mae_db"]:  # type: ignore[operator]
                best = {"mae_db": validation["mae_db"], "step": number,
                        "weights": model.get_weights()}
                waited = 0
            else:
                waited += 1
                if config.patience and waited >= config.patience:
                    if verbose:
                        print(f"  parada: {waited} avaliacoes sem descer desde o "
                              f"passo {best['step']}", flush=True)
                    break

    if best["weights"] is not None:
        model.set_weights(best["weights"])

    if config.evaluate_at_end:
        predictions, metrics = evaluate(model, frames, stores, standardizer,
                                        config.embed_batch, split="validacao")
    else:
        predictions, metrics = None, None
    elapsed = time.time() - started

    out_dir = config.resolved_output()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / WEIGHTS_FILE).parent.mkdir(exist_ok=True)
    model.save_weights(out_dir / WEIGHTS_FILE)
    standardizer.save(out_dir / "standardizer.npz")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if metrics is not None:
        predictions.to_csv(out_dir / predictions_file("validacao"), index=False)
        (out_dir / metrics_file("validacao")).write_text(json.dumps(metrics, indent=2),
                                                         encoding="utf-8")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "elapsed_seconds": round(elapsed, 1),
        "config": config.as_dict(),
        "splits": {name: int(len(frame)) for name, frame in frames.items()},
        "steps_executed": executed,
        "best_step": best["step"] or executed,
        "best_val_mae_db": best["mae_db"] if best["weights"] is not None else None,
        "checkpoints": checkpoints,
        # Da validacao; o teste vai para `metrics_teste.json`, no fim.
        "summary": summarize(metrics) if metrics is not None else None,
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(f"  gravado em {out_dir}")
    return manifest


def summarize(metrics: Dict[str, object]) -> Dict[str, float]:
    """Os numeros da escada: a tarefa e o controle sem travessia de implementacao."""
    overall = metrics["overall"]  # type: ignore[index]
    same = metrics["same_arm_control"]  # type: ignore[index]
    out = {
        "drive_exact": float(overall["drive_level"]["exact"]),
        "mae_db": float(overall["mae_db"]),
        "same_arm_drive_exact": float(same["drive_level"]["exact"]),
    }
    if "direct_readout" in metrics:  # so o regressor
        direct = metrics["direct_readout"]  # type: ignore[index]
        out["direct_drive_exact"] = float(direct["drive_level"]["exact"])
        out["direct_mae_db"] = float(direct["mae_db"])
    return out


def load_baselines(baselines_dir: Path, split: str = "validacao") -> List[Dict[str, object]]:
    """Uma linha por baseline gravado, mais o acaso de drive tirado do alfabeto dele."""
    rows: List[Dict[str, object]] = []
    for name, stem in BASELINES.items():
        path = Path(baselines_dir) / f"{stem}_{split}.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["overall"]
        if not rows:
            rows.append({"run": "chance", "technique": "baseline",
                         "drive_exact": 1.0 / metrics["alphabet"]["drive_level"]})
        rows.append({"run": name, "technique": "baseline",
                     "drive_exact": float(overall["drive_level"]["exact"]),
                     "mae_db": float(overall["mae_db"])})
    return rows


def compare(
    results_dir: Path = RESULTS_ROOT,
    runs: Optional[Sequence[str]] = None,
    baselines_dir: Optional[Path] = None,
    split: str = "validacao",
) -> pd.DataFrame:
    """A escada numa particao: baselines por cima, depois uma linha por execucao.

    Sem `runs`, pega toda subpasta com `run.json`, em ordem alfabetica; entram as
    que ja foram avaliadas em `split`. Os baselines saem de `baselines_dir`, por
    padrao a pasta acima de `results_dir`.
    """
    results_dir = Path(results_dir)
    if runs is None:
        runs = sorted(path.parent.name for path in results_dir.glob("*/run.json"))
    rows = load_baselines(baselines_dir or results_dir.parent, split)
    for name in runs:
        manifest_path = results_dir / name / "run.json"
        metrics_path = results_dir / name / metrics_file(split)
        if not (manifest_path.exists() and metrics_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "run": name,
            "technique": manifest["config"]["technique"],
            "seed": manifest["config"]["seed"],
            "time_pool": manifest["config"].get("encoder", {}).get("time_pool", "mean"),
            **summarize(metrics),
            "steps": manifest["steps_executed"],
            "best_step": manifest.get("best_step"),
        })
    return pd.DataFrame(rows, columns=None if rows else ["run", "technique"])


def evaluate_runs(
    results_dir: Path = RESULTS_ROOT,
    runs: Optional[Sequence[str]] = None,
    split: str = "teste",
    dataset_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Avalia execucoes gravadas numa particao, com os pesos que o treino deixou.

    E a passada final sobre o teste: grava `metrics_<split>.json` e
    `predictions_<split>.csv` em cada execucao e devolve a escada dessa particao.
    Le cada execucao com os arms e a validacao com que ela treinou.
    """
    from gefx.disent.probes import load_run

    results_dir = Path(results_dir)
    if runs is None:
        runs = sorted(path.parent.name for path in results_dir.glob("*/run.json"))
    for name in runs:
        run_dir = results_dir / name
        model, manifest = load_run(run_dir)
        run_config = manifest["config"]
        root = Path(dataset_root or run_config["dataset_root"])
        frames = split_frames(root, run_config.get("arms"),
                              run_config.get("validation_recordings", VALIDATION_RECORDINGS))
        stores = {key: FeatureStore(root, frames[key], run_config["feature"])
                  for key in EVAL_SPLITS[split]}
        standardizer = PixelStandardizer.load(run_dir / "standardizer.npz")
        predictions, metrics = evaluate(model, frames, stores, standardizer,
                                        run_config.get("embed_batch", 128), split=split)
        predictions.to_csv(run_dir / predictions_file(split), index=False)
        metrics["git_revision"] = git_revision()
        (run_dir / metrics_file(split)).write_text(json.dumps(metrics, indent=2),
                                                  encoding="utf-8")
    return compare(results_dir, runs, split=split)
