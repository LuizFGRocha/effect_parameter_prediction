"""Treino do encoder e os dois controles que separam aprendizado de arquitetura.

- `supcon`: o encoder, treinado com o SupCon (`losses.py`) sobre `z_e`;
- `random_encoder`: a mesma rede sem nenhum passo, o que a arquitetura ja entrega;
- `bn_only`: a rede sem treino com so as estatisticas moveis da BatchNorm
  calibradas, sem gradiente. Com zero passos elas ficam na inicializacao (0, 1),
  sem relacao com as ativacoes reais, o que deixa o `random_encoder` em
  desvantagem artificial.

Os tres rodam pelo mesmo codigo, arquitetura, amostrador e semente.
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
from gefx.disent.model import EffectModel, EncoderConfig
from gefx.disent.retrieval import retrieve_by_arm
from gefx.disent.sampler import GridIndex, build_index, class_balanced_batch
from gefx.disent.sidecar import split_frames

TECHNIQUES: Tuple[str, ...] = ("random_encoder", "bn_only", "supcon")

RESULTS_ROOT = Path("results/disent/v2/encoder")

#: Arquivos de `gefx disent retrieve`, ao lado de `RESULTS_ROOT`.
BASELINES: Dict[str, str] = {"B0": "b0.json", "B1": "b1.json"}


@dataclass
class TrainConfig:
    """Tudo o que define uma execucao. Vai inteiro para o `run.json`."""

    dataset_root: Path = Path("datasets/disent_v2")
    feature: str = "Spec"
    technique: str = "supcon"
    arms: Optional[Tuple[str, ...]] = None
    steps: int = 4000
    configs_per_batch: int = 8
    views_per_config: int = 8
    learning_rate: float = 1e-3
    temperature: float = DEFAULT_TEMPERATURE
    eval_every: int = 500
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
def make_step(model: EffectModel, optimizer, temperature: float):
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


def make_bn_step(model: EffectModel):
    """Passo para frente em modo de treino: so as medias moveis da BatchNorm mudam."""
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def step(x):
        model.encode(x, training=True)

    return step


# --- avaliacao ----------------------------------------------------------------
def embed(model, store: FeatureStore, standardizer: PixelStandardizer,
          batch: int = 128) -> np.ndarray:
    """`z_e` de todas as linhas do `store`, na ordem do `frame`."""
    out = np.empty((len(store), model.encoder_config.effect_dim), dtype=np.float32)
    for block, features in store.stream(np.arange(len(store)), chunk=batch):
        out[block] = np.asarray(
            model.encode(standardizer.transform(features), training=False))
    return out


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
        same_arm=False,
    )
    same = retrieve_by_arm(
        frames["query"], frames["catalog"], z_query, z_catalog,
        same_arm=True,
    )
    metrics: Dict[str, object] = dict(cross.metrics)
    metrics["same_arm_control"] = same.metrics["overall"]
    return cross.predictions, metrics


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
    frames = split_frames(root, config.arms)
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
    model = EffectModel(config.encoder)

    steps = 0 if config.technique == "random_encoder" else config.steps
    if config.technique == "bn_only":
        step = make_bn_step(model)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
        step = make_step(model, optimizer, config.temperature)

    rng = np.random.default_rng(config.seed)
    history: List[Dict[str, float]] = []
    checkpoints: List[Dict[str, object]] = []
    started = time.time()

    for number in range(1, steps + 1):
        batch = class_balanced_batch(
            index, rng, config.configs_per_batch, config.views_per_config
        )
        features = tf.constant(standardizer.transform(stores["train"].take(batch.rows)))
        if config.technique == "bn_only":
            step(features)
        else:
            loss = float(step(features, tf.constant(batch.config, dtype=tf.int32)))
            history.append({"loss": loss, "step": number})
            if verbose and (number == 1 or number % 100 == 0):
                print(f"  passo {number:5d}/{steps}  {config.technique}={loss:.4f}", flush=True)

        if config.eval_every and number % config.eval_every == 0 and number < steps:
            _, partial = evaluate(model, frames, stores, standardizer, config.embed_batch)
            overall = partial["overall"]  # type: ignore[index]
            checkpoints.append({"step": number,
                                "drive_exact": overall["drive_level"]["exact"],
                                "mae_db": overall["mae_db"]})
            if verbose:
                print(f"  [avaliacao no passo {number}] "
                      f"drive_exact={overall['drive_level']['exact']:.4f} "
                      f"mae_db={overall['mae_db']:.2f}", flush=True)

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
        "splits": {name: int(len(frame)) for name, frame in frames.items()},
        "steps_executed": steps,
        "checkpoints": checkpoints,
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
    return {
        "drive_exact": float(overall["drive_level"]["exact"]),
        "mae_db": float(overall["mae_db"]),
        "same_arm_drive_exact": float(same["drive_level"]["exact"]),
    }


def load_baselines(baselines_dir: Path) -> List[Dict[str, object]]:
    """Uma linha por baseline gravado, mais o acaso de drive tirado do alfabeto dele."""
    rows: List[Dict[str, object]] = []
    for name, filename in BASELINES.items():
        path = Path(baselines_dir) / filename
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
) -> pd.DataFrame:
    """A escada: baselines por cima, depois uma linha por execucao gravada.

    Sem `runs`, pega toda subpasta com `run.json` avaliado, em ordem alfabetica. Os
    baselines saem de `baselines_dir`, por padrao a pasta acima de `results_dir`.
    """
    results_dir = Path(results_dir)
    if runs is None:
        runs = sorted(path.parent.name for path in results_dir.glob("*/run.json"))
    rows = load_baselines(baselines_dir or results_dir.parent)
    for name in runs:
        manifest_path = results_dir / name / "run.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("summary"):  # `evaluate_at_end=False`
            continue
        rows.append({
            "run": name,
            "technique": manifest["config"]["technique"],
            "seed": manifest["config"]["seed"],
            **manifest["summary"],
            "steps": manifest["steps_executed"],
        })
    return pd.DataFrame(rows, columns=None if rows else ["run", "technique"])
