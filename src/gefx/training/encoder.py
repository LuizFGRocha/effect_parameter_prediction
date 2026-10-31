"""Treino do encoder do POC II e dos controles que separam aprendizado de arquitetura.

- `supcon`: o encoder, treinado com o SupCon (`losses.py`) sobre `z_e`; com
  `architecture.content_dim`, tambem sobre `z_c`, rotulado pelo trecho de
  conteudo, e a perda e a soma das duas, com pesos iguais (nas CSN a perda de
  cada nocao de similaridade entra na mesma media do batch, sem peso proprio). Os positivos de cada
  cabeca dividem so o seu fator, e e isso que deixa cada uma invariante ao
  outro: nao ha adversario nem ortogonalidade, que no estudo antigo nao tiraram
  a implementacao. Precisa do batch por trecho
  (`batch_by="content"`), o unico com positivos de conteudo;
- `regressao`: a mesma CNN com uma saida so, o drive em dB normalizado, por MSE
  (o regressor do POC I). Controla se `z_e` precisa de mais de uma dimensao
  para o nivel;
- `random_encoder`: a mesma rede sem nenhum passo, o que a arquitetura ja entrega;
- `bn_only`: a rede sem treino com so as estatisticas moveis da BatchNorm
  calibradas: o mesmo treino com taxa zero, em que os pesos nao saem do lugar.
  Com zero passos elas ficam na inicializacao (0, 1), sem relacao com as
  ativacoes reais, o que deixa o `random_encoder` em desvantagem artificial.

Os quatro rodam pelo mesmo codigo, arquitetura, amostrador e semente, e pelo
`model.fit` do Keras, como o POC I.

Protocolo: `steps` passos em epocas de `eval_every` passos. No fim de cada uma, o
erro de validacao (a mesma busca do teste, em dB, sobre `val_query` e
`val_catalog`) entra nos logs, e o `EarlyStopping(restore_best_weights=True)` do
Keras, sem paciencia, devolve no fim os pesos do minimo (Goodfellow et al. 2016,
alg. 7.1). O treino nao para antes: como no SupCon oficial, o cosseno roda
inteiro. A parada antecipada (16 avaliacoes sem descer) cortava a taxa antes do
fim e saiu.

A taxa de aprendizado decai em cosseno ao longo de `steps`, de
`architecture.learning_rate` a `learning_rate * lr_min_fraction`, como no SupCon
oficial (`--cosine` em HobbitLong/SupContrast: ate `lr * 0.1 ** 3`). E o
`keras.optimizers.schedules.CosineDecay`. O POC I tinha taxa fixa; com ela a
validacao oscilava o bastante para o minimo ser um vale isolado.

Velocidade: o modelo e compilado pelo XLA (`jit_compile`), com a mesma conta em
float32 -- so a fusao das operacoes e o algoritmo de convolucao mudam, e com e
sem XLA deram o mesmo resultado (CLAUDE.md). Fica desligado com `deterministic`: o gradiente do MaxPool no XLA nao tem versao
deterministica. Os lotes sao sorteados todos antes, na ordem, e lidos do disco
numa thread enquanto a GPU roda o anterior (`FeatureStore.batches`).

O treino so le a validacao: as metricas que grava (`metrics_validacao.json`) sao
dela. O teste fica para `evaluation.runner.evaluate_runs(split="teste")`, uma
passada no fim com os modelos finais.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from gefx.config import ArchitectureConfig, git_revision, set_global_seeds
from gefx.data.dataset import FeatureStore
from gefx.data.grid import DATASET_ROOT, VALIDATION_RECORDINGS, split_frames, validate_pairing
from gefx.evaluation.retrieval import (
    drive_range,
    evaluate,
    metrics_file,
    predictions_file,
    search,
)
from gefx.training.architecture import (
    ARCHITECTURES,
    REGRESSION,
    TECHNIQUES,
    build_network,
    head,
)
from gefx.training.losses import DEFAULT_TEMPERATURE, sup_con_loss
from gefx.training.sampler import build_index, class_balanced_batch
from gefx.training.scaling import PixelStandardizer

RESULTS_ROOT = Path("results/disent/v2/validacao/encoder")

#: Onde cada execucao grava os pesos, relativo a pasta dela.
WEIGHTS_FILE = Path("weights") / "encoder.weights.h5"
STANDARDIZER_FILE = "standardizer.npz"


@dataclass
class EncoderTrainConfig:
    """Tudo o que define uma execucao. Vai inteiro para o `run.json`."""

    dataset_root: Path = DATASET_ROOT
    technique: str = "supcon"
    arms: Optional[Tuple[str, ...]] = None
    #: O teto, que e tambem o comprimento do cosseno (a escolha esta no CLAUDE.md).
    steps: int = 20000
    #: O P x K: P classes, K vistas de cada.
    classes_per_batch: int = 8
    views_per_class: int = 8
    #: A classe do P x K: `level` (o nivel de drive) ou `content` (o trecho, em
    #: `views_per_class` niveis; o que a cabeca de conteudo exige).
    batch_by: str = "level"
    #: O fim do cosseno, em fracao da taxa inicial: `0.1 ** 3`, como no original.
    lr_min_fraction: float = 1e-3
    temperature: float = DEFAULT_TEMPERATURE
    #: Passos por epoca: a validacao roda no fim de cada uma. 0 desliga, e ficam
    #: os pesos do fim.
    eval_every: int = 500
    #: Gravacoes de cada lado da validacao, tiradas do treino.
    validation_recordings: int = VALIDATION_RECORDINGS
    embed_batch: int = 128
    seed: int = 20260908
    #: Sem isto a semente so fixa a inicializacao, e duas execucoes divergem. Desliga
    #: o XLA.
    deterministic: bool = False
    #: Desligar quando quem mede e outra coisa (curva de diversidade).
    evaluate_at_end: bool = True
    output_dir: Optional[Path] = None
    architecture: ArchitectureConfig = field(default_factory=lambda: ARCHITECTURES["poc2"])

    def resolved_output(self) -> Path:
        return Path(self.output_dir or RESULTS_ROOT / self.technique)

    def as_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["dataset_root"] = str(self.dataset_root)
        out["output_dir"] = str(self.resolved_output())
        out["arms"] = list(self.arms) if self.arms else None
        out["architecture"]["kernel_size"] = list(self.architecture.kernel_size)
        return out


def run_name(technique: str, seed: int) -> str:
    """A pasta de uma execucao: `<tecnica>` na semente padrao, `<tecnica>_s<semente>`
    nas outras."""
    return technique if seed == EncoderTrainConfig.seed else f"{technique}_s{seed}"


def learning_rate(config: EncoderTrainConfig):
    """O `CosineDecay` do Keras ao longo de `steps`, como o `--cosine` do SupCon
    oficial; zero no `bn_only`, em que os pesos nao podem sair do lugar."""
    import keras

    if config.technique == "bn_only":
        return 0.0
    return keras.optimizers.schedules.CosineDecay(
        config.architecture.learning_rate, decay_steps=max(config.steps, 1),
        alpha=config.lr_min_fraction)


def supcon_loss(temperature: float):
    """O SupCon como perda do Keras: sobre `z_e` rotulado pelo nivel de drive, ou
    sobre `z_c` rotulado pelo trecho de conteudo."""

    def loss(label, code):
        import tensorflow as tf

        # Uma vista por linha: [bsz, n_views=1, dim], como o SupConLoss pede.
        return sup_con_loss(code[:, None, :], tf.reshape(label, [-1]), temperature)

    return loss


def validation_error(model, frames, stores, standardizer, batch: int = 128) -> Dict[str, float]:
    """A busca do teste sobre a validacao: o erro em dB que escolhe os pesos.

    Entre implementacoes, como o teste; com um arm so no treino (a curva de
    diversidade), dentro dele, que e a unica busca possivel sem olhar o retirado.
    """
    same_arm = frames["val_catalog"]["arm"].nunique() < 2
    (result,), _ = search(model, frames, stores, standardizer, "validacao", batch,
                          catalog_arms=("mesmo" if same_arm else "outros",))
    overall = result.metrics["overall"]
    return {"mae_db": float(overall["mae_db"]),
            "drive_exact": float(overall["drive_level"]["exact"]),
            "same_arm": bool(same_arm)}


#: Padronizadores ja ajustados neste processo. O ajuste depende so das linhas de
#: treino, nao da tecnica nem da semente, e custa uma passada pelo memmap do
#: treino (~70 s): as execucoes de um mesmo comando o reaproveitam. A linha e o
#: par (arm, arquivo): o `file_name` se repete em todos os arms, entao so ele
#: daria a mesma chave a todo recorte de 7 arms do leave-one-out.
_STANDARDIZERS: Dict[Tuple[str, int], PixelStandardizer] = {}


def fitted_standardizer(store: FeatureStore, root: Path) -> PixelStandardizer:
    rows = zip(store.frame["arm"], store.frame["file_name"])
    key = (str(Path(root).resolve()), hash(tuple(rows)))
    if key not in _STANDARDIZERS:
        _STANDARDIZERS[key] = PixelStandardizer.fit(store)
    return _STANDARDIZERS[key]


def train(config: EncoderTrainConfig, verbose: bool = True) -> Dict[str, object]:
    """Treina (ou so calibra) uma tecnica e grava o que a torna reproduzivel."""
    import keras

    if config.technique not in TECHNIQUES:
        raise KeyError(f"tecnica desconhecida: {config.technique!r}. Ha {list(TECHNIQUES)}")
    if config.architecture.content_dim and config.batch_by != "content":
        # Sem positivo, a ancora entra no SupCon com perda zero: `z_c` nao
        # aprenderia nada, sem erro nenhum.
        raise ValueError("a cabeca de conteudo precisa do batch por trecho "
                         "(batch_by='content'): por nivel quase nao ha positivo de conteudo")
    set_global_seeds(config.seed, deterministic=config.deterministic)
    jit_compile = not config.deterministic
    started = time.time()

    root = Path(config.dataset_root)
    validate_pairing(root)
    frames = split_frames(root, config.arms, config.validation_recordings)
    stores = {name: FeatureStore(root, frame) for name, frame in frames.items()}
    index = build_index(frames["train"], arms=config.arms)
    if not index.frame["file_name"].equals(stores["train"].frame["file_name"]):
        raise ValueError("o indice da grade e o cache de features estao em ordens diferentes")
    standardizer = fitted_standardizer(stores["train"], root)
    input_dim = (*stores["train"].feature_shape, 1)
    model = build_network(input_dim, config.architecture, config.technique)
    # A busca de validacao e a avaliacao leem so `z_e`, com os mesmos pesos.
    effect = head(model, "z_e")

    steps = 0 if config.technique == "random_encoder" else config.steps
    if config.technique == REGRESSION:
        lo, hi = drive_range(index.frame)
        labels = ((index.frame["drive_db_equivalente"].to_numpy() - lo) / (hi - lo))
        labels = labels.astype(np.float32)[:, None]
        loss = "mean_squared_error"
    elif config.architecture.content_dim:
        labels = (index.level_label, index.content_label)
        loss = [supcon_loss(config.temperature), supcon_loss(config.temperature)]
    else:
        labels, loss = index.level_label, supcon_loss(config.temperature)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate(config)),
                  loss=loss, jit_compile=jit_compile)

    checkpoints = []
    history: Dict[str, list] = {}
    stopper = None
    if steps:
        epoch_steps = config.eval_every or steps
        if steps % epoch_steps:
            raise ValueError(f"steps={steps} nao e multiplo de eval_every={epoch_steps}")
        # Os lotes saem todos do `rng` aqui, em ordem; so a leitura vai para a thread.
        rng = np.random.default_rng(config.seed)
        blocks = [class_balanced_batch(index, rng, config.classes_per_batch,
                                       config.views_per_class, config.batch_by)
                  for _ in range(steps)]
        callbacks = []
        if config.eval_every:
            def on_epoch_end(epoch, logs):
                validation = validation_error(effect, frames, stores, standardizer,
                                              config.embed_batch)
                logs["val_mae_db"] = validation["mae_db"]
                logs["val_drive_exact"] = validation["drive_exact"]
                checkpoints.append({"step": (epoch + 1) * epoch_steps, **validation})
                if verbose:
                    print(f"  passo {(epoch + 1) * epoch_steps:5d}/{steps}  "
                          f"{config.technique}={logs['loss']:.4f}  validacao: "
                          f"drive_exact={validation['drive_exact']:.4f} "
                          f"mae_db={validation['mae_db']:.3f}", flush=True)

            stopper = keras.callbacks.EarlyStopping(
                monitor="val_mae_db", mode="min", patience=steps, restore_best_weights=True)
            callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end), stopper]
        fitted = model.fit(stores["train"].batches(blocks, standardizer, labels),
                           epochs=steps // epoch_steps, steps_per_epoch=epoch_steps,
                           shuffle=False, callbacks=callbacks, verbose=0)
        history = {key: [float(value) for value in values]
                   for key, values in fitted.history.items()}

    if config.evaluate_at_end:
        predictions, metrics = evaluate(effect, frames, stores, standardizer,
                                        config.embed_batch, split="validacao")
    else:
        predictions, metrics = None, None
    elapsed = time.time() - started

    out_dir = config.resolved_output()
    (out_dir / WEIGHTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(out_dir / WEIGHTS_FILE)
    standardizer.save(out_dir / STANDARDIZER_FILE)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if metrics is not None:
        predictions.to_csv(out_dir / predictions_file("validacao"), index=False)
        (out_dir / metrics_file("validacao")).write_text(json.dumps(metrics, indent=2),
                                                         encoding="utf-8")

    best_found = stopper is not None and stopper.best_weights is not None
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {**config.as_dict(), "jit_compile": jit_compile},
        "input_shape": list(input_dim),
        "splits": {name: int(len(frame)) for name, frame in frames.items()},
        "steps_executed": steps,
        "best_step": (stopper.best_epoch + 1) * config.eval_every if best_found else steps,
        "best_val_mae_db": float(stopper.best) if best_found else None,
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
