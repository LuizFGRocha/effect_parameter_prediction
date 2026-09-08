"""Sondas lineares sobre os codigos aprendidos.

Existe porque a perda do adversario **nao e evidencia de remocao**. No estudo da
etapa 5 o classificador de implementacao ficou no acaso (ln 7 = 1,946) do
primeiro passo ao ultimo, e mesmo assim uma regressao logistica ajustada depois
sobre o mesmo `z_e` le a implementacao a 25,7% contra 14,3% de acaso. Um
adversario no acaso significa que ele parou de achar o sinal, nao que o sinal
saiu -- e o resultado conhecido de Elazar & Goldberg (2018) sobre remocao
adversaria.

E a sonda que decide entre as duas leituras, e por isso ela mora no codigo e nao
num script: o numero entra no relatorio.

Este modulo e o comeco da etapa 6. DCI e MIG entram aqui depois; a interface
(`probe_report` sobre um diretorio de execucao) ja e a que eles vao usar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: Fatores sondados e o que cada resposta significa. `arm` e `content_id` sao os
#: que o desemaranhamento promete remover de `z_e`; `drive_level` e o que ele
#: promete manter -- sondar so os dois primeiros mediria metade da afirmacao.
PROBE_FACTORS: Dict[str, str] = {
    "arm": "implementacao (deve sair de z_e)",
    "content_id": "conteudo (deve sair de z_e)",
    "drive_level": "configuracao (deve ficar em z_e)",
}

DEFAULT_FOLDS = 3


def linear_probes(
    codes: np.ndarray,
    frame: pd.DataFrame,
    factors: Sequence[str] = tuple(PROBE_FACTORS),
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Acerto de uma regressao logistica por fator, com o acaso ao lado.

    Linear de proposito: a pergunta e se o fator esta **legivel** no codigo, e
    uma sonda nao-linear responderia outra coisa (se o fator e recuperavel por
    algum modelo, o que quase sempre e verdade). Validacao cruzada porque o
    numero de amostras por classe e pequeno -- 20 conteudos de catalogo.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    codes = np.asarray(codes, dtype=np.float64)
    if len(codes) != len(frame):
        raise ValueError(f"{len(codes)} codigos e {len(frame)} linhas")

    out: Dict[str, Dict[str, float]] = {}
    for factor in factors:
        if factor not in frame.columns:
            raise KeyError(f"fator ausente no sidecar: {factor!r}")
        labels = frame[factor].astype("category").cat.codes.to_numpy()
        classes = int(len(set(labels)))
        chance = 1.0 / classes
        partition = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        accuracy = float(
            cross_val_score(
                LogisticRegression(max_iter=2000), codes, labels,
                cv=partition, n_jobs=folds,
            ).mean()
        )
        out[factor] = {
            "accuracy": accuracy,
            "chance": chance,
            "classes": classes,
            # Quanto do caminho entre o acaso e o acerto total foi percorrido.
            # Comparar 25,7% em 7 classes com 16,5% em 20 nao diria nada sem isso.
            "above_chance": (accuracy - chance) / (1.0 - chance),
        }
    return out


def probe_run(
    run_dir: Path,
    dataset_root: Path = Path("datasets/disent"),
    split: str = "catalog",
    feature: str = "Spec",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> Dict[str, object]:
    """Sonda o `z_e` de uma execucao ja treinada, a partir do que ela gravou."""
    from gefx.disent.features import FeatureStore, PixelStandardizer
    from gefx.disent.model import BetaVAE, DisentModel, EncoderConfig, HeadConfig
    from gefx.disent.train import embed, split_frames

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    encoder_config = EncoderConfig.from_dict(manifest["config"]["encoder"])
    # O controle beta-VAE tem outro encoder (a posterior sai com o dobro da
    # largura). Sondar os dois pelo mesmo caminho e o ponto: o que se compara e
    # o `z_e` que cada um oferece a busca, e nao a arquitetura que o produziu.
    if manifest["config"]["technique"] == "beta_vae":
        model = BetaVAE(encoder_config, beta=manifest["config"].get("beta", 4.0))
    else:
        model = DisentModel(encoder_config, HeadConfig(**manifest["heads"]))
    model.load_weights(run_dir / "weights")

    frame = split_frames(dataset_root, manifest["config"].get("arms"))[split]
    store = FeatureStore(dataset_root, frame, feature)
    codes = embed(model, store, PixelStandardizer.load(run_dir / "standardizer.npz"))
    return {
        "technique": manifest["config"]["technique"],
        "split": split,
        "n": int(len(frame)),
        "probes": linear_probes(codes, frame, folds=folds, seed=seed),
    }


def probe_study(
    results_dir: Path = Path("results/disent/etapa5"),
    dataset_root: Path = Path("datasets/disent"),
    techniques: Optional[Sequence[str]] = None,
    split: str = "catalog",
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Uma linha por (tecnica, fator). Escreve nada; quem grava e quem chama."""
    from gefx.disent.train import STUDY_ORDER

    results_dir = Path(results_dir)
    wanted = list(techniques) if techniques else list(STUDY_ORDER)
    rows: List[Dict[str, object]] = []
    for name in wanted:
        run_dir = results_dir / name
        if not (run_dir / "run.json").exists():
            continue
        report = probe_run(run_dir, dataset_root, split=split, folds=folds, seed=seed)
        for factor, numbers in report["probes"].items():  # type: ignore[union-attr]
            rows.append({"technique": name, "factor": factor,
                         "meaning": PROBE_FACTORS.get(factor, ""), **numbers})
    if not rows:
        raise FileNotFoundError(f"nenhuma execucao com run.json em {results_dir}")
    return pd.DataFrame(rows)
