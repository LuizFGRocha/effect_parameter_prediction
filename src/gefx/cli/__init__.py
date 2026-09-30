"""CLI unica do projeto: `gefx <subcomando>`.

Os imports pesados (TensorFlow, pedalboard) ficam dentro de cada handler para o
`--help` nao pagar o custo de carrega-los.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from gefx.config import FEATURE_CHOICES

CONFIG_HELP = "Arquivo YAML de experimento. Flags explicitas sobrescrevem o que vier dele."


# --- handlers ----------------------------------------------------------------
def _cmd_render_dataset(args: argparse.Namespace) -> None:
    from gefx.config import load_config
    from gefx.data.render import generate_dataset

    config = load_config(
        args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_file=args.samples_per_file,
        render_seed=args.seed,
        use_full_audio=args.use_full_audio or None,
        segment_seconds=args.segment_seconds,
        ignore_start_seconds=args.ignore_start_seconds,
        ignore_end_seconds=args.ignore_end_seconds,
        legacy=True if args.legacy else (False if args.no_legacy else None),
    )
    generate_dataset(config.dataset)


def _cmd_train(args: argparse.Namespace) -> None:
    from gefx.config import load_config
    from gefx.training.trainer import train

    config = load_config(
        args.config,
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        feature=args.feature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        seed=args.seed,
        chain_key=args.chain_key,
        rebuild_cache=args.rebuild_cache or None,
        clear_session=False if args.no_clear_session else None,
        deterministic=args.deterministic or None,
        n_conv=args.n_conv,
        n_full=args.n_full,
        n_nodes=args.n_nodes,
        n_filters=args.n_filters,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    train(config)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from gefx.evaluation.runner import evaluate_run

    evaluate_run(args.results_root, legacy=args.legacy)


def _cmd_compare_features(args: argparse.Namespace) -> None:
    from gefx.evaluation.compare import compare_features

    compare_features(args.results_base, args.output_csv, args.output_plot)


def _cmd_cross_impl_render(args: argparse.Namespace) -> None:
    from gefx.crossimpl.render import RenderOptions, render

    render(
        RenderOptions(
            input_dir=args.input_dir,
            output_root=args.output_root,
            n=args.n,
            segment_seconds=args.segment_seconds,
            seed=args.seed,
            arms=args.arm,
            effects=args.effect,
            legacy=args.legacy,
        )
    )


def _cmd_cross_impl_eval(args: argparse.Namespace) -> None:
    from gefx.crossimpl.evaluate import EvalOptions, evaluate

    evaluate(
        EvalOptions(
            feature=args.feature,
            models_root=args.models_root,
            output_root=args.output_root,
        )
    )


def _cmd_disent_calibrate(args: argparse.Namespace) -> None:
    from gefx.disent.calibrate import calibrate

    calibrate(
        roster_path=Path(args.roster),
        levels_path=Path(args.levels),
        input_dir=Path(args.input_dir),
        curves_csv=Path(args.curves),
        points=args.points,
        n_segments=args.segments,
        workers=args.workers,
        reference_db=tuple(args.ref_db),
        n_levels=args.n_levels,
    )


def _cmd_disent_tune(args: argparse.Namespace) -> None:
    from gefx.disent.tune import Tuner, serve

    serve(Tuner(Path(args.recording), Path(args.roster), Path(args.levels)),
          port=args.port, open_browser=not args.no_browser)


def _cmd_disent_render(args: argparse.Namespace) -> None:
    from gefx.disent.render import RenderOptions, render

    render(
        RenderOptions(
            input_dir=Path(args.input_dir),
            output_root=Path(args.output_root),
            roster=Path(args.roster),
            levels=Path(args.levels),
            n_contents=args.contents,
            segments_per_file=args.segments_per_file,
            segment_seconds=args.segment_seconds,
            seed=args.seed,
            split_seed=args.split_seed,
            arms=args.arm,
            splits=args.split,
            workers=args.workers,
        )
    )


def _cmd_disent_between(args: argparse.Namespace) -> None:
    from gefx.disent.between import between_study
    from gefx.disent.calibrate import between_levels

    if args.write_levels:
        between_levels(Path(args.levels), Path(args.curves), Path(args.write_levels))
        print(f"niveis intermediarios em {args.write_levels}")
        return
    print(between_study(Path(args.results_dir), Path(args.output_root), Path(args.between_root),
                        runs=args.run or ("supcon",), k=args.k).to_string(index=False))


def _cmd_disent_cache(args: argparse.Namespace) -> None:
    from gefx.disent.features import build_all_caches

    build_all_caches(Path(args.output_root), args.feature, arms=args.arm, rebuild=args.rebuild)


def _cmd_disent_retrieve(args: argparse.Namespace) -> None:
    import json

    from gefx.disent.retrieval import baseline_b0, baseline_b1

    root = Path(args.output_root)
    if args.baseline == "b0":
        result = baseline_b0(root, arms=args.arm)
        rotulo = "B0 -- vizinho mais proximo no Spec padronizado"
    else:
        result = baseline_b1(root, arms=args.arm)
        rotulo = "B1 -- regressor do POC I, sem retreino"
    overall = result.metrics["overall"]
    print(f"{rotulo}, entre implementacoes, {overall['n']} consultas")
    item = overall["drive_level"]
    print(f"  {'drive_level':12s} exato {item['exact']:.1%} (acaso {item['chance']:.1%})  "
          f"+-1 {item['within_one']:.1%}  MAE {item['mae_levels']:.2f} niveis")
    print(f"  {'drive':12s} MAE {overall['mae_db']:.2f} dB equivalentes")
    print("\npor arm de consulta (drive exato):")
    for arm, item in sorted(result.metrics["per_query_arm"].items()):
        print(f"  {arm:16s} {item['drive_level']['exact']:.1%}  "
              f"MAE {item['drive_level']['mae_levels']:.2f}  "
              f"{item['mae_db']:.2f} dB")

    destino = Path(args.results_dir) / args.baseline
    destino.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(destino.with_suffix(".csv"), index=False)
    destino.with_suffix(".json").write_text(json.dumps(result.metrics, indent=2),
                                            encoding="utf-8")
    print(f"\nescrito em {destino}.{{csv,json}}")


def _cmd_disent_plots(args: argparse.Namespace) -> None:
    from gefx.disent.plots import build_all

    escritos = build_all(Path(args.results_dir), Path(args.out_dir) if args.out_dir else None)
    print(f"{len(escritos)} figuras:")
    for caminho in escritos:
        print(f"  {caminho}")


def _write_table(tabela, destino: Path) -> None:
    print(tabela.to_string(index=False))
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


def _cmd_disent_train(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from gefx.disent.model import ARCHITECTURES
    from gefx.disent.train import RESULTS_ROOT, TECHNIQUES, TrainConfig, compare, train

    escolhidas = list(dict.fromkeys(args.technique or ["supcon"]))
    desconhecidas = [nome for nome in escolhidas if nome not in TECHNIQUES]
    if desconhecidas:
        raise SystemExit(f"tecnica desconhecida: {desconhecidas}. Ha {list(TECHNIQUES)}")
    encoder = ARCHITECTURES[args.arch]
    if args.time_pool:
        encoder = replace(encoder, time_pool=args.time_pool)
    raiz = Path(args.results_dir)
    for nome in escolhidas:
        # A semente padrao fica em `<tecnica>/`; as outras, ao lado.
        pasta = nome if args.seed == TrainConfig.seed else f"{nome}_s{args.seed}"
        print(f"[{pasta}]")
        train(TrainConfig(
            dataset_root=Path(args.output_root),
            feature=args.feature,
            technique=nome,
            arms=tuple(args.arm) if args.arm else None,
            steps=args.steps,
            configs_per_batch=args.configs_per_batch,
            views_per_config=args.views_per_config,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            eval_every=args.eval_every,
            seed=args.seed,
            deterministic=args.deterministic,
            output_dir=raiz / pasta,
            encoder=encoder,
        ))

    print()
    _write_table(compare(raiz), raiz / "escada.csv")


def _cmd_disent_probe(args: argparse.Namespace) -> None:
    from gefx.disent.probes import probe_study

    tabela = probe_study(
        Path(args.results_dir), Path(args.output_root),
        runs=args.run, split=args.split, folds=args.folds, seed=args.seed,
    )
    _write_table(tabela, Path(args.results_dir) / "sondas.csv")


def _cmd_disent_diversity(args: argparse.Namespace) -> None:
    from gefx.disent.loo import arm_diversity_curve

    tabela = arm_diversity_curve(
        args.held_out, Path(args.output_root), Path(args.results_dir),
        steps=args.steps, seed=args.seed,
        reuse=Path(args.reuse) if args.reuse else None,
    )
    print()
    print(tabela.to_string(index=False))
    print(f"\ntabela em {Path(args.results_dir) / 'resumo.csv'}")


def _cmd_disent_loo(args: argparse.Namespace) -> None:
    from gefx.disent.loo import leave_one_arm_out, transfer_cost

    tabela = leave_one_arm_out(
        Path(args.output_root), Path(args.results_dir),
        steps=args.steps, seed=args.seed,
        seeds=[args.seed] + [s for s in (args.extra_seed or []) if s != args.seed],
    )
    custo = transfer_cost(tabela)
    print()
    print(custo.to_string(index=False))
    print(f"\ncusto medio de nunca ter visto a implementacao: "
          f"{custo.custo_pontos.mean():+.1f} pontos")
    print(custo.groupby("estrato").custo_pontos.mean().round(1).to_string())
    custo.to_csv(Path(args.results_dir) / "custo_de_transferencia.csv", index=False)


def _cmd_disent_validate(args: argparse.Namespace) -> None:
    from gefx.disent.sidecar import validate_pairing

    summary = validate_pairing(Path(args.output_root))
    print("grade pareada e completa:")
    for name, value in summary.items():
        print(f"  {name:14s} {value}")


def _cmd_inspect_plugin(args: argparse.Namespace) -> None:
    from gefx.effects.inspect import inspect_plugin

    inspect_plugin(
        args.plugin_path,
        sweep=args.sweep,
        sweep_all=args.sweep_all,
        steps=args.steps,
        json_out=args.json,
        plugin_name=args.plugin_name,
    )


# --- parser ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gefx",
        description="Recuperacao de configuracoes de efeitos de guitarra a partir do audio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # gefx render-dataset
    render_dataset = sub.add_parser(
        "render-dataset", help="Renderiza um dataset a partir das gravacoes limpas."
    )
    render_dataset.add_argument("--config", default=None, help=CONFIG_HELP)
    render_dataset.add_argument("--input-dir", default=None)
    render_dataset.add_argument("--output-dir", default=None)
    render_dataset.add_argument("--samples-per-file", type=int, default=None,
                                help="Renders por arquivo-fonte por cadeia.")
    render_dataset.add_argument("--seed", type=int, default=None,
                                help="Seed da renderizacao (dataset.render_seed).")
    render_dataset.add_argument("--legacy", action="store_true",
                                help="Ordena as colunas JSON do sidecar alfabeticamente, "
                                     "como nos datasets ja renderizados.")
    render_dataset.add_argument("--no-legacy", action="store_true",
                                help="Forca a convencao nova mesmo que o YAML peca legacy.")
    render_dataset.add_argument("--use-full-audio", action="store_true",
                                help="Processa o audio inteiro em vez de um segmento aleatorio.")
    render_dataset.add_argument("--segment-seconds", type=float, default=None)
    render_dataset.add_argument("--ignore-start-seconds", type=float, default=None)
    render_dataset.add_argument("--ignore-end-seconds", type=float, default=None)
    render_dataset.set_defaults(func=_cmd_render_dataset)

    # gefx train
    train_parser = sub.add_parser("train", help="Treina um regressor por cadeia.")
    train_parser.add_argument("--config", default=None, help=CONFIG_HELP)
    train_parser.add_argument("--dataset-root", default=None)
    train_parser.add_argument("--results-root", default=None)
    train_parser.add_argument("--feature", default=None, choices=FEATURE_CHOICES)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--test-size", type=float, default=None)
    train_parser.add_argument("--split-seed", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Seed global do treino (numpy/random/tensorflow). "
                                   "Nao afeta a renderizacao: essa e dataset.render_seed.")
    train_parser.add_argument("--chain-key", default=None,
                              help="Treina so esta cadeia (ex.: distortion__chorus).")
    train_parser.add_argument("--rebuild-cache", action="store_true",
                              help="Recomputa o cache de features antes de treinar.")
    train_parser.add_argument("--no-clear-session", action="store_true",
                              help="Nao limpa a sessao do TensorFlow entre cadeias.")
    train_parser.add_argument("--deterministic", action="store_true",
                              help="Kernels deterministicos: duas runs com o mesmo seed ficam "
                                   "identicas, ao custo de velocidade.")
    arch = train_parser.add_argument_group("arquitetura")
    arch.add_argument("--n-conv", type=int, default=None)
    arch.add_argument("--n-full", type=int, default=None)
    arch.add_argument("--n-nodes", type=int, default=None)
    arch.add_argument("--n-filters", type=int, default=None)
    arch.add_argument("--dropout", type=float, default=None)
    arch.add_argument("--learning-rate", type=float, default=None)
    train_parser.set_defaults(func=_cmd_train)

    # gefx evaluate
    evaluate_parser = sub.add_parser(
        "evaluate", help="Agrega metricas e graficos de um run ja treinado."
    )
    evaluate_parser.add_argument("--results-root", required=True)
    evaluate_parser.add_argument("--legacy", action="store_true",
                                 help="Ordena os parametros alfabeticamente, como nas "
                                      "tabelas ja geradas, em vez da ordem do vetor alvo.")
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    # gefx compare-features
    compare_parser = sub.add_parser(
        "compare-features", help="Compara features a partir de <base>/<FEATURE>/chain_metrics.csv."
    )
    compare_parser.add_argument("--results-base", required=True)
    compare_parser.add_argument("--output-csv", default=None)
    compare_parser.add_argument("--output-plot", default=None)
    compare_parser.set_defaults(func=_cmd_compare_features)

    # gefx cross-impl
    cross = sub.add_parser("cross-impl", help="Estudo de generalizacao entre implementacoes.")
    cross_sub = cross.add_subparsers(dest="cross_command", required=True)

    cross_render = cross_sub.add_parser("render", help="Renderiza o conjunto pareado.")
    cross_render.add_argument("--arm", action="append", default=None, help="Repetivel. Padrao: todos.")
    cross_render.add_argument("--effect", action="append", default=None, help="Repetivel. Padrao: todos.")
    cross_render.add_argument("--n", type=int, default=300, help="Renders por (efeito, braco).")
    cross_render.add_argument("--input-dir", default="datasets/unprocessed_samples")
    cross_render.add_argument("--output-root", default="datasets/cross_impl")
    cross_render.add_argument("--segment-seconds", type=float, default=2.0)
    cross_render.add_argument("--seed", type=int, default=7)
    cross_render.add_argument("--legacy", action="store_true",
                              help="Ordena as colunas JSON do sidecar alfabeticamente.")
    cross_render.set_defaults(func=_cmd_cross_impl_render)

    cross_eval = cross_sub.add_parser("eval", help="Roda os modelos treinados em cada braco.")
    cross_eval.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    cross_eval.add_argument("--models-root", default="results/second_main_run")
    cross_eval.add_argument("--output-root", default="datasets/cross_impl")
    cross_eval.set_defaults(func=_cmd_cross_impl_eval)

    # gefx disent
    disent = sub.add_parser("disent", help="POC II: recuperacao por um codigo de efeito desemaranhado.")
    disent_sub = disent.add_subparsers(dest="disent_command", required=True)

    disent_calibrate = disent_sub.add_parser(
        "calibrate", help="Niveis de drive de todos os arms, pareados pelo Rnonlin.")
    disent_calibrate.add_argument("--roster", default="experiments/disent_roster.yaml")
    disent_calibrate.add_argument("--levels", default="experiments/disent_levels.yaml",
                                  help="Reescrito inteiro.")
    disent_calibrate.add_argument("--input-dir", default="datasets/unprocessed_samples")
    disent_calibrate.add_argument("--curves", default="results/disent/calibracao/curvas.csv")
    disent_calibrate.add_argument("--points", type=int, default=33,
                                  help="Posicoes do knob varridas por arm.")
    disent_calibrate.add_argument("--segments", type=int, default=8)
    disent_calibrate.add_argument("--workers", type=int, default=8)
    disent_calibrate.add_argument("--ref-db", type=float, nargs=2, metavar=("MIN", "MAX"),
                                  default=[18.45, 39.0],
                                  help="Faixa da referencia em dB; niveis igualmente espacados "
                                       "em Rnonlin.")
    disent_calibrate.add_argument("--n-levels", type=int, default=8)
    disent_calibrate.set_defaults(func=_cmd_disent_calibrate)

    disent_tune = disent_sub.add_parser(
        "tune", help="Pagina local para ouvir os niveis de cada arm, com A/B contra a "
                     "referencia. So escuta: nao altera os niveis.")
    disent_tune.add_argument("--recording", required=True, help="Sua gravacao de guitarra (wav).")
    disent_tune.add_argument("--roster", default="experiments/disent_roster.yaml")
    disent_tune.add_argument("--levels", default="experiments/disent_levels.yaml")
    disent_tune.add_argument("--port", type=int, default=8765)
    disent_tune.add_argument("--no-browser", action="store_true")
    disent_tune.set_defaults(func=_cmd_disent_tune)

    disent_render = disent_sub.add_parser(
        "render", help="Renderiza a grade combinatoria conteudo x configuracao x arm."
    )
    disent_render.add_argument("--arm", action="append", default=None,
                               help="Repetivel. Padrao: todos os arms do roster.")
    disent_render.add_argument("--input-dir", default="datasets/unprocessed_samples")
    disent_render.add_argument("--output-root", default="datasets/disent_v2")
    disent_render.add_argument("--roster", default="experiments/disent_roster.yaml",
                               help="Os plugins de cada arm.")
    disent_render.add_argument("--levels", default="experiments/disent_levels.yaml",
                               help="Os knobs de drive de cada nivel (so `levels` e usado).")
    disent_render.add_argument("--contents", type=int, default=400,
                               help="Gravacoes usadas; a particao e por gravacao.")
    disent_render.add_argument("--segments-per-file", type=int, default=5,
                               help="Trechos consecutivos por gravacao; cada um e um conteudo.")
    disent_render.add_argument("--segment-seconds", type=float, default=2.0)
    disent_render.add_argument("--split", action="append", default=None,
                               choices=["train", "catalog", "query"],
                               help="Repetivel. So estas particoes. Padrao: todas.")
    disent_render.add_argument("--seed", type=int, default=20260906)
    disent_render.add_argument("--split-seed", type=int, default=20260906)
    disent_render.add_argument("--workers", type=int, default=4)
    disent_render.set_defaults(func=_cmd_disent_render)

    disent_retrieve = disent_sub.add_parser(
        "retrieve", help="Baselines sem aprendizado do POC II: B0 e B1.")
    disent_retrieve.add_argument("--baseline", default="b0", choices=["b0", "b1"],
                                 help="b0: vizinho mais proximo no Spec padronizado, a "
                                      "entrada do encoder. b1: regressor do POC I, sem "
                                      "retreino.")
    disent_retrieve.add_argument("--output-root", default="datasets/disent_v2")
    disent_retrieve.add_argument("--results-dir", default="results/disent/v2",
                                 help="Grava <results-dir>/<baseline>.{csv,json}, onde "
                                      "a escada e as figuras os procuram.")
    disent_retrieve.add_argument("--arm", action="append", default=None,
                                 help="Restringe o roster. Padrao: todos.")
    disent_retrieve.set_defaults(func=_cmd_disent_retrieve)

    disent_plots = disent_sub.add_parser(
        "plots", help="Figuras dos baselines e do encoder, do que ja esta gravado.")
    disent_plots.add_argument("--results-dir", default="results/disent/v2",
                              help="Baselines na raiz, o encoder em <results-dir>/encoder.")
    disent_plots.add_argument("--out-dir", default=None,
                              help="Padrao: <results-dir>/figuras.")
    disent_plots.set_defaults(func=_cmd_disent_plots)

    disent_cache = disent_sub.add_parser(
        "cache", help="Extrai o cache de features de cada arm."
    )
    disent_cache.add_argument("--output-root", default="datasets/disent_v2")
    disent_cache.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    disent_cache.add_argument("--arm", action="append", default=None)
    disent_cache.add_argument("--rebuild", action="store_true")
    disent_cache.set_defaults(func=_cmd_disent_cache)

    disent_train = disent_sub.add_parser(
        "train",
        help="Treina o encoder (supcon) ou roda um dos controles sem treino.",
    )
    disent_train.add_argument("--output-root", default="datasets/disent_v2",
                              help="Raiz do dataset do POC II.")
    disent_train.add_argument("--results-dir", default="results/disent/v2/encoder")
    disent_train.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    disent_train.add_argument("--technique", action="append", default=None,
                              choices=["supcon", "random_encoder", "bn_only",
                                       "regressao"],
                              help="Repetivel. Padrao: supcon. random_encoder: "
                                   "zero passos. bn_only: so calibra a BatchNorm, sem "
                                   "gradiente. regressao: o mesmo tronco com uma saida "
                                   "so, o drive em dB, por MSE.")
    disent_train.add_argument("--arm", action="append", default=None,
                              help="Restringe as implementacoes.")
    disent_train.add_argument("--steps", type=int, default=4000)
    disent_train.add_argument("--configs-per-batch", type=int, default=8)
    disent_train.add_argument("--views-per-config", type=int, default=8)
    disent_train.add_argument("--learning-rate", type=float, default=1e-3)
    disent_train.add_argument("--temperature", type=float, default=0.07)
    disent_train.add_argument("--eval-every", type=int, default=500,
                              help="0 desliga a avaliacao intermediaria.")
    disent_train.add_argument("--seed", type=int, default=20260908,
                              help="Fora do padrao, grava em <tecnica>_s<semente>.")
    disent_train.add_argument("--deterministic", action="store_true",
                              help="Nucleos deterministicos do TensorFlow. Sem isto a "
                                   "semente fixa so a inicializacao e duas execucoes "
                                   "identicas divergem. Custa ~20%% de velocidade.")
    disent_train.add_argument("--arch", default="poc2", choices=["poc2", "poc1"],
                              help="poc2: o encoder do relatorio. poc1: a CNN do POC I "
                                   "(2 blocos de 6 e 12 filtros, sem media no tempo, "
                                   "duas densas de 64).")
    disent_train.add_argument("--time-pool", default=None,
                              choices=["mean", "max", "flatten"],
                              help="Reducao do eixo do tempo no tronco; sem isto, a da "
                                   "arquitetura. max e flatten sao a ablacao do poc2.")
    disent_train.set_defaults(func=_cmd_disent_train)

    disent_between = disent_sub.add_parser(
        "between",
        help="Consultas nos pontos medios entre niveis contra o catalogo da grade.",
    )
    disent_between.add_argument("--write-levels", default=None,
                                help="So grava o arquivo de niveis intermediarios, para "
                                     "`render --levels` (com --split query).")
    disent_between.add_argument("--levels", default="experiments/disent_levels.yaml")
    disent_between.add_argument("--curves", default="results/disent/calibracao/curvas.csv")
    disent_between.add_argument("--results-dir", default="results/disent/v2/encoder")
    disent_between.add_argument("--output-root", default="datasets/disent_v2",
                                help="Dataset da grade, de onde sai o catalogo.")
    disent_between.add_argument("--between-root", default="datasets/disent_v2_entre",
                                help="Dataset das consultas nos pontos medios.")
    disent_between.add_argument("--run", action="append", default=None,
                                help="Repetivel. Padrao: supcon.")
    disent_between.add_argument("--k", type=int, default=10)
    disent_between.set_defaults(func=_cmd_disent_between)

    disent_probe = disent_sub.add_parser(
        "probe",
        help="Sondas lineares sobre o z_e: que fatores o codigo ainda deixa ler.",
    )
    disent_probe.add_argument("--results-dir", default="results/disent/v2/encoder")
    disent_probe.add_argument("--output-root", default="datasets/disent_v2")
    disent_probe.add_argument("--run", action="append", default=None,
                              help="Repetivel. Padrao: toda execucao em --results-dir.")
    disent_probe.add_argument("--split", default="catalog",
                              help="Particao sondada. O catalogo e o padrao porque "
                                   "e o que a busca de fato consulta.")
    disent_probe.add_argument("--folds", type=int, default=3)
    disent_probe.add_argument("--seed", type=int, default=0)
    disent_probe.set_defaults(func=_cmd_disent_probe)

    disent_loo = disent_sub.add_parser(
        "loo", help="Leave-one-arm-out: a transferencia para implementacao inedita.")
    disent_loo.add_argument("--output-root", default="datasets/disent_v2")
    disent_loo.add_argument("--results-dir", default="results/disent/v2/encoder/loo")
    disent_loo.add_argument("--steps", type=int, default=4000)
    disent_loo.add_argument("--seed", type=int, default=20260908)
    disent_loo.add_argument("--extra-seed", type=int, action="append", default=None,
                            help="Repetivel. Repete o leave-one-out inteiro com outra "
                                 "semente; e o unico jeito de por barra no custo por "
                                 "arm, que tem so 800 consultas.")
    disent_loo.set_defaults(func=_cmd_disent_loo)

    disent_diversity = disent_sub.add_parser(
        "diversity",
        help="B2 e B3: treina com 1, 2, ... N-1 implementacoes e mede a transferencia.",
    )
    disent_diversity.add_argument("--output-root", default="datasets/disent_v2")
    disent_diversity.add_argument("--results-dir",
                                  default="results/disent/v2/encoder/diversidade")
    disent_diversity.add_argument("--held-out", required=True,
                                  help="O arm que nunca entra no treino.")
    disent_diversity.add_argument("--steps", type=int, default=4000)
    disent_diversity.add_argument("--seed", type=int, default=20260908)
    disent_diversity.add_argument("--reuse", default="results/disent/v2/encoder/loo",
                                  help="Diretorio do leave-one-out: o ultimo ponto da "
                                       "curva e a mesma execucao e nao e retreinado.")
    disent_diversity.set_defaults(func=_cmd_disent_diversity)

    disent_validate = disent_sub.add_parser(
        "validate", help="Confere que a grade esta cruzada e pareada entre os arms."
    )
    disent_validate.add_argument("--output-root", default="datasets/disent_v2")
    disent_validate.set_defaults(func=_cmd_disent_validate)

    # gefx inspect-plugin
    inspect_parser = sub.add_parser(
        "inspect-plugin", help="Lista parametros de um VST3 e varre o mapeamento knob->unidade."
    )
    inspect_parser.add_argument("plugin_path", help="Caminho do .vst3 (ou .component).")
    inspect_parser.add_argument("--sweep", action="append", default=[], metavar="PARAM",
                                help="Varre um parametro; pode repetir.")
    inspect_parser.add_argument("--sweep-all", action="store_true",
                                help="Varre todos os parametros continuos.")
    inspect_parser.add_argument("--steps", type=int, default=11, help="Pontos da varredura.")
    inspect_parser.add_argument("--json", type=Path, default=None, help="Salva o relatorio em JSON.")
    inspect_parser.add_argument("--plugin-name", default=None,
                                help="Sub-plugin a carregar, para arquivos com varios.")
    inspect_parser.set_defaults(func=_cmd_inspect_plugin)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
