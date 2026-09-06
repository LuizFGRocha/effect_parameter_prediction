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
        seed=args.seed,
        use_full_audio=args.use_full_audio or None,
        segment_seconds=args.segment_seconds,
        ignore_start_seconds=args.ignore_start_seconds,
        ignore_end_seconds=args.ignore_end_seconds,
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

    evaluate_run(args.results_root)


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
    render_dataset.add_argument("--seed", type=int, default=None)
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
                              help="Seed global (numpy/random/tensorflow).")
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
    cross_render.set_defaults(func=_cmd_cross_impl_render)

    cross_eval = cross_sub.add_parser("eval", help="Roda os modelos treinados em cada braco.")
    cross_eval.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    cross_eval.add_argument("--models-root", default="results/second_main_run")
    cross_eval.add_argument("--output-root", default="datasets/cross_impl")
    cross_eval.set_defaults(func=_cmd_cross_impl_eval)

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
