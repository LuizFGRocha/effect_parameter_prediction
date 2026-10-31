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
def _given(args: argparse.Namespace, *names: str) -> dict:
    """So as flags passadas: as outras ficam no padrao de quem as recebe, que e o
    unico lugar onde o padrao esta escrito."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _dataset_root(args: argparse.Namespace) -> Path:
    from gefx.data.grid import DATASET_ROOT

    return Path(args.output_root or DATASET_ROOT)


def _results_dir(args: argparse.Namespace, *below: str) -> Path:
    """`--results-dir`, ou `<RESULTS_ROOT>/<below...>` (o encoder da validacao)."""
    from gefx.training.encoder import RESULTS_ROOT

    return Path(args.results_dir) if args.results_dir else RESULTS_ROOT.joinpath(*below)


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

    render(RenderOptions(arms=args.arm, effects=args.effect, legacy=args.legacy,
                         **_given(args, "input_dir", "output_root", "n", "segment_seconds",
                                  "seed")))


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
    from gefx.effects.levels import calibrate

    calibrate(**_given(args, "roster_path", "levels_path", "input_dir", "curves_csv", "points",
                       "n_segments", "workers", "reference_db", "n_levels"))


def _cmd_disent_render(args: argparse.Namespace) -> None:
    from gefx.data.render_grid import RenderOptions, render

    render(RenderOptions(**_given(args, "input_dir", "output_root", "roster", "levels",
                                  "n_contents", "segments_per_file", "segment_seconds", "seed",
                                  "split_seed", "arms", "splits", "workers")))


def _cmd_disent_between(args: argparse.Namespace) -> None:
    from gefx.effects.levels import between_levels
    from gefx.evaluation.between import between_study

    if args.write_levels:
        between_levels(out_path=args.write_levels, **_given(args, "levels_path", "curves_csv"))
        print(f"niveis intermediarios em {args.write_levels}")
        return
    print(between_study(_results_dir(args), _dataset_root(args), args.between_root,
                        **_given(args, "runs", "k")).to_string(index=False))


def _cmd_disent_retrieve(args: argparse.Namespace) -> None:
    import json

    from gefx.data.grid import split_file
    from gefx.evaluation.retrieval import baseline_b0, baseline_b1

    root = _dataset_root(args)
    if args.baseline == "b0":
        result = baseline_b0(root, arms=args.arm, split=args.split)
        rotulo = "B0 -- vizinho mais proximo no Spec padronizado"
    else:
        result = baseline_b1(root, arms=args.arm, split=args.split)
        rotulo = "B1 -- regressor do POC I, sem retreino"
    overall = result.metrics["overall"]
    print(f"{rotulo}, entre implementacoes, {args.split}, {overall['n']} consultas")
    item = overall["drive_level"]
    print(f"  {'drive_level':12s} exato {item['exact']:.1%} (acaso {item['chance']:.1%})  "
          f"+-1 {item['within_one']:.1%}  MAE {item['mae_levels']:.2f} niveis")
    print(f"  {'drive':12s} MAE {overall['mae_db']:.2f} dB equivalentes")
    print("\npor arm de consulta (drive exato):")
    for arm, item in sorted(result.metrics["per_query_arm"].items()):
        print(f"  {arm:16s} {item['drive_level']['exact']:.1%}  "
              f"MAE {item['drive_level']['mae_levels']:.2f}  "
              f"{item['mae_db']:.2f} dB")

    # Os baselines ficam na pasta acima das execucoes do encoder.
    raiz = Path(args.results_dir) if args.results_dir else _results_dir(args).parent
    raiz.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(raiz / split_file(args.baseline, args.split, "csv"), index=False)
    destino = raiz / split_file(args.baseline, args.split, "json")
    destino.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    print(f"\nescrito em {destino.with_suffix('')}.{{csv,json}}")


def _cmd_disent_plots(args: argparse.Namespace) -> None:
    from gefx.evaluation.plots import build_all

    escritos = build_all(**_given(args, "results_dir", "out_dir"), split=args.split)
    print(f"{len(escritos)} figuras:")
    for caminho in escritos:
        print(f"  {caminho}")


def _add_split(parser: argparse.ArgumentParser, default: str = "validacao",
               help: str = "validacao: o que o desenvolvimento le. teste: so na passada "
                           "final.") -> None:
    from gefx.data.grid import EVAL_SPLITS

    parser.add_argument("--split", default=default, choices=list(EVAL_SPLITS), help=help)


def _write_table(tabela, destino: Path) -> None:
    print(tabela.to_string(index=False))
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


#: Flags de `gefx disent train` que vao direto para o `EncoderTrainConfig`. Sem a
#: flag, fica o padrao dele: os numeros nao se repetem aqui.
_TRAIN_FLAGS = ("steps", "seed", "batch_by")


def _cmd_disent_train(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from gefx.data.grid import split_file
    from gefx.evaluation.runner import compare
    from gefx.training.architecture import ARCHITECTURES
    from gefx.training.encoder import EncoderTrainConfig, run_name, train

    flags = _given(args, *_TRAIN_FLAGS)
    architecture = ARCHITECTURES[args.arch]
    if args.learning_rate is not None:
        architecture = replace(architecture, learning_rate=args.learning_rate)
    if args.content_dim is not None:
        architecture = replace(architecture, content_dim=args.content_dim)
    base = EncoderTrainConfig(
        dataset_root=_dataset_root(args),
        deterministic=args.deterministic,
        architecture=architecture,
        **flags,
    )
    raiz = _results_dir(args)
    for nome in dict.fromkeys(args.technique or ["supcon"]):
        pasta = run_name(nome, base.seed)
        print(f"[{pasta}]")
        train(replace(base, technique=nome, output_dir=raiz / pasta))

    print()
    _write_table(compare(raiz), raiz / split_file("escada", "validacao", "csv"))


def _cmd_disent_probe(args: argparse.Namespace) -> None:
    from gefx.data.grid import split_file
    from gefx.evaluation.probes import probe_study

    raiz = _results_dir(args)
    tabela = probe_study(raiz, _dataset_root(args), split=args.split,
                         **_given(args, "runs", "folds", "seed"))
    _write_table(tabela, raiz / split_file("sondas", args.split, "csv"))


def _cmd_disent_diversity(args: argparse.Namespace) -> None:
    from gefx.evaluation.loo import arm_diversity_curve, summary_file

    raiz = _results_dir(args, "diversidade")
    tabela = arm_diversity_curve(
        args.held_out, _dataset_root(args), raiz, split=args.split,
        **_given(args, "steps", "seed", "reuse"),
    )
    print()
    print(tabela.to_string(index=False))
    print(f"\ntabela em {raiz / summary_file(args.split)}")


def _cmd_disent_loo(args: argparse.Namespace) -> None:
    from gefx.evaluation.loo import cost_file, leave_one_arm_out, transfer_cost
    from gefx.training.encoder import EncoderTrainConfig

    raiz = _results_dir(args, "loo")
    tabela = leave_one_arm_out(
        _dataset_root(args), raiz, steps=args.steps,
        seeds=list(dict.fromkeys([EncoderTrainConfig.seed if args.seed is None else args.seed,
                                  *(args.extra_seed or [])])),
        split=args.split,
    )
    # O encoder treinado com todos os arms, na pasta acima da do leave-one-out, nas
    # mesmas sementes.
    custo = transfer_cost(tabela, raiz.parent, split=args.split)
    print()
    print(custo.to_string(index=False))
    print(f"\ncusto medio de nunca ter visto a implementacao: "
          f"{custo.custo_pontos.mean():+.1f} pontos")
    print(custo.groupby("estrato").custo_pontos.mean().round(1).to_string())
    custo.to_csv(raiz / cost_file(args.split), index=False)


def _cmd_disent_evaluate(args: argparse.Namespace) -> None:
    from gefx.data.grid import split_file
    from gefx.evaluation.runner import evaluate_runs

    raiz = _results_dir(args)
    _write_table(evaluate_runs(raiz, args.run, split=args.split),
                 raiz / split_file("escada", args.split, "csv"))


def _cmd_disent_compare(args: argparse.Namespace) -> None:
    from gefx.evaluation.metrics import compare_runs

    tabela = compare_runs([Path(run) for run in args.a], [Path(run) for run in args.b],
                          split=args.split)
    print(f"acerto exato do nivel (%), A - B com IC95% por gravacao, {args.split}:")
    print(tabela.to_string(index=False))


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
    cross_render.add_argument("--n", type=int, default=None, help="Renders por (efeito, braco).")
    cross_render.add_argument("--input-dir", default=None)
    cross_render.add_argument("--output-root", default=None)
    cross_render.add_argument("--segment-seconds", type=float, default=None)
    cross_render.add_argument("--seed", type=int, default=None)
    cross_render.add_argument("--legacy", action="store_true",
                              help="Ordena as colunas JSON do sidecar alfabeticamente.")
    cross_render.set_defaults(func=_cmd_cross_impl_render)

    cross_eval = cross_sub.add_parser("eval", help="Roda os modelos treinados em cada braco.")
    cross_eval.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    cross_eval.add_argument("--models-root", default="results/second_main_run")
    cross_eval.add_argument("--output-root", default="datasets/cross_impl")
    cross_eval.set_defaults(func=_cmd_cross_impl_eval)

    # gefx disent
    from gefx.training.architecture import ARCHITECTURES, TECHNIQUES

    disent = sub.add_parser("disent", help="POC II: recuperacao por um codigo de efeito desemaranhado.")
    disent_sub = disent.add_subparsers(dest="disent_command", required=True)

    # Sem padrao nas flags do POC II: sem a flag, vale o de quem a recebe (as
    # dataclasses de opcoes, `DATASET_ROOT`, `RESULTS_ROOT`), e o numero nao se
    # repete aqui.
    disent_calibrate = disent_sub.add_parser(
        "calibrate", help="Niveis de drive de todos os arms, pareados pelo Rnonlin.")
    disent_calibrate.add_argument("--roster", dest="roster_path", type=Path, default=None)
    disent_calibrate.add_argument("--levels", dest="levels_path", type=Path, default=None,
                                  help="Reescrito inteiro.")
    disent_calibrate.add_argument("--input-dir", type=Path, default=None)
    disent_calibrate.add_argument("--curves", dest="curves_csv", type=Path, default=None)
    disent_calibrate.add_argument("--points", type=int, default=None,
                                  help="Posicoes do knob varridas por arm.")
    disent_calibrate.add_argument("--segments", dest="n_segments", type=int, default=None,
                                  help="Trechos de treino em que cada knob e medido.")
    disent_calibrate.add_argument("--workers", type=int, default=None)
    disent_calibrate.add_argument("--ref-db", dest="reference_db", type=float, nargs=2,
                                  metavar=("MIN", "MAX"), default=None,
                                  help="Faixa da referencia em dB; niveis igualmente espacados "
                                       "em Rnonlin.")
    disent_calibrate.add_argument("--n-levels", type=int, default=None)
    disent_calibrate.set_defaults(func=_cmd_disent_calibrate)

    disent_render = disent_sub.add_parser(
        "render", help="Renderiza a grade combinatoria conteudo x nivel x arm."
    )
    disent_render.add_argument("--arm", dest="arms", action="append", default=None,
                               help="Repetivel. Padrao: todos os arms do roster.")
    disent_render.add_argument("--input-dir", type=Path, default=None)
    disent_render.add_argument("--output-root", type=Path, default=None)
    disent_render.add_argument("--roster", type=Path, default=None,
                               help="Os plugins de cada arm.")
    disent_render.add_argument("--levels", type=Path, default=None,
                               help="Os knobs de drive de cada nivel (so `levels` e usado).")
    disent_render.add_argument("--contents", dest="n_contents", type=int, default=None,
                               help="Gravacoes usadas; a particao e por gravacao.")
    disent_render.add_argument("--segments-per-file", type=int, default=None,
                               help="Trechos consecutivos por gravacao; cada um e um conteudo.")
    disent_render.add_argument("--segment-seconds", type=float, default=None)
    disent_render.add_argument("--split", dest="splits", action="append", default=None,
                               choices=["train", "catalog", "query"],
                               help="Repetivel. So estas particoes. Padrao: todas.")
    disent_render.add_argument("--seed", type=int, default=None)
    disent_render.add_argument("--split-seed", type=int, default=None)
    disent_render.add_argument("--workers", type=int, default=None)
    disent_render.set_defaults(func=_cmd_disent_render)

    disent_retrieve = disent_sub.add_parser(
        "retrieve", help="Baselines sem aprendizado do POC II: B0 e B1.")
    disent_retrieve.add_argument("--baseline", default="b0", choices=["b0", "b1"],
                                 help="b0: vizinho mais proximo no Spec padronizado, a "
                                      "entrada do encoder. b1: regressor do POC I, sem "
                                      "retreino.")
    disent_retrieve.add_argument("--output-root", default=None)
    disent_retrieve.add_argument("--results-dir", default=None,
                                 help="Grava <results-dir>/<baseline>_<split>.{csv,json}, "
                                      "onde a escada e as figuras os procuram. Padrao: a "
                                      "pasta acima das execucoes do encoder.")
    _add_split(disent_retrieve)
    disent_retrieve.add_argument("--arm", action="append", default=None,
                                 help="Restringe o roster. Padrao: todos.")
    disent_retrieve.set_defaults(func=_cmd_disent_retrieve)

    disent_plots = disent_sub.add_parser(
        "plots", help="Figuras dos baselines e do encoder, do que ja esta gravado.")
    disent_plots.add_argument("--results-dir", type=Path, default=None,
                              help="Baselines na raiz, o encoder em <results-dir>/encoder.")
    disent_plots.add_argument("--out-dir", type=Path, default=None,
                              help="Padrao: <results-dir>/figuras_<split>.")
    _add_split(disent_plots)
    disent_plots.set_defaults(func=_cmd_disent_plots)

    disent_train = disent_sub.add_parser(
        "train",
        help="Treina o encoder (supcon) ou roda um dos controles sem treino.",
    )
    disent_train.add_argument("--output-root", default=None,
                              help="Raiz do dataset do POC II.")
    disent_train.add_argument("--results-dir", default=None)
    disent_train.add_argument("--technique", action="append", default=None,
                              choices=list(TECHNIQUES),
                              help="Repetivel. Padrao: supcon. random_encoder: "
                                   "zero passos. bn_only: so calibra a BatchNorm, sem "
                                   "gradiente. regressao: o mesmo tronco com uma saida "
                                   "so, o drive em dB, por MSE.")
    disent_train.add_argument("--steps", type=int, default=None,
                              help="Os passos, e o comprimento do cosseno. Ficam os pesos "
                                   "do minimo do erro de validacao.")
    disent_train.add_argument("--learning-rate", type=float, default=None,
                              help="A taxa do inicio do cosseno.")
    disent_train.add_argument("--seed", type=int, default=None,
                              help="Fora do padrao, grava em <tecnica>_s<semente>.")
    disent_train.add_argument("--deterministic", action="store_true",
                              help="Nucleos deterministicos do TensorFlow, sem o XLA. Sem "
                                   "isto a semente fixa so a inicializacao e duas execucoes "
                                   "identicas divergem.")
    disent_train.add_argument("--arch", default="poc2", choices=list(ARCHITECTURES),
                              help="Os degraus da escada, na ordem: poc1 (a CNN do POC "
                                   "I), poc1_media (+ media no tempo), poc2 (+ filtros "
                                   "32, 64: o encoder do relatorio), poc2_fundo (+ quatro "
                                   "blocos: medido, nao entrou).")
    disent_train.add_argument("--batch-by", default=None, choices=["level", "content"],
                              help="A classe do batch P x K: level (padrao, P niveis x "
                                   "K trechos) ou content (P trechos x K niveis; o unico "
                                   "com positivos de conteudo).")
    disent_train.add_argument("--content-dim", type=int, default=None,
                              help="Largura de z_c, a cabeca de conteudo (padrao 0: sem "
                                   "ela). Exige --batch-by content.")
    disent_train.set_defaults(func=_cmd_disent_train)

    disent_between = disent_sub.add_parser(
        "between",
        help="Consultas nos pontos medios entre niveis contra o catalogo da grade.",
    )
    disent_between.add_argument("--write-levels", type=Path, default=None,
                                help="So grava o arquivo de niveis intermediarios, para "
                                     "`render --levels` (com --split query).")
    disent_between.add_argument("--levels", dest="levels_path", type=Path, default=None)
    disent_between.add_argument("--curves", dest="curves_csv", type=Path, default=None)
    disent_between.add_argument("--results-dir", default=None)
    disent_between.add_argument("--output-root", default=None,
                                help="Dataset da grade, de onde sai o catalogo (o do "
                                     "teste: os pontos medios so existem para as "
                                     "gravacoes de consulta, entao isto e da passada final).")
    disent_between.add_argument("--between-root", type=Path,
                                default=Path("datasets/disent_v2_entre"),
                                help="Dataset das consultas nos pontos medios.")
    disent_between.add_argument("--run", dest="runs", action="append", default=None,
                                help="Repetivel. Padrao: supcon.")
    disent_between.add_argument("--k", type=int, default=None)
    disent_between.set_defaults(func=_cmd_disent_between)

    disent_probe = disent_sub.add_parser(
        "probe",
        help="Sondas lineares sobre o z_e: que fatores o codigo ainda deixa ler.",
    )
    disent_probe.add_argument("--results-dir", default=None)
    disent_probe.add_argument("--output-root", default=None)
    disent_probe.add_argument("--run", dest="runs", action="append", default=None,
                              help="Repetivel. Padrao: toda execucao em --results-dir.")
    _add_split(disent_probe, help="Sonda o catalogo dessa busca, o que ela de fato "
                                  "consulta.")
    disent_probe.add_argument("--folds", type=int, default=None)
    disent_probe.add_argument("--seed", type=int, default=None)
    disent_probe.set_defaults(func=_cmd_disent_probe)

    disent_loo = disent_sub.add_parser(
        "loo", help="Leave-one-arm-out: a transferencia para implementacao inedita.")
    disent_loo.add_argument("--output-root", default=None)
    disent_loo.add_argument("--results-dir", default=None,
                            help="Padrao: <encoder>/loo; o custo de transferencia le o "
                                 "encoder com todos os arms na pasta acima.")
    disent_loo.add_argument("--steps", type=int, default=None,
                            help="O teto de passos; padrao, o do train.")
    disent_loo.add_argument("--seed", type=int, default=None, help="Padrao, a do train.")
    disent_loo.add_argument("--extra-seed", type=int, action="append", default=None,
                            help="Repetivel. Repete o leave-one-out inteiro com outra "
                                 "semente; e o unico jeito de por barra no custo por "
                                 "arm, que tem so 800 consultas.")
    _add_split(disent_loo)
    disent_loo.set_defaults(func=_cmd_disent_loo)

    disent_diversity = disent_sub.add_parser(
        "diversity",
        help="B2 e B3: treina com 1, 2, ... N-1 implementacoes e mede a transferencia.",
    )
    disent_diversity.add_argument("--output-root", default=None)
    disent_diversity.add_argument("--results-dir", default=None,
                                  help="Padrao: <encoder>/diversidade.")
    disent_diversity.add_argument("--held-out", required=True,
                                  help="O arm que nunca entra no treino.")
    disent_diversity.add_argument("--steps", type=int, default=None,
                                  help="O teto de passos; padrao, o do train.")
    disent_diversity.add_argument("--seed", type=int, default=None, help="Padrao, a do train.")
    disent_diversity.add_argument("--reuse", type=Path, default=None,
                                  help="Diretorio do leave-one-out (padrao <encoder>/loo): "
                                       "o ultimo ponto da curva e a mesma execucao e nao e "
                                       "retreinado.")
    _add_split(disent_diversity)
    disent_diversity.set_defaults(func=_cmd_disent_diversity)

    disent_evaluate = disent_sub.add_parser(
        "evaluate",
        help="Avalia execucoes ja treinadas numa particao. E a passada final do teste, "
             "com os modelos finais; o treino so le a validacao.",
    )
    disent_evaluate.add_argument("--results-dir", default=None)
    disent_evaluate.add_argument("--run", action="append", default=None,
                                 help="Repetivel. Padrao: toda execucao em --results-dir.")
    _add_split(disent_evaluate, "teste")
    disent_evaluate.set_defaults(func=_cmd_disent_evaluate)

    disent_compare = disent_sub.add_parser(
        "compare",
        help="Diferenca pareada A - B no acerto, com IC95%% por bootstrap sobre gravacoes.")
    disent_compare.add_argument("--a", action="append", required=True,
                                help="Pasta de uma execucao. Repetivel, uma por semente.")
    disent_compare.add_argument("--b", action="append", required=True,
                                help="Idem, na mesma ordem de sementes que --a.")
    _add_split(disent_compare)
    disent_compare.set_defaults(func=_cmd_disent_compare)

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
