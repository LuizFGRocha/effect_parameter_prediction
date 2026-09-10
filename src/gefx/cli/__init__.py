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
    from gefx.disent.calibrate import calibrate_arms

    calibrate_arms(
        arm_keys_wanted=args.arm,
        input_dir=Path(args.input_dir),
        output=Path(args.output),
        points=args.points,
        n_probes=args.probes,
        segment_seconds=args.segment_seconds,
        **({"descriptor": args.descriptor} if args.descriptor else {}),
        range_mode=args.range_mode,
        level_mode=args.level_mode,
    )


def _cmd_disent_render(args: argparse.Namespace) -> None:
    from gefx.disent.render import RenderOptions, render

    render(
        RenderOptions(
            input_dir=Path(args.input_dir),
            output_root=Path(args.output_root),
            calibration=Path(args.calibration) if args.calibration else None,
            n_contents=args.contents,
            segment_seconds=args.segment_seconds,
            seed=args.seed,
            split_seed=args.split_seed,
            arms=args.arm,
            workers=args.workers,
        )
    )


def _cmd_disent_cache(args: argparse.Namespace) -> None:
    from gefx.disent.features import build_all_caches

    build_all_caches(Path(args.output_root), args.feature, arms=args.arm, rebuild=args.rebuild)


def _cmd_disent_retrieve(args: argparse.Namespace) -> None:
    import json

    from gefx.disent.retrieval import baseline_b0, cross_arm_table

    result = baseline_b0(
        Path(args.output_root),
        query_split=args.query_split,
        catalog_split=args.catalog_split,
        arms=args.arm,
        same_arm=args.same_arm,
    )
    overall = result.metrics["overall"]
    rotulo = "mesmo arm (controle)" if args.same_arm else "entre implementacoes"
    print(f"B0 -- vizinho mais proximo, {rotulo}, {overall['n']} consultas")
    for axis in ("drive_level", "tone_level"):
        item = overall[axis]
        print(f"  {axis:12s} exato {item['exact']:.1%} (acaso {item['chance']:.1%})  "
              f"+-1 {item['within_one']:.1%}  MAE {item['mae_levels']:.2f} niveis")
    print(f"  {'config':12s} exato {overall['config_exact']:.1%} "
          f"(acaso {overall['config_chance']:.2%})")
    print(f"  {'drive':12s} MAE {overall['mae_db']:.2f} dB equivalentes")
    print("\npor arm de consulta (drive exato):")
    for arm, item in sorted(result.metrics["per_query_arm"].items()):
        print(f"  {arm:16s} {item['drive_level']['exact']:.1%}  "
              f"MAE {item['drive_level']['mae_levels']:.2f}  "
              f"{item['mae_db']:.2f} dB")

    if args.output:
        destino = Path(args.output)
        destino.parent.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(destino.with_suffix(".csv"), index=False)
        destino.with_suffix(".json").write_text(
            json.dumps(result.metrics, indent=2), encoding="utf-8"
        )
        cross_arm_table(result.predictions).to_csv(
            destino.parent / f"{destino.stem}_cross_arm.csv"
        )
        print(f"\nescrito em {destino.parent}/{destino.stem}.{{csv,json}}")


def _cmd_disent_plots(args: argparse.Namespace) -> None:
    from gefx.disent.plots import build_all, build_etapa5

    constroi = build_etapa5 if args.etapa5 else build_all
    escritos = constroi(Path(args.results_dir), Path(args.out_dir) if args.out_dir else None)
    print(f"{len(escritos)} figuras:")
    for caminho in escritos:
        print(f"  {caminho}")


def _cmd_disent_train(args: argparse.Namespace) -> None:
    from gefx.disent.train import (
        STUDY_ORDER, TECHNIQUES, WEIGHT_VARIANTS, TrainConfig, compare, train,
    )

    conhecidas = {**TECHNIQUES, **WEIGHT_VARIANTS}

    escolhidas = args.technique or ["full"]
    # A ordem de execucao decide quais criterios podem ser avaliados, entao ela e
    # imposta e nao herdada da linha de comando.
    escolhidas = [nome for nome in STUDY_ORDER if nome in set(escolhidas)] or escolhidas
    saida_base = Path(args.output_dir) if args.output_dir else None
    # As referencias sao acumuladas na ordem em que as tecnicas rodam: o
    # `random_encoder` tem de vir antes de quem ele avalia, senao o criterio mais
    # duro do estudo fica de fora.
    referencias: dict = {}
    for nome in escolhidas:
        if nome not in conhecidas:
            raise SystemExit(f"tecnica desconhecida: {nome}. Ha {sorted(conhecidas)}")
        print(f"[{nome}]")
        config = TrainConfig(
            dataset_root=Path(args.output_root),
            feature=args.feature,
            technique=nome,
            arms=tuple(args.arm) if args.arm else None,
            steps=args.steps,
            configs_per_batch=args.configs_per_batch,
            views_per_config=args.views_per_config,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            beta=args.beta,
            eval_every=args.eval_every,
            seed=args.seed,
            deterministic=args.deterministic,
            permute_labels=args.permute_labels,
            output_dir=(saida_base / nome) if saida_base else None,
        )
        manifesto = train(config, references=referencias)
        referencias[nome] = manifesto["decision"]["measured"]
        for criterio, veredito in manifesto["decision"]["verdicts"].items():
            print(f"    {'PASSA' if veredito else 'FALHA'}  {criterio}")

    raiz = saida_base or Path("results/disent/etapa5")
    # As variantes de peso nao estao em `TECHNIQUES`, que e o padrao do `compare`.
    tabela = compare(raiz, techniques=sorted(set(TECHNIQUES) | set(escolhidas)))
    print()
    print(tabela.to_string(index=False))
    destino = raiz / "comparacao.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


def _cmd_disent_probe(args: argparse.Namespace) -> None:
    from gefx.disent.diagnostics import probe_study

    tabela = probe_study(
        Path(args.results_dir), Path(args.output_root),
        techniques=args.technique, split=args.split, folds=args.folds, seed=args.seed,
    )
    print(tabela.to_string(index=False))
    destino = Path(args.results_dir) / "sondas.csv"
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


def _cmd_disent_structure(args: argparse.Namespace) -> None:
    from gefx.disent.diagnostics import structure_study

    tabela = structure_study(
        Path(args.results_dir), Path(args.output_root),
        techniques=args.technique, split=args.split, seed=args.seed, trees=args.trees,
    )
    print(tabela.to_string(index=False))
    destino = Path(args.results_dir) / "estrutura.csv"
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


def _cmd_disent_bootstrap(args: argparse.Namespace) -> None:
    from gefx.disent.diagnostics import bootstrap_study, find_runs

    execucoes = find_runs(Path(args.results_dir))
    pares = []
    for texto in args.pair or []:
        if texto.count(":") != 1:
            raise SystemExit(f"par mal formado: {texto!r}. Use a:b")
        esquerda, direita = texto.split(":")
        for nome in (esquerda, direita):
            if nome not in execucoes:
                raise SystemExit(
                    f"execucao sem predictions.csv: {nome}. Ha {sorted(execucoes)}"
                )
        pares.append((esquerda, direita))
    if not pares:
        raise SystemExit("nenhum par: use --pair a:b (repetivel)")

    tabela = bootstrap_study(execucoes, pares, reps=args.reps, seed=args.seed)
    print(tabela.to_string(index=False))
    destino = Path(args.results_dir) / "bootstrap.csv"
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


def _cmd_disent_diversity(args: argparse.Namespace) -> None:
    from gefx.disent.loo import arm_diversity_curve

    tabela = arm_diversity_curve(
        Path(args.output_root), Path(args.results_dir), held_out=args.held_out,
        technique=args.technique, steps=args.steps, seed=args.seed,
        reuse=Path(args.reuse) if args.reuse else None,
    )
    print()
    print(tabela.to_string(index=False))
    print(f"\ntabela em {Path(args.results_dir) / 'resumo.csv'}")


def _cmd_disent_loo(args: argparse.Namespace) -> None:
    from gefx.disent.loo import leave_one_arm_out, transfer_cost

    tabela = leave_one_arm_out(
        Path(args.output_root), Path(args.results_dir), technique=args.technique,
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


def _cmd_disent_ablate(args: argparse.Namespace) -> None:
    from gefx.disent.diagnostics import ablate_representation

    tabela = ablate_representation(Path(args.output_root), seed=args.seed)
    print(tabela.to_string(index=False))
    destino = Path(args.results_dir) / "ablacao_representacao.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)
    print(f"\ntabela em {destino}")


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
    disent = sub.add_parser("disent", help="POC II: recuperacao em espaco latente desemaranhado.")
    disent_sub = disent.add_subparsers(dest="disent_command", required=True)

    disent_calibrate = disent_sub.add_parser(
        "calibrate", help="Calibra os extremos do knob de drive de cada arm contra a referencia."
    )
    disent_calibrate.add_argument("--arm", action="append", default=None,
                                  help="Repetivel. Padrao: todos os arms do roster.")
    disent_calibrate.add_argument("--input-dir", default="datasets/unprocessed_samples")
    disent_calibrate.add_argument("--output", default="datasets/disent/arms_calibration.json")
    disent_calibrate.add_argument("--points", type=int, default=33, help="Pontos da varredura.")
    disent_calibrate.add_argument("--probes", type=int, default=8,
                                  help="Segmentos de guitarra usados como probe.")
    disent_calibrate.add_argument("--segment-seconds", type=float, default=2.0)
    # Sem valor cravado aqui: o padrao e o `PRIMARY_DESCRIPTOR` do modulo, para o
    # CLI nao sobrepor silenciosamente a escolha de desenho (ja aconteceu).
    disent_calibrate.add_argument("--descriptor", default=None,
                                  choices=["thd", "crest_drop", "hf_ratio", "flatness", "thd_flatness"],
                                  help="Padrao: o PRIMARY_DESCRIPTOR de disent/calibrate.py.")
    disent_calibrate.add_argument("--range-mode", default="intersection",
                                  choices=["intersection", "reference"],
                                  help="intersection: faixa que todos os arms alcancam.")
    disent_calibrate.add_argument("--level-mode", default="descriptor",
                                  choices=["descriptor", "knob"],
                                  help="descriptor: casa os 8 niveis contra a referencia. "
                                       "knob: uniformes no knob nativo (condicao de comparacao).")
    disent_calibrate.set_defaults(func=_cmd_disent_calibrate)

    disent_render = disent_sub.add_parser(
        "render", help="Renderiza a grade combinatoria conteudo x configuracao x arm."
    )
    disent_render.add_argument("--arm", action="append", default=None,
                               help="Repetivel. Padrao: os arms aprovados na calibracao.")
    disent_render.add_argument("--input-dir", default="datasets/unprocessed_samples")
    disent_render.add_argument("--output-root", default="datasets/disent")
    disent_render.add_argument("--calibration", default=None,
                               help="Padrao: <output-root>/arms_calibration.json")
    disent_render.add_argument("--contents", type=int, default=100,
                               help="Gravacoes distintas usadas como conteudo.")
    disent_render.add_argument("--segment-seconds", type=float, default=2.0)
    disent_render.add_argument("--seed", type=int, default=20260906)
    disent_render.add_argument("--split-seed", type=int, default=20260906)
    disent_render.add_argument("--workers", type=int, default=4)
    disent_render.set_defaults(func=_cmd_disent_render)

    disent_retrieve = disent_sub.add_parser(
        "retrieve",
        help="Baseline B0: recuperacao por vizinho mais proximo, sem aprendizado.")
    disent_retrieve.add_argument("--output-root", default="datasets/disent")
    disent_retrieve.add_argument("--arm", action="append", default=None,
                                 help="Restringe o roster. Padrao: todos.")
    disent_retrieve.add_argument("--query-split", default="query")
    disent_retrieve.add_argument("--catalog-split", default="catalog")
    disent_retrieve.add_argument("--same-arm", action="store_true",
                                 help="Controle: deixa o arm da consulta no catalogo.")
    disent_retrieve.add_argument("--output", default=None,
                                 help="Prefixo para gravar predicoes e metricas.")
    disent_retrieve.set_defaults(func=_cmd_disent_retrieve)

    disent_plots = disent_sub.add_parser(
        "plots", help="Figuras dos baselines, a partir do que `retrieve` gravou.")
    disent_plots.add_argument("--results-dir", default="results/disent")
    disent_plots.add_argument("--etapa5", action="store_true",
                              help="Figuras do estudo comparativo (results/disent/etapa5) "
                                   "em vez das dos baselines da etapa 4.")
    disent_plots.add_argument("--out-dir", default=None,
                              help="Padrao: <results-dir>/figuras.")
    disent_plots.set_defaults(func=_cmd_disent_plots)

    disent_cache = disent_sub.add_parser(
        "cache", help="Extrai o cache de features de cada arm."
    )
    disent_cache.add_argument("--output-root", default="datasets/disent")
    disent_cache.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    disent_cache.add_argument("--arm", action="append", default=None)
    disent_cache.add_argument("--rebuild", action="store_true")
    disent_cache.set_defaults(func=_cmd_disent_cache)

    disent_train = disent_sub.add_parser(
        "train",
        help="Etapa 5: treina o encoder desemaranhado e roda o estudo comparativo.",
    )
    disent_train.add_argument("--output-root", default="datasets/disent",
                              help="Raiz do dataset do POC II.")
    disent_train.add_argument("--feature", default="Spec", choices=FEATURE_CHOICES)
    disent_train.add_argument("--technique", action="append", default=None,
                              help="Repetivel. Padrao: full. Rode 'random_encoder' "
                                   "primeiro para que o criterio mais duro valha.")
    disent_train.add_argument("--arm", action="append", default=None,
                              help="Restringe as implementacoes (leave-one-arm-out).")
    disent_train.add_argument("--steps", type=int, default=4000)
    disent_train.add_argument("--configs-per-batch", type=int, default=8)
    disent_train.add_argument("--views-per-config", type=int, default=8)
    disent_train.add_argument("--learning-rate", type=float, default=1e-3)
    disent_train.add_argument("--temperature", type=float, default=0.07)
    disent_train.add_argument("--beta", type=float, default=4.0,
                              help="Peso do KL no controle beta-VAE.")
    disent_train.add_argument("--eval-every", type=int, default=500,
                              help="0 desliga a avaliacao intermediaria.")
    disent_train.add_argument("--seed", type=int, default=20260908)
    disent_train.add_argument("--deterministic", action="store_true",
                              help="Nucleos deterministicos do TensorFlow. Sem isto a "
                                   "semente fixa so a inicializacao e duas execucoes "
                                   "identicas divergem (medido: ate 7 pontos numa "
                                   "metrica de 800 consultas). Custa ~20%% de velocidade.")
    disent_train.add_argument("--permute-labels", action="store_true",
                              help="Controle de permutacao: embaralha a configuracao "
                                   "dentro de cada (conteudo, implementacao). Testa se "
                                   "o ganho vem do rotulo ou do procedimento de treino.")
    disent_train.add_argument("--output-dir", default=None,
                              help="Padrao: results/disent/etapa5/<tecnica>.")
    disent_train.set_defaults(func=_cmd_disent_train)

    disent_probe = disent_sub.add_parser(
        "probe",
        help="Sondas lineares sobre o z_e das execucoes da etapa 5.",
    )
    disent_probe.add_argument("--results-dir", default="results/disent/etapa5")
    disent_probe.add_argument("--output-root", default="datasets/disent")
    disent_probe.add_argument("--technique", action="append", default=None)
    disent_probe.add_argument("--split", default="catalog",
                              help="Particao sondada. O catalogo e o padrao porque "
                                   "e o que a busca de fato consulta.")
    disent_probe.add_argument("--folds", type=int, default=3)
    disent_probe.add_argument("--seed", type=int, default=0)
    disent_probe.set_defaults(func=_cmd_disent_probe)

    disent_loo = disent_sub.add_parser(
        "loo", help="Etapa 7: leave-one-arm-out, a transferencia para implementacao inedita.")
    disent_loo.add_argument("--output-root", default="datasets/disent")
    disent_loo.add_argument("--results-dir", default="results/disent/etapa5/loo")
    disent_loo.add_argument("--technique", default="contrastive_aux")
    disent_loo.add_argument("--steps", type=int, default=4000)
    disent_loo.add_argument("--seed", type=int, default=20260908)
    disent_loo.add_argument("--extra-seed", type=int, action="append", default=None,
                            help="Repetivel. Repete o leave-one-out inteiro com outra "
                                 "semente; e o unico jeito de por barra no custo por "
                                 "arm, que tem so 800 consultas.")
    disent_loo.set_defaults(func=_cmd_disent_loo)

    disent_structure = disent_sub.add_parser(
        "structure",
        help="Etapa 6: DCI e MIG sobre [z_e | z_c], e a massa de cada fator por bloco.",
    )
    disent_structure.add_argument("--results-dir", default="results/disent/etapa5")
    disent_structure.add_argument("--output-root", default="datasets/disent")
    disent_structure.add_argument("--technique", action="append", default=None)
    disent_structure.add_argument("--split", default="catalog")
    disent_structure.add_argument("--trees", type=int, default=200,
                                  help="Arvores da floresta que mede a importancia.")
    disent_structure.add_argument("--seed", type=int, default=0)
    disent_structure.set_defaults(func=_cmd_disent_structure)

    disent_bootstrap = disent_sub.add_parser(
        "bootstrap",
        help="IC 95% agrupado por conteudo para a diferenca entre duas execucoes.",
    )
    disent_bootstrap.add_argument("--results-dir", default="results/disent/etapa5")
    disent_bootstrap.add_argument("--pair", action="append", default=None,
                                  help="Repetivel, no formato a:b. As 5.600 consultas "
                                       "sao 20 conteudos x 40 config x 7 arms: um IC "
                                       "por linha sai 3,5x estreito demais.")
    disent_bootstrap.add_argument("--reps", type=int, default=4000)
    disent_bootstrap.add_argument("--seed", type=int, default=0)
    disent_bootstrap.set_defaults(func=_cmd_disent_bootstrap)

    disent_diversity = disent_sub.add_parser(
        "diversity",
        help="B2 e B3: treina com 1, 2, ... N-1 implementacoes e mede a transferencia.",
    )
    disent_diversity.add_argument("--output-root", default="datasets/disent")
    disent_diversity.add_argument("--results-dir",
                                  default="results/disent/etapa5/diversidade")
    disent_diversity.add_argument("--held-out", default="byod-mxr")
    disent_diversity.add_argument("--technique", default="contrastive_aux")
    disent_diversity.add_argument("--steps", type=int, default=4000)
    disent_diversity.add_argument("--seed", type=int, default=20260908)
    disent_diversity.add_argument("--reuse", default="results/disent/etapa5/loo",
                                  help="Diretorio do leave-one-out: o ultimo ponto da "
                                       "curva e a mesma execucao e nao e retreinado.")
    disent_diversity.set_defaults(func=_cmd_disent_diversity)

    disent_ablate = disent_sub.add_parser(
        "ablate",
        help="Separa o ganho sobre o B0 em representacao, escala, dimensao e arquitetura.",
    )
    disent_ablate.add_argument("--output-root", default="datasets/disent")
    disent_ablate.add_argument("--results-dir", default="results/disent/etapa5")
    disent_ablate.add_argument("--seed", type=int, default=0)
    disent_ablate.set_defaults(func=_cmd_disent_ablate)

    disent_validate = disent_sub.add_parser(
        "validate", help="Confere que a grade esta cruzada e pareada entre os arms."
    )
    disent_validate.add_argument("--output-root", default="datasets/disent")
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
