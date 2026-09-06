# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UFMG "Projeto Orientado em Computação" (POC) research code: recovering guitar effect
parameter settings from audio. POC I reproduced Hinrichs et al. (CNN regression over
audio features) on a more realistic dataset derived from the Rossi et al. recordings.
POC II asks whether those models generalize to *other implementations* of the same
effect (`gefx.crossimpl`).

Reference material lives in `documents/`: `hinrichs_transcription.md` and
`rossi_transcription.md` (paper transcriptions) plus the `*.tex` Portuguese
proposals/reports. Code comments, docstrings and CLI help are in Portuguese; match that
when editing.

## Setup and commands

```bash
source .venv/bin/activate      # existing venv (Python 3.12, TF 2.21 + CUDA)
pip install -e .               # installs the `gefx` package and the `gefx` CLI
```

`src/gefx/` is a real installable package — there is no `sys.path` hack and no
requirement to run from the repo root. Everything goes through one CLI:

```bash
gefx render-dataset --output-dir datasets/default --samples-per-file 56
gefx train --config experiments/base.yaml --feature Spec --results-root results/my_run
gefx train --dataset-root datasets/default --chain-key distortion   # single chain
gefx evaluate --results-root results/my_run
gefx compare-features --results-base results/feature_runs

bash scripts/run_model_features.sh        # full sweep -> results/feature_runs/<FEATURE>/

# Cross-implementation study (needs the VST3s in plugins/real/)
gefx cross-impl render --arm pedalboard --arm lsp --n 300
gefx cross-impl eval --feature Spec --models-root results/second_main_run

# Discover a third-party plugin's parameters / knob→physical-unit mapping
gefx inspect-plugin plugins/real/ChowCentaur.vst3
gefx inspect-plugin plugins/real/ChowPhaserMono.vst3 --sweep lfo_freq
```

There is no linter or build step. `pytest` (config in `pyproject.toml`, dev deps under
the `dev` extra) runs a fast unit suite over the pure logic and the on-disk formats — no
GPU, no VST3, no fixture bigger than a `tmp_path`. `validate_sidecar_integrity`
(`gefx/data/metadata.py`, run automatically at the start of every training run) remains
the consistency check on a *dataset*, which the suite does not touch.

## Experiment configuration

An experiment is a YAML file (`experiments/base.yaml` reproduces the POC I setup) with
three sections — `dataset`, `architecture`, `train` — mapped to dataclasses in
`gefx/config.py`. Precedence is **dataclass default → YAML (`--config`) → explicit CLI
flag**; a flag left unset never overrides the YAML. Unknown sections or keys raise rather
than being silently ignored.

Every run writes `<results-root>/run.json` with the resolved config, the git SHA, a
timestamp and the per-chain metrics, and each chain's `metrics.json` records the
architecture that produced it.

Seeds: there are three, deliberately named apart because CLI overrides are flat — a field
name shared by two sections would be written to both at once. `dataset.render_seed`
picks the audio segments and the parameter vectors (i.e. the data); `train.split_seed`
picks the test rows; `train.seed` fixes `random`, `numpy` and Keras (weight init, dropout,
batch order). `--seed` on `train` sets the last one, `--seed` on `render-dataset` sets the
first. `train.seed` alone is *not* enough for two runs to match bit for bit on GPU —
residual nondeterminism is around 1e-4 — so `--deterministic` additionally enables
TensorFlow's deterministic kernels (verified to give identical predictions, at a cost in
speed).

`render_seed` reproduces a dataset only *together with the exact contents of the input
directory*: the spawned `SeedSequence` children are zipped against the sorted wav list, so
inserting a file anywhere but at the end re-seeds every file after it.

`dataset.legacy` (and `--legacy` on `evaluate` / `cross-impl render`) restores two older
conventions: alphabetically sorted `effect_presence`/`raw_parameter_dict` in the sidecar,
and alphabetically sorted parameter rows in `parameter_metrics.csv`. Both default to
catalog/target order now. `experiments/base.yaml` sets `legacy: true`, so re-rendering the
POC I dataset or re-running `gefx evaluate --legacy` reproduces the committed artifacts
byte for byte.

## Architecture

Pipeline: **render → cache features → train per chain → evaluate**. Each stage
communicates through files on disk, not through function calls.

**`effects/catalog.py` is the single source of truth.** `EFFECT_PARAMETER_RANGES`
defines every effect, its parameters, their `[min, max]` physical ranges, and whether
each parameter is predicted (`"predict": False` means it's held fixed and excluded from
the target vector). Parameters are stored **normalized to [0,1]** everywhere; raw
physical values are derived from the ranges. Adding or reordering entries here silently
invalidates existing datasets and trained models, since the target vector's length and
ordering come from it.

A *chain* is an ordered list of effects identified by a `chain_key` — the effect names
joined by `__` (e.g. `distortion__chorus__slapback_delay`). Chain keys are directory
names in both datasets and results. `EFFECT_CHAINS` = every single effect, plus the
three stacked subsets of `CANONICAL_EFFECT_CHAIN_ORDER`.

**`data/render.py`** loads each clean wav, picks a random segment, loudness-normalizes
(pyloudnorm, −26 LUFS via `gefx/audio.py`), renders it through a Pedalboard chain with
randomly sampled normalized parameters, normalizes again, and writes
`<output-dir>/<chain_key>/<file>.wav` plus a single `metadata.csv` sidecar at the dataset
root. Parallelized per source file with `ProcessPoolExecutor`; determinism comes from a
`SeedSequence` spawned per file, so results are stable regardless of completion order.
**The order of RNG draws inside `render_source_file` is part of the contract** — changing
it changes the dataset produced by a given seed. Rendered filenames carry a binary
effect-presence suffix (`les_bridge_fing01__10000000__s0000.wav`).

**`data/`** is the reader and cache layer, split by concern: `features.py` (extraction),
`cache.py` (`.npz` + `file_names.json`), `metadata.py` (sidecar), `dataset.py`
(`load_chain_dataset`). On first use of a (chain, feature) pair the features are
extracted for every wav in the chain folder and cached **next to the audio, inside the
dataset**. `file_names.json` is the canonical row ordering: `y` is built by looking each
cached filename up in `metadata.csv`, so cache and sidecar must stay in sync. Delete the
`.npz`/`file_names.json`, or pass `--rebuild-cache`, after changing feature extraction.
Features: `Spec` (rescaled STFT magnitude), `MFCC40`, `Chroma`, `GFCC40` (spafe,
resampled to 16 kHz and axis-swapped).

**`training/`** trains one small Conv2D→Dense regressor **per chain**, sigmoid output of
width = number of predictable parameters, MSE loss. `scaling.py` is the only place that
applies per-row `StandardScaler`s; they're fit on train only and persisted, because
`crossimpl` must reproduce the exact same scaling. `inference.py` is the shared
load-model-and-predict path used by both training and the cross-impl study — keeping them
on one code path is what makes the comparison meaningful. Each chain writes to
`<results-root>/<chain_key>/`: `model.keras`, `feature_scalers.pkl`, `history.json`,
`metrics.json`, `predictions.csv`, `split_indices.npz`. Keras sessions are cleared
between chains to bound memory.

**`evaluation/`** consumes only those per-chain files: `metrics.py` recomputes
per-parameter MAE/MSE with standard errors from `predictions.csv` rather than trusting
the aggregate in `metrics.json`; `plots.py` holds the shared primitives (`grouped_bars`,
`parity_grid`); `runner.py` drives a single run and `compare.py` sits one level above,
expecting `<results-base>/<FEATURE>/chain_metrics.csv`.

Note the nesting difference — `gefx train --results-root` writes chain dirs directly, so
a full sweep puts the feature name in the path (`results/second_main_run/Spec/distortion/`)
and `gefx cross-impl eval` resolves models as `<models-root>/<feature>/<effect>/`.

**`crossimpl/`** renders a *paired* evaluation set: the same audio segments and the same
normalized parameter vectors, rendered through several "arms" (implementations).
`pedalboard` is the mandatory in-domain reference arm — without it you cannot separate
"the model doesn't generalize" from "this eval set is just different".
`effects/registry.py` maps each reference parameter to a third-party plugin parameter
with a conversion function and an `exact` flag: exact mappings consume the *raw* physical
value (same unit), while non-exact ones (`onto(lo, hi)`, used where no physical
equivalence exists) consume the *normalized* value and are read via Spearman ρ rather
than MAE. Evaluation reports MAE, isotonic-calibrated MAE (an optimistic in-sample
ceiling on what monotonic calibration could buy), and ρ. Delay effects are deliberately
excluded — no third-party delay passed the sanity check.

Plugin quirks live in `effects/vst_adapter.py` and are worth preserving: the first
process call after `load_plugin` still uses stale parameters, so it is burned on silence
in `load_arm`; some plugins require stereo input (detected by the `ValueError` on a mono
probe); some parameters are only addressable via their displayed string (`BY_DISPLAY`).

## Conventions and gotchas

- **Datasets, results, plugins, `documents/` and `*.md` are gitignored**, except that
  `results/second_main_run/` and `results/teste_artigo/` were force-added and *are*
  tracked — they are the two reference results, and `cross-impl eval` defaults to the
  former's models. Anything else written under `results/` stays local. `documents/` is
  ignored wholesale, so the paper transcriptions and the `.tex` sources are local-only —
  read them from disk, don't expect them in git history. `CLAUDE.md` survives the `*.md`
  rule through an explicit `!CLAUDE.md` negation.
- Feature caches live inside the dataset directory, so copying a dataset copies stale
  caches with it. `datasets/default` is ~98 GB, of which ~57 GB is cache (`Spec.npz`
  alone is 41 GB).
- `datasets/other_dataset_structured` is the reference paper's data restructured into
  this repo's chain-key layout, and feeds `results/teste_artigo`. **It is not reproducible
  from code in this repo** — the script that produced it was never committed, and the raw
  download it came from has been deleted. Treat it as source data, not a derived artifact.
