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

## POC II drive levels (`gefx.disent`)

The level of distortion `i` must sound the same in every implementation (arm) — that
equivalence is what the cross-arm retrieval tests. Two files:

- `experiments/disent_roster.yaml` — hand-edited: each arm's plugin, `drive_param`,
  `fixed` params, `stratum`, `sweep: [min, max]` (the knob range to scan; required for
  the pedalboard backend and for knobs that go to −inf).
- `experiments/disent_levels.yaml` — written only by `calibrate`, never by hand: per arm
  `levels` (what `render` uses) and `unmatched` (1-based levels the arm can't reach).
  `parse_roster` refuses levels that aren't strictly increasing or differ in count
  between arms.

```bash
gefx disent calibrate                       # ~3.5 min, 8 processes; rewrites the levels file
gefx disent tune --recording guitar.wav     # localhost page to listen: A/B or L/R vs reference
gefx disent render                          # uses `levels`; copies roster + levels into the dataset
```

How `calibrate` works (`gefx/disent/calibrate.py`): every arm's knob is swept and
scored with **Rnonlin** (Tan, Moore, Zacharov & Mattila, JAES 2004 — gammatone band
cross-correlation between dry and wet; 1 = clean). The reference (`pedalboard-tanh`)
spans `--ref-db 18.45 39` with levels **equally spaced in 1 − Rnonlin**, and each other
arm gets the knob matching each level's Rnonlin. Levels outside an arm's reach go to the
end of its knob and are flagged `unmatched`. Curves start at each arm's cleanest knob (at
very low gain BYOD-MXR gets *less* clean: noise, not drive).

Decisions already made — don't reopen without new evidence:

- **Levels are always the automatic Rnonlin matching.** Manual adjustment by ear (sliders
  and saving in `gefx disent tune`, and the Reaper JSFX) was dropped on 2026-09-27;
  `unmatched` levels are accepted as they come out. `tune` stays as a **read-only**
  listening page — it has no save route and must not get one back.
- **Crest-factor drop was tried and rejected**: it saturates at high drive and is fooled
  by filtering after the clipper (BYOD read as half-scale; boosting the input doesn't
  help). Spectral descriptors (centroid, flatness) and % clipped samples were also worse.
- Rnonlin was chosen because reference levels the user separated **by ear** had the most
  regular steps in 1 − Rnonlin (CV 0.15–0.18 vs 0.21 in dB). Sanity check: `lsp-tanh`
  (same curve and unit as the reference) lands within ~1 dB of it.
- 18.45 dB is the first audibly distorted level (guitar at −26 LUFS); 39 dB keeps
  `byod-bigmuff` (the weakest arm, Rnonlin floor 0.830) off its flat top — above that its
  top levels sounded identical. The weakest arm caps the range for everyone: a level some
  arm can't reach would leave a hole in the fully-crossed grid (`GridIndex` and
  `validate_pairing` require it).
- The reference is the anchor of the dB unit that the evaluation reports.
- **No tone factor** (dropped 2026-09-27). It was our own low-pass after the
  nonlinearity, meant as an exactly shared positive control, but each pedal's own EQ
  (in some, gain-dependent) made "tone level t" differ across arms. The grid is drive
  only (`d0`…`d7`, `config_index == drive_level`). Positive controls are same-arm
  retrieval (the diagonal) and `lsp-tanh`; content is an *invariance* check (a probe
  on `z_e` should stay near chance), not a positive control.
- **Content = 2 s segments, 5 per recording** (2026-09-27): all 400 Rossi recordings
  are 11.5 s, so 5 consecutive non-overlapping segments fit inside the 0.5 s edge
  margins. That is 2000 contents (`<recording>_s0`…`_s4`) with the same 2 s input, so
  the POC I regressor (B1, trained on 2 s Spec) stays comparable — longer segments were
  rejected for that reason and for the doubled input cost. The split is **by
  recording** (`source_audio_id`): all segments of a performance share a split, and
  `retrieve` refuses a query and catalog that share a recording. The segments measured
  well (median 5% of frames 30 dB below the peak, 3% without an attack), so no
  onset-based segment selection.

**Encoder loss: SupCon only** (2026-09-27, technique `supcon`). The user wants the
method to rest on published code, so they can review it against the original and write
less new code.

- `losses.py` is a line-by-line TensorFlow translation of `SupConLoss` from the official
  repo (HobbitLong/SupContrast, commit 72fd989, BSD-2), temperature 0.07, with the
  original's English comments; only `mask` and `contrast_mode='one'` are left out. Keep
  it a translation; don't "improve" it.
- The auxiliary drive regression (the old `contrastive_aux`) was **dropped**: our own
  code with no published reference, and not distinguishable from pure contrastive
  (44.0 vs 42.7% drive exact, +1.25 with by-content IC95 [−1.95; +4.41],
  `documents/poc2_estado_2026-09-07.md`, old 40-config grid).
- **Rank-N-Contrast was tried and removed** (Zha et al. 2023, kaiwenzha/Rank-N-Contrast,
  ported in commit 08b264f, removed after). Rationale for trying: drive is ordered and
  RnC puts that order into the distances of `z_e`. On `disent_v2` (one seed each, commit
  08b264f) it **tied** SupCon everywhere, with IC95 resampled by recording:
  - on the grid: 68.4 vs 68.2% drive exact, 0.89 vs 0.88 dB (diff +0.26 pt,
    [−1.05; +1.61]); errors ≥ 2 levels 1.5 vs 1.8%;
  - between levels (`gefx disent between`, queries at the 7 midpoints): right neighbour
    92.2 vs 91.7%, k-NN error 1.23 vs 1.22 dB (diff [−0.031; +0.038]). Both are below
    half a step (1.47 dB), and neighbours split ~44/48% between the two adjacent levels:
    **SupCon's `z_e` is already ordered**, because adjacent levels sound alike;
  - but RnC left twice the content readable by a linear probe (27.0 vs 13.3%).
  It also needed a temperature deviation (0.1 instead of 2, our `z_e` is unit-norm) and
  its repo has no license. Don't bring it back without new evidence.

**What `z_e` contains** (2026-09-27, `supcon` seed 1, catalog split; analysis run by
hand, numbers to be regenerated by `probe` and `loo`):

- 93% of `z_e`'s variance is between drive levels: drive dominates the geometry.
- **The pedal is a constant offset per pedal**, not missing invariance in general. The
  global arm probe *rose* after training (43% untrained → 56%), but the within-level
  variance due to the pedal *fell* 3.6× (0.0148 → 0.0041): what's left is small but
  clean. Subtracting each pedal's mean `z_e` (no labels) takes the arm probe to 14.3%
  (chance 12.5%) and raises cross-arm retrieval 68.2 → 70.5% (same-arm control 71.1%).
  Why it's there: Rnonlin equalizes the *amount* of distortion, not the *character*
  (EQ, harmonic signature, BYOD's post-clip filter, Rat's filter; Rnonlin ignores
  linear filtering by design), and SupCon has no term against a residual that doesn't
  hurt level separation. The probe identifies bigmuff/mxr/Rat/hardclip at ~95% and
  confuses the three smooth waveshapers (tanh, pedalboard-tanh, sine); retrieval
  neighbours follow the same families (bigmuff → mxr 62%).
- `retrieval.center_by_arm` does that centering (per-channel mean normalization, as in
  speech). `loo` reports each condition with and without it (`centrado`); it assumes
  the pedal's unlabeled samples cover the levels evenly.
- **Content is reduced, not removed.** The global content probe (81% → 13%) overstated
  it: level dominates `z_e` and content's direction changes with the level. Within one
  level, content still reads at 51–55% (untrained 74–76%, chance 0.25%), and the nearest
  neighbour in another pedal at the same level is the same segment 36–39% of the time
  (untrained 67–71%). Within a level, content is 66% of `z_e`'s variance. Retrieval
  doesn't suffer (query and catalog share no recordings). Hence `linear_probes` now
  probes the nuisance factors **within each drive level** (`within="drive_level"`).
- Don't bring back the GRL for the pedal: in the old study it didn't remove it.
- Framing: Koo et al. 2023 (FXencoder, contrastive encoder that keeps only the effects)
  for audio; Wang et al. §3.3.2 (grouped supervision, ML-VAE/DC-IGN) for DRL. The
  time-average in the encoder is the "sequence-level factor" bias of FHVAE (Hsu et al.
  2017/2018).
- **FHVAE was evaluated and rejected as a base**: its code is TF 1.0/Python 2.7/Kaldi
  (unusable here, ScalableFHVAE likewise and unlicensed); it is an unsupervised VAE
  (a decoder = more code); and its sequence-level latent would take drive, arm and guitar
  timbre together, since all are constant in time. DSVAE has the same objection.

**Adding a pedal**: find its drive parameter with `gefx inspect-plugin <vst3>`, add it to
the roster with a `sweep`, run `calibrate`. It fits if its Rnonlin reaches
≥ 0.989 at the low end and ≤ 0.846 at the high end (the printed range); otherwise it gets
`unmatched` levels. Then either accept them, drop the pedal, or lower `--ref-db` MAX —
which changes every arm's levels, so do it deliberately.

Pedals evaluated on 2026-09-27:

- Added: `dm-rat` (knob `distortion`, `filter` at 0 = open, like the hardware) reaches
  0.998–0.721.
- Left out: ChowCentaur only reaches 0.930 (to be evaluated later at the levels it
  reaches, not as a training arm); TAL-Bitcrusher is not a drive (`sample_rate` is
  non-monotonic and inverted, `compand` stops at 0.931); TSE 808 is VST2-only, which
  pedalboard can't load.
- Dropped: VZtec Fuzz (Face voice, `gain` 50, knob `input_level`). Its cleanest point is
  0.9821 whatever the gain or voice, so level 1 was `unmatched`; fitting it would need
  `--ref-db` MIN ≥ 20.64 dB (≈21 dB off its flat top), which moves every arm's levels.
  It is also slightly stateful across calls despite `reset=True`.
- Plugin robustness kept from the Fuzz: the first instance of a plugin in a process can
  come out NaN from sample 8192 of every call (the Fuzz did in ~half of the processes;
  not concurrency or timing — an earlier "two instances at once" diagnosis was wrong).
  `load_arm`'s probe catches it and loads another instance, keeping the broken one alive
  in `_BROKEN` (freeing it first doesn't help). `LoadedArm.render` also raises on
  non-finite output instead of writing it.

State as of 2026-09-27: 8 arms calibrated (the 7 original + `dm-rat`). `dm-rat` is
dm-Rat v0.1.2 (github.com/davemollen/dm-Rat); it was calibrated with its Windows build
and its Linux build (`plugins/real/dm-Rat.vst3/Contents/x86_64-linux/`) has not been run
yet. Next steps on Linux:

1. `gefx inspect-plugin plugins/real/dm-Rat.vst3` must list `distortion`, `filter` and
   `volume` (names can differ between builds; `LoadedArm` fails loudly if they do).
2. Check it hits the targets at its stored levels — the two lines must agree to ~0.001
   (BYOD, calibrated on Linux, matched its targets on Windows within 0.001). If they don't,
   re-run `gefx disent calibrate` (all arms; it rewrites the levels file):
   ```bash
   python -c "
   from pathlib import Path
   from gefx.disent.arms import load_roster, load_levels
   from gefx.disent.calibrate import arm_curve, load_segments
   r = load_roster(); segs, sr = load_segments(Path('datasets/unprocessed_samples'), 8)
   print('alvo  ', [round(t, 3) for t in load_levels()['targets']])
   print('dm-rat', [round(v, 3) for v in arm_curve(r.arm('dm-rat'), segs, sr, r.arm('dm-rat').levels)[1]])
   "
   ```
3. Run `pytest` — this session's changes (read-only `tune`, `load_arm` NaN retry, no
   manual levels) were only tested on Windows, where the Keras tests can't run.
4. Render everything into a new root (the old `datasets/disent` has the 7-arm roster
   and the tone grid): `gefx disent render --output-root datasets/disent_v2`, then train.

Steps 1–3 were done on Linux on 2026-09-27: the Linux dm-Rat build exposes the same
parameters and hits the targets within 0.001, and the suite passes.

The repo's `.git/config` has `core.autocrlf=false` and `core.filemode=false`, set for the
Windows side of the dual boot (no longer used); unset them if they get in the way.
