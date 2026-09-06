#!/usr/bin/env bash
# Varredura das quatro features + comparacao entre elas.
# Escreve em <RESULTS_BASE>/<FEATURE>/, que e o layout que `gefx compare-features`
# e `gefx cross-impl eval --models-root` esperam.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-datasets/default}"
RESULTS_BASE="${RESULTS_BASE:-results/feature_runs}"
CONFIG="${CONFIG:-experiments/base.yaml}"

FEATURES=("MFCC40" "Spec" "Chroma" "GFCC40")

mkdir -p "$RESULTS_BASE"

for feature in "${FEATURES[@]}"; do
  results_root="$RESULTS_BASE/${feature}"
  echo "== feature: $feature -> $results_root"
  gefx train \
    --config "$CONFIG" \
    --dataset-root "$DATASET_ROOT" \
    --feature "$feature" \
    --results-root "$results_root"
  gefx evaluate --results-root "$results_root"
done

gefx compare-features --results-base "$RESULTS_BASE"
