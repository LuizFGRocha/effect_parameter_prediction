#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="datasets/default"
RESULTS_BASE="results/feature_runs"
EPOCHS=70
BATCH_SIZE=32
TEST_SIZE=0.20
SPLIT_SEED=42

FEATURES=("MFCC40" "Spec" "Chroma" "GFCC40")

mkdir -p "$RESULTS_BASE"

for feature in "${FEATURES[@]}"; do
  results_root="$RESULTS_BASE/${feature}"
  echo "Running feature: $feature -> $results_root"
  python src/model.py \
    --dataset-root "$DATASET_ROOT" \
    --feature "$feature" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --test-size "$TEST_SIZE" \
    --split-seed "$SPLIT_SEED" \
    --results-root "$results_root"
  python src/results_evaluation.py \
    --results-root "$results_root"
done
