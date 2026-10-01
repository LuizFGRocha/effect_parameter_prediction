#!/usr/bin/env bash
# Os resultados do POC II no protocolo com validacao: encoder `poc2` (a CNN do
# POC I + media no tempo + filtros (32, 64, 96, 128)), ate 20.000 passos, parada
# pelo erro de validacao com paciencia de 8 avaliacoes (4.000 passos), pesos do
# minimo. Grava em results/disent/v2/validacao/.
#
# Uso: bash scripts/poc2_validacao.sh    (~6 h de GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "== baselines $(date)"
gefx disent retrieve --baseline b0
gefx disent retrieve --baseline b1

for semente in 20260908 2 3; do
  echo "== encoder e controles, semente $semente $(date)"
  gefx disent train --seed "$semente" \
    --technique random_encoder --technique bn_only --technique supcon --technique regressao
done

echo "== sondas $(date)"
gefx disent probe
echo "== entre niveis $(date)"
gefx disent between --run supcon --run supcon_s2 --run supcon_s3
echo "== figuras $(date)"
gefx disent plots
echo "== fim $(date)"
