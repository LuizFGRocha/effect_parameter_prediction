#!/usr/bin/env bash
# O desenvolvimento do POC II: so a validacao. Encoder `poc2` (a CNN do POC I +
# media no tempo + filtros (32, 64, 96, 128)), ate 20.000 passos com a taxa em
# cosseno (como o SupCon oficial), rodado inteiro; ficam os pesos do minimo do erro
# de validacao.
# Grava em results/disent/v2/validacao/. O teste fica para scripts/poc2_teste.sh.
#
# Uso: bash scripts/poc2_validacao.sh    (~11 h de GPU: ~3 h a escada de tecnicas,
# ~4 h o leave-one-arm-out, ~3 h a curva de diversidade)
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

echo "== leave-one-arm-out $(date)"
gefx disent loo
echo "== curva de diversidade $(date)"
# O ultimo ponto (todos os arms menos o byod-mxr) e a execucao do leave-one-out.
gefx disent diversity --held-out byod-mxr

echo "== sondas $(date)"
gefx disent probe
echo "== figuras $(date)"
gefx disent plots
echo "== fim $(date)"
