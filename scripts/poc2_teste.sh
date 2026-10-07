#!/usr/bin/env bash
# A passada final do POC II: o teste, uma vez, com os modelos que o
# desenvolvimento deixou (scripts/poc2_validacao.sh). Nada aqui treina.
#
# Uso: bash scripts/poc2_teste.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

gefx disent retrieve --baseline b0 --split teste
gefx disent retrieve --baseline b1 --split teste
gefx disent evaluate --split teste
gefx disent probe --split teste
gefx disent between --run supcon --run supcon_s2 --run supcon_s3
# O leave-one-out e a curva reaproveitam as execucoes da validacao.
gefx disent loo --extra-seed 2 --split teste
gefx disent diversity --held-out byod-mxr --split teste
gefx disent diversity --held-out byod-mxr --seed 2 --split teste \
  --results-dir results/disent/v2/validacao/encoder/diversidade_s2
# Os degraus da escada (scripts/poc2_escada.sh); o `poc2` e o encoder, acima.
for degrau in poc1 poc1_media poc1_media_largo; do
  gefx disent evaluate --split teste --results-dir "results/disent/v2/validacao/escada/$degrau"
done
gefx disent plots --split teste
