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
gefx disent probe --split catalog
gefx disent between --run supcon --run supcon_s2 --run supcon_s3
gefx disent plots --split teste
# Depois que o leave-one-out e a curva de diversidade tiverem rodado na validacao
# (as execucoes sao reaproveitadas; sem elas, estes comandos treinariam):
#   gefx disent loo --split teste
#   gefx disent diversity --held-out byod-mxr --split teste
