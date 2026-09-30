#!/usr/bin/env bash
# Ablacao da media no tempo do tronco do encoder do POC II.
#
# Tres variantes (mean = o encoder do relatorio; max = reducao de mesmo tamanho;
# flatten = sem reducao, ~6x mais pesos) x tres degraus (random_encoder, bn_only,
# supcon), cada variante em <RAIZ>/<variante>/ com a sua escada.csv. Depois, as
# sondas dentro de cada nivel (sondas.csv), que medem o que a media promete:
# conteudo fora de z_e "de graca".
#
# A variante mean e treinada de novo, na mesma revisao das outras, em vez de
# reaproveitar results/disent/v2/encoder (commit 08b264f, busca antiga).
#
# Uso: bash scripts/ablacao_media_tempo.sh [semente]    (~2 h de GPU por semente)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SEMENTE="${1:-20260908}"
RAIZ="${RAIZ:-results/disent/v2/ablacao_tempo}"
VARIANTES=(${VARIANTES:-mean max flatten})

for variante in "${VARIANTES[@]}"; do
  echo "== $variante, semente $SEMENTE $(date)"
  gefx disent train --time-pool "$variante" --seed "$SEMENTE" \
    --technique random_encoder --technique bn_only --technique supcon \
    --results-dir "$RAIZ/$variante"
done

for variante in "${VARIANTES[@]}"; do
  echo "== sondas, $variante $(date)"
  gefx disent probe --results-dir "$RAIZ/$variante"
done
echo "== fim $(date)"
