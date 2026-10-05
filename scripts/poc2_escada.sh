#!/usr/bin/env bash
# A escada do POC I ao POC II no protocolo atual (validacao, ate 20.000 passos com
# cosseno, pesos do minimo da validacao): um degrau por mudanca da CNN do POC I,
# cada um com 3 sementes. O degrau final, `poc2`, e o de scripts/poc2_validacao.sh
# (results/disent/v2/validacao/encoder/) e nao e retreinado aqui.
#
#   poc1              a CNN do POC I              regressao e supcon
#   poc1_media        + media no tempo            regressao e supcon
#   poc1_media_largo  + filtros (32, 64)          supcon
#   poc2              + quatro blocos             (ja em encoder/)
#
# Compare degraus com `gefx disent compare --a <degrau>/supcon ... --b ...`.
# Uso: bash scripts/poc2_escada.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

raiz=results/disent/v2/validacao/escada
for semente in 20260908 2 3; do
  for degrau in poc1 poc1_media poc1_media_largo; do
    tecnicas=(--technique supcon)
    if [ "$degrau" != poc1_media_largo ]; then tecnicas+=(--technique regressao); fi
    echo "== $degrau, semente $semente $(date)"
    gefx disent train --arch "$degrau" --seed "$semente" "${tecnicas[@]}" \
      --results-dir "$raiz/$degrau"
  done
done

for degrau in poc1 poc1_media poc1_media_largo; do
  echo "== sondas $degrau $(date)"
  gefx disent probe --results-dir "$raiz/$degrau"
done
echo "== fim $(date)"
