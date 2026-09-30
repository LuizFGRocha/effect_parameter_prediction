#!/usr/bin/env bash
# Escada do POC I ao POC II, um degrau por mudanca.
#
#   B1                    o regressor do POC I, sem retreino (gefx disent retrieve)
#   poc1 + regressao      a mesma CNN e a mesma perda, retreinadas nos dados do POC II
#   poc1 + supcon         so a perda muda (e a saida: 32 dimensoes na esfera)
#   poc2 + regressao      so o tronco muda (results/disent/v2/encoder/regressao*)
#   poc2 + supcon         o encoder do relatorio (results/disent/v2/encoder/supcon*)
#
# Todos os degraus retreinados usam o mesmo amostrador, passos, taxa e sementes.
# Depois, as sondas do poc1 e a curva de orcamento: supcon por 16.000 passos nas
# duas arquiteturas, avaliado a cada 1.000, para ver se 4.000 passos bastam.
#
# Uso: bash scripts/escada_poc1.sh    (~6 h de GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RAIZ="${RAIZ:-results/disent/v2/escada}"
SEMENTES=(20260908 2 3)

echo "== poc1, random_encoder $(date)"
gefx disent train --arch poc1 --technique random_encoder --results-dir "$RAIZ/poc1"

for tecnica in regressao supcon; do
  for semente in "${SEMENTES[@]}"; do
    echo "== poc1, $tecnica, semente $semente $(date)"
    gefx disent train --arch poc1 --technique "$tecnica" --seed "$semente" \
      --results-dir "$RAIZ/poc1"
  done
done

echo "== sondas, poc1 $(date)"
gefx disent probe --results-dir "$RAIZ/poc1"

for arch in poc2 poc1; do
  echo "== orcamento, $arch $(date)"
  gefx disent train --arch "$arch" --technique supcon --steps 16000 --eval-every 1000 \
    --results-dir "$RAIZ/orcamento/$arch"
done
echo "== fim $(date)"
