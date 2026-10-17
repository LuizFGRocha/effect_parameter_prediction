#!/usr/bin/env bash
# O desemaranhamento em dois blocos, z_e + z_c, em degraus a partir do encoder do
# relatorio (`poc2`, results/disent/v2/validacao/encoder/, nao retreinado aqui),
# cada um com 3 sementes, na validacao:
#
#   encoder/            poc2: batch P x K por nivel, so z_e
#   duas_cabecas/trecho    + batch P x K por trecho (--batch-by content), so z_e
#   duas_cabecas/conteudo  + cabeca z_c (--content-dim 32); e o encoder nao treinado
#
# Depois, as sondas (z_e e z_c) e as diferencas pareadas entre os degraus.
# Uso: bash scripts/poc2_duas_cabecas.sh
# DADOS e RAIZ trocam o dataset e a pasta de resultados.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

dados=${DADOS:-datasets/disent_v2}
raiz=${RAIZ:-results/disent/v2/validacao}
saida=$raiz/duas_cabecas

for semente in 20260908 2 3; do
  echo "== trecho, semente $semente $(date)"
  gefx disent train --output-root "$dados" --seed "$semente" --technique supcon \
    --batch-by content --results-dir "$saida/trecho"
  tecnicas=(--technique supcon)
  if [ "$semente" = 20260908 ]; then tecnicas+=(--technique random_encoder); fi
  echo "== conteudo, semente $semente $(date)"
  gefx disent train --output-root "$dados" --seed "$semente" "${tecnicas[@]}" \
    --batch-by content --content-dim 32 --results-dir "$saida/conteudo"
done

for degrau in trecho conteudo; do
  echo "== sondas $degrau $(date)"
  gefx disent probe --output-root "$dados" --results-dir "$saida/$degrau"
done

sementes() { for s in supcon supcon_s2 supcon_s3; do echo "--$1 $2/$s"; done; }
echo "== trecho - poc2"
gefx disent compare $(sementes a "$saida/trecho") $(sementes b "$raiz/encoder")
echo "== conteudo - trecho"
gefx disent compare $(sementes a "$saida/conteudo") $(sementes b "$saida/trecho")
echo "== conteudo - poc2"
gefx disent compare $(sementes a "$saida/conteudo") $(sementes b "$raiz/encoder")
echo "== fim $(date)"
