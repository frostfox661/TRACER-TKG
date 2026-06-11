#!/usr/bin/env bash
# Stage I: factual encoder pretraining (no counterfactual branch).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

DATASET="${1:-ICEWS14}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-0.001}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
OUT="$ROOT/checkpoints/stage1/${DATASET}_seed${SEED}.pt"
mkdir -p "$(dirname "$OUT")" "$ROOT/models"

python -u train/main.py \
  -d "$DATASET" \
  --data-root "$DATA_ROOT" \
  --train-history-len 7 \
  --test-history-len 7 \
  --dilate-len 1 \
  --n-layers 2 \
  --evaluate-every 1 \
  --gpu "$GPU" \
  --n-hidden 200 \
  --self-loop \
  --decoder convtranse \
  --encoder uvrgcn \
  --layer-norm \
  --weight 0.5 \
  --entity-prediction \
  --angle 10 \
  --discount 1 \
  --pre-weight 0.9 \
  --pre-type all \
  --add-static-graph \
  --temperature 0.03 \
  --use-cl \
  --lr "$LR" \
  --n-epochs "$EPOCHS" \
  --early-stop-patience 5 \
  --seed "$SEED" \
  --model-state-file "$OUT"

echo "Stage I checkpoint: $OUT"
