#!/usr/bin/env bash
# Stage II: TRACER counterfactual-aware training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

DATASET="${1:-ICEWS14}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-8}"
LR="${LR:-0.0005}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CF_TRAIN="${CF_TRAIN:-$ROOT/data/counterfactual/$DATASET/cf_train.txt}"
INIT_CKPT="${INIT_CKPT:-$ROOT/checkpoints/stage1/${DATASET}_seed${SEED}.pt}"
OUT="$ROOT/models/tracer_${DATASET}_seed${SEED}.pt"

mkdir -p "$(dirname "$OUT")"
[[ -f "$CF_TRAIN" ]] || { echo "Missing counterfactual file: $CF_TRAIN"; exit 1; }
[[ -f "$INIT_CKPT" ]] || { echo "Missing Stage I checkpoint: $INIT_CKPT"; exit 1; }

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
  --init-checkpoint "$INIT_CKPT" \
  --use-cf \
  --cf-train-path "$CF_TRAIN" \
  --cf-omega 1.0 \
  --cf-loss-weight 0.0 \
  --cf-aux-weight 0.0 \
  --cf-fuse-alpha 0.0 \
  --cf-rank-weight 0.08 \
  --cf-rank-margin 0.05 \
  --cf-consistency-weight 0.03 \
  --cf-residual-weight 0.25 \
  --cf-quality-tau 0.03 \
  --cf-quality-beta 20.0 \
  --model-state-file "$OUT"

echo "Stage II checkpoint: $OUT"
