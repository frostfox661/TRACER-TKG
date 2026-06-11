#!/usr/bin/env bash
# Evaluate a bundled Stage II checkpoint (test only, no training).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PYTHON="${PYTHON:-python}"

DATASET="${1:-ICEWS14}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CF_TRAIN="${CF_TRAIN:-$ROOT/data/counterfactual/$DATASET/cf_train.txt}"
CKPT="${CKPT:-$ROOT/checkpoints/stage2/${DATASET}_seed${SEED}.pt}"
RESULT_FILE="${RESULT_FILE:-$ROOT/result/eval_${DATASET}_seed${SEED}.csv}"

mkdir -p "$(dirname "$RESULT_FILE")"
[[ -f "$CF_TRAIN" ]] || { echo "Missing counterfactual file: $CF_TRAIN"; exit 1; }
[[ -f "$CKPT" ]] || { echo "Missing Stage II checkpoint: $CKPT"; exit 1; }
[[ -f "$DATA_ROOT/$DATASET/his_dict/train_s_r.npy" ]] || {
  echo "Missing history caches under $DATA_ROOT/$DATASET/his_dict/"
  echo "Run: bash scripts/prepare_data.sh"
  exit 1
}

echo "Dataset:    $DATASET"
echo "Seed:       $SEED"
echo "Checkpoint: $CKPT"
echo ""

"$PYTHON" -u train/main.py \
  -d "$DATASET" \
  --data-root "$DATA_ROOT" \
  --test \
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
  --lr 0.0005 \
  --n-epochs 8 \
  --seed "$SEED" \
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
  --model-state-file "$CKPT" \
  --result-file "$RESULT_FILE"
