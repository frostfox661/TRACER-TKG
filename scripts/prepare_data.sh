#!/usr/bin/env bash
# Build static word graphs and query-history caches required by training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/data"

DATASETS=(ICEWS14 ICEWS18 ICEWS05-15 GDELT)

for ds in "${DATASETS[@]}"; do
  echo "=== $ds ==="
  if [[ -f "$ds/ent2word.py" ]]; then
    (cd "$ds" && python ent2word.py)
  fi
done

python get_his_subg.py

echo "Done. Each dataset folder should now contain his_graph_for/, his_graph_inv/, his_dict/."
