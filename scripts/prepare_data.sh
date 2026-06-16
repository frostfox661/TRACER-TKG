#!/usr/bin/env bash
# Build static word graphs and query-history caches required by training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/data"

if [[ -n "${DATASETS:-}" ]]; then
  # Example: DATASETS=ICEWS14 bash scripts/prepare_data.sh
  read -r -a DATASETS_ARR <<< "${DATASETS}"
else
  DATASETS_ARR=(ICEWS14 ICEWS18 ICEWS05-15 GDELT)
fi

for ds in "${DATASETS_ARR[@]}"; do
  echo "=== $ds ==="
  if [[ -f "$ds/ent2word.py" ]]; then
    (cd "$ds" && python ent2word.py)
  fi
done

python get_his_subg.py --datasets "${DATASETS_ARR[@]}"

echo "Done. Each dataset folder should now contain his_graph_for/, his_graph_inv/, his_dict/."
