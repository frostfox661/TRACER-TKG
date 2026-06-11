#!/usr/bin/env python3
"""Batch-evaluate bundled Stage II checkpoints against reference metrics."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "checkpoints" / "stage2" / "reference_metrics.json"
METRIC_RE = re.compile(
    r"\(all_filter\) MRR, Hits@ \(1,3,5\):([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)"
)
DATASETS = ("ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT")
DEFAULT_SEEDS = (42, 17, 29)


def load_reference() -> dict:
    with REF_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_eval(dataset: str, seed: int, gpu: int, python: str) -> dict[str, float]:
    import os

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHON": python}
    env["GPU"] = str(gpu)
    env["SEED"] = str(seed)
    env["RESULT_FILE"] = str(ROOT / "result" / f"_verify_{dataset}_seed{seed}.csv")
    if os.environ.get("DATA_ROOT"):
        env["DATA_ROOT"] = os.environ["DATA_ROOT"]
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_eval_pretrained.sh"), dataset],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    match = METRIC_RE.search(out)
    if proc.returncode != 0 or not match:
        tail = "\n".join(out.strip().splitlines()[-25:])
        raise RuntimeError(f"{dataset} seed={seed} failed:\n{tail}")
    return {
        "filter_all_MRR": float(match.group(1)),
        "filter_all_H@1": float(match.group(2)),
        "filter_all_H@3": float(match.group(3)),
        "filter_all_H@10": float(match.group(4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    reference = load_reference()
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    seeds = [args.seed] if args.seed is not None else list(DEFAULT_SEEDS)

    print(f"Reference: {REF_PATH}")
    print(f"Python:    {args.python}")
    print()

    ok = True
    compared = 0
    for dataset in datasets:
        for seed in seeds:
            ckpt = ROOT / "checkpoints" / "stage2" / f"{dataset}_seed{seed}.pt"
            if not ckpt.is_file():
                print(f"SKIP missing checkpoint: {ckpt.name}")
                continue
            got = run_eval(dataset, seed, args.gpu, args.python)
            ref = reference.get(dataset, {}).get(str(seed))
            print(f"=== {dataset} seed={seed} ===")
            for key, value in got.items():
                line = f"  {key}: {value:.6f}"
                if ref and key in ref:
                    compared += 1
                    diff = abs(value - ref[key])
                    mark = "OK" if diff < args.tolerance else f"DIFF {diff:.2e}"
                    line += f"  (ref {ref[key]:.6f}, {mark})"
                    if diff >= args.tolerance:
                        ok = False
                print(line)
            print()

    if compared == 0:
        print("No reference entries matched the selected runs.")
    elif ok:
        print(f"All {compared} metric comparisons match within {args.tolerance}.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
