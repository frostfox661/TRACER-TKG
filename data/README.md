# Data

## Factual TKG splits

| Directory | Dataset | Source |
|-----------|---------|--------|
| `ICEWS14/` | ICEWS 2014 | [ICEWS](https://www.andybeger.com/icews/) event data; processed splits follow common TKG extrapolation benchmarks |
| `ICEWS18/` | ICEWS 2018 | Same |
| `ICEWS05-15/` | ICEWS 2005–2015 | Same |
| `GDELT/` | GDELT | [GDELT Project](https://www.gdeltproject.org/) |

Each folder contains `train.txt`, `valid.txt`, `test.txt`, `entity2id.txt`, `relation2id.txt`, `e-w-graph.txt`, and related files.

### Suggested citations

If you use these datasets, please cite the original sources in addition to TRACER:

- **ICEWS:** E. Boschee, J. Lautenschlager, S. O'Brien, S. Shellman, J. Starz, and M. Ward. ICEWS Coded Event Data. Harvard Dataverse, 2015.
- **GDELT:** K. G. Leetaru and P. A. Schrodt. GDELT: Global Data on Events, Location, and Tone, 1979–2012. *ISA Annual Convention*, 2013.

### Terms of use

ICEWS and GDELT are subject to their respective providers' terms. The splits bundled here are **preprocessed for research reproducibility** only. Redistribution or commercial use may require separate permission from the data owners. When in doubt, download fresh splits from the official sources and rerun `scripts/prepare_data.sh`.

**Not stored in git (generated locally):** `his_graph_for/`, `his_graph_inv/`, `his_dict/` — create with:

```bash
DATASETS=ICEWS14 bash scripts/prepare_data.sh   # one benchmark
bash scripts/prepare_data.sh                    # all four (GDELT is slow)
```

## Counterfactual training files

Aligned counterfactual quadruples for Stage II:

```
counterfactual/
├── ICEWS14/cf_train.txt
├── ICEWS18/cf_train.txt
├── ICEWS05-15/cf_train.txt
└── GDELT/cf_train.txt
```

Format: tab-separated `head relation tail time` (integer ids). Produced by `cf_gen/generate_cf.py` and time-aligned with factual `train.txt`. Regenerating CF data requires your own LLM API or local model (see `cf_gen/README.md`).
