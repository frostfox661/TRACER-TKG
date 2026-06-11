# Stage I checkpoints

Factual encoder weights used as initialization for Stage II (`--init-checkpoint`). Stored with **Git LFS** — see root `.gitattributes`.

| File | Dataset | Seed | Notes |
|------|---------|------|-------|
| `ICEWS14_seed42.pt` | ICEWS14 | 42 | Factual pretrain, no CF branch |
| `ICEWS18_seed42.pt` | ICEWS18 | 42 | Same hyperparameters as paper |
| `ICEWS05-15_seed42.pt` | ICEWS05-15 | 42 | Same |
| `GDELT_seed42.pt` | GDELT | 42 | Same |

To retrain Stage I:

```bash
bash scripts/run_stage1_factual.sh ICEWS14
```

Output overwrites the matching `checkpoints/stage1/<DATASET>_seed<SEED>.pt`.

Stage II (main experiment) weights for evaluation are under [../stage2/README.md](../stage2/README.md).
