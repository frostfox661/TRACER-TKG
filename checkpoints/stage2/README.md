# Stage II checkpoints (main experiment)

TRACER Stage II weights for the paper's main recipe. Files are stored with **Git LFS** (see root `.gitattributes`); run `git lfs pull` after clone if weights are pointer stubs. Each file is a **test-ready** checkpoint (load with `--test`; no training required).

| File | Dataset | Seed |
|------|---------|------|
| `ICEWS14_seed{42,17,29}.pt` | ICEWS14 | 42 / 17 / 29 |
| `ICEWS18_seed{42,17,29}.pt` | ICEWS18 | 42 / 17 / 29 |
| `ICEWS05-15_seed{42,17,29}.pt` | ICEWS05-15 | 42 / 17 / 29 |
| `GDELT_seed{42,17,29}.pt` | GDELT | 42 / 17 / 29 |

Hyperparameters match Stage II training: `lr=5e-4`, 8 epochs, `cf_rank_weight=0.08`, `cf_consistency_weight=0.03`, `cf_residual_weight=0.25`, `cf_fuse_alpha=0`, `cf_loss_weight=0`, `cf_aux_weight=0`.

`reference_metrics.json` lists expected **filter_all** MRR / Hits@1,3,10 for automated verification (`scripts/verify_pretrained.py`). ICEWS18 and GDELT currently include seed 42 only; ICEWS14 and ICEWS05-15 include all three seeds.

Quick test (after `bash scripts/prepare_data.sh`):

```bash
GPU=0 SEED=42 bash scripts/run_eval_pretrained.sh ICEWS14
```
