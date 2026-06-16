# TRACER

**TRACER** (*Trajectory-Aligned Counterfactual-Enhanced Regularization*) — official implementation for temporal knowledge graph (TKG) extrapolation with counterfactual-aware training.

## Repository layout

```
TRACER/
├── README.md
├── LICENSE
├── requirements.txt
├── data/                         # Raw TKG splits + counterfactual training files
│   ├── ICEWS14/ … GDELT/
│   ├── counterfactual/           # cf_train.txt per dataset (from cf_gen/)
│   └── get_his_subg.py           # Build history subgraph caches
├── cf_gen/                       # Offline counterfactual graph generation (IV.A)
├── train/                        # Recurrent R-GCN encoder + TRACER losses
├── rgcn/                         # Graph encoder / data utilities
├── scripts/                      # Training + evaluation entry points
├── checkpoints/
│   ├── stage1/                   # Stage I factual weights (for Stage II init)
│   └── stage2/                   # Stage II main-experiment weights (for test-only eval)
└── models/                       # Outputs from re-training (gitignored)
```

## Clone

Checkpoints (~600 MB) are tracked with **Git LFS**. After installing [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git clone https://github.com/frostfox661/TRACER-TKG.git
cd TRACER-TKG
```

If weights are missing after clone, run `git lfs pull` inside the repo.

## Environment

Tested with **Python 3.9** and CUDA 11.x (PyTorch 1.12 + DGL 1.1). Setup:

```bash
conda create -n tracer python=3.9 -y
conda activate tracer

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install dgl==1.1.0+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

`requirements.txt` pins the remaining direct imports (`numpy`, `pandas`, `scipy`, `rdflib`, `tqdm`, `networkx`). If DGL fails to load CUDA libraries, try:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

For other CUDA / driver versions, pick matching PyTorch and DGL wheels from [pytorch.org](https://pytorch.org) and [dgl.ai](https://www.dgl.ai/pages/start.html). Regenerating counterfactual data needs extra packages — see `cf_gen/requirements.txt`.

## 1. Prepare factual data caches

Raw `train/valid/test` are already under `data/<DATASET>/`. **Before any training or evaluation**, build static word graphs and per-timestamp query subgraphs:

```bash
# Quick path for a single benchmark (recommended for ICEWS14 eval, ~1 min):
DATASETS=ICEWS14 bash scripts/prepare_data.sh

# All four datasets (GDELT is large and may take hours):
bash scripts/prepare_data.sh
```

This runs `ent2word.py` (where present) and `get_his_subg.py`, creating `his_graph_for/`, `his_graph_inv/`, `his_dict/` inside each dataset folder. Override the dataset list with `DATASETS` (space-separated names), e.g. `DATASETS="ICEWS14 ICEWS18" bash scripts/prepare_data.sh`.

Dataset sources and citation notes: [data/README.md](data/README.md).

## 2. Quick reproduction (pretrained Stage II weights)

Bundled **Stage II** checkpoints live under `checkpoints/stage2/` (12 files: 4 datasets × seeds 42, 17, 29). Evaluation uses **factual history only**; counterfactual graphs are not used at inference time.

### Single dataset

```bash
GPU=0 SEED=42 bash scripts/run_eval_pretrained.sh ICEWS14
```

Supported datasets: `ICEWS14`, `ICEWS18`, `ICEWS05-15`, `GDELT`.

The script loads `checkpoints/stage2/<DATASET>_seed<SEED>.pt`, runs the test split, and prints **filter_all** MRR / Hits@1,3,10. Example output line:

```
(all_filter) MRR, Hits@ (1,3,5):0.497278, 0.385362, 0.558947, 0.712047
```

### Verify all bundled checkpoints

Compares against `checkpoints/stage2/reference_metrics.json` (paper numbers for ICEWS14 / ICEWS05-15 seeds 42,17,29; ICEWS18 / GDELT seed 42):

```bash
python scripts/verify_pretrained.py --gpu 0
# or one dataset:
python scripts/verify_pretrained.py --gpu 0 --dataset ICEWS14 --seed 42
```

### Manual invocation

```bash
export PYTHONPATH="$(pwd)"
python train/main.py -d ICEWS14 --data-root ./data --test --gpu 0 \
  --model-state-file checkpoints/stage2/ICEWS14_seed42.pt \
  --train-history-len 7 --test-history-len 7 \
  --add-static-graph --use-cl --pre-type all --use-cf \
  --cf-train-path data/counterfactual/ICEWS14/cf_train.txt \
  ...  # see scripts/run_eval_pretrained.sh for the full flag set
```

See [checkpoints/stage2/README.md](checkpoints/stage2/README.md) for checkpoint naming and reference metrics.

## 3. (Optional) Regenerate counterfactual data

Bundled counterfactual files live at `data/counterfactual/<DATASET>/cf_train.txt`. To regenerate from scratch:

```bash
cd cf_gen
pip install -r requirements.txt
cp llm.local.json.example llm.local.json   # configure your LLM endpoint
python generate_cf.py -m <model_alias> --data-dir ../data/ICEWS14 -n 10 --test-mode
```

See [cf_gen/README.md](cf_gen/README.md).

## 4. Two-stage training (full reproduction)

### Stage I — factual encoder

Trains the temporal encoder on factual history only (no counterfactual branch). A checkpoint for seed 42 is **already included** under `checkpoints/stage1/`.

```bash
GPU=0 SEED=42 bash scripts/run_stage1_factual.sh ICEWS14
```

### Stage II — TRACER counterfactual-aware training

Loads Stage I weights, enables the counterfactual branch and TRACER regularizers (ranking + consistency + residual injection).

```bash
GPU=0 SEED=42 bash scripts/run_stage2_tracer.sh ICEWS14
```

| Stage | Script | Key flags |
|-------|--------|-----------|
| I | `run_stage1_factual.sh` | factual only, `lr=1e-3`, 15 epochs |
| II | `run_stage2_tracer.sh` | `--init-checkpoint`, `--use-cf`, `lr=5e-4`, 8 epochs |

Stage II default hyperparameters: `cf_rank_weight=0.08`, `cf_rank_margin=0.05`, `cf_consistency_weight=0.03`, `cf_residual_weight=0.25`, `cf_quality_tau=0.03`, `cf_quality_beta=20`, `cf_fuse_alpha=0`, `cf_loss_weight=0`, `cf_aux_weight=0`.

## License

This repository is released under the [MIT License](LICENSE). Third-party datasets (ICEWS, GDELT) remain subject to their original terms — see [data/README.md](data/README.md).

## Citation

If you use this code, please cite the TRACER paper (bibtex to be added upon publication).
