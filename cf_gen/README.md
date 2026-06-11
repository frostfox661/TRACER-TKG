# Counterfactual graph generation (IV.A)

This directory implements **IV.A Semantically Constrained Counterfactual History Construction** from the TRACER paper: an LLM builds `cf_train.txt` aligned line-by-line with `train.txt` under **pivot gating**, **high-contrast intervention**, and **short-horizon rollout**.

## Layout

```
cf_gen/
├── generate_cf.py            # Main entry point (V5 pipeline)
├── llm.local.json.example    # LLM config template (copy and fill in)
├── prompts/                  # Prompt documentation (V5)
└── cf_runs/                  # Output runs (gitignored)
```

Raw datasets live at `../data/<DATASET>/`.

## Environment

```bash
cd cf_gen
pip install -r requirements.txt
```

## Configure the LLM

Copy the template and add your model settings (**do not commit** `llm.local.json`; it is in `.gitignore`):

```bash
cp llm.local.json.example llm.local.json
```

### Option A: local vLLM

1. Prepare an instruct-style causal LM (Hugging Face repo ID or local path).
2. Add an alias under `local_models` in `llm.local.json`, for example:

```json
{
  "local_models": {
    "Qwen2.5-7B-Instruct": {
      "path": "/path/to/model-or-hf-repo"
    }
  }
}
```

3. Run:

```bash
python generate_cf.py -m Qwen2.5-7B-Instruct --data-dir ../data/ICEWS14 -n 10 --test-mode
```

You can skip the config file and pass a path directly (`-m` is only used for output folder naming):

```bash
python generate_cf.py --model-path /path/to/model -m my-model --data-dir ../data/ICEWS14 -n 10
```

Or set `CFGEN_MODEL_PATH` / `CFGEN_MODEL_ALIAS`.

### Option B: OpenAI-compatible API

1. Fill `base_url`, `model`, and `api_key` under `api_profiles` in `llm.local.json`.
2. Run:

```bash
python generate_cf.py -m your-api-profile --data-dir ../data/ICEWS14
```

`api_key` can also come from the `CFGEN_API_KEY` environment variable.

If the same alias appears in both `local_models` and `api_profiles`, pass `--backend vllm` or `--backend api`.

### Optional environment variables

| Variable | Meaning |
|----------|---------|
| `CFGEN_MODEL_ALIAS` | Model alias (from `llm.local.json`) |
| `CFGEN_MODEL_PATH` | Local model path (direct vLLM) |
| `CFGEN_BACKEND` | `vllm` or `api` |
| `CFGEN_LLM_CONFIG` | Config file path (default `llm.local.json`) |
| `CFGEN_API_KEY` | API key (alternative to config file) |

vLLM: `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DTYPE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, etc.

## Usage

From `cf_gen/` (**after LLM configuration**):

```bash
# Smoke test
python generate_cf.py -m <alias> --data-dir ../data/ICEWS14 -n 10 --test-mode

# Full run
python generate_cf.py -m <alias> --data-dir ../data/ICEWS14

# Other datasets
python generate_cf.py -m <alias> --data-dir ../data/ICEWS18
python generate_cf.py -m <alias> --data-dir ../data/ICEWS05-15
python generate_cf.py -m <alias> --data-dir ../data/GDELT

# Resume
python generate_cf.py -m <alias> --data-dir ../data/ICEWS14 \
  --resume-dir cf_runs/<previous_run_dir>
```

## Outputs

Each run creates a subfolder under `cf_runs/`: `{model_alias}_{dataset}_{timestamp}_v5/`

| File | Description |
|------|-------------|
| `cf_train.txt` | Counterfactual quadruples; same line count as `train.txt` |
| `meta.json` | Model, timing, `v5_quality`, etc. |
| `state.json` | Checkpoint for resume |

## V5 pipeline summary

1. **Pivot gate**: `YES` / `NO`; if `NO`, copy the factual line.
2. **Local history**: co-occurring events for the same `(s,o)` before intervention.
3. **High-contrast intervention**: `INTERVENTION||s||r||o`.
4. **Short-horizon rollout**: counterfactual rewrites for related slots in the time window.

Tunable constants at the top of `generate_cf.py`: `CF_V5_ROLLOUT_LR`, `CF_V5_ROLLOUT_MAX_SLOTS`, `CF_V4_HIST_TOP`, etc.
