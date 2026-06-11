import argparse
import json
import os
import random
import re
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Iterable
from collections import Counter, defaultdict

from tqdm import tqdm
from vllm import LLM, SamplingParams

CF_REL_CAND_K: int = 80
CF_ENT_CAND_K: int = 120
CF_ENT_NEI_TOP: int = 80
CF_V4_HIST_TOP: int = 12
CF_V5_ROLLOUT_LR: int = 2
CF_V5_ROLLOUT_MAX_SLOTS: int = 3

CF_GEN_ROOT = Path(__file__).resolve().parent
TRACER_ROOT = CF_GEN_ROOT.parent
DATA_DIR = TRACER_ROOT / "data" / "ICEWS14"
TRAIN_FILE = DATA_DIR / "train.txt"
ENTITY2ID_FILE = DATA_DIR / "entity2id.txt"
RELATION2ID_FILE = DATA_DIR / "relation2id.txt"
DEFAULT_RUNS_PARENT = CF_GEN_ROOT / "cf_runs"
REPO_ROOT = CF_GEN_ROOT
DEFAULT_LLM_CONFIG_PATH = REPO_ROOT / "llm.local.json"


class LlmSettings:
    __slots__ = ("backend", "alias", "model_path", "api_url", "api_key")

    def __init__(
        self,
        backend: str,
        alias: str,
        model_path: str,
        api_url: str | None = None,
        api_key: str | None = None,
    ):
        self.backend = backend
        self.alias = alias
        self.model_path = model_path
        self.api_url = api_url
        self.api_key = api_key


def _load_llm_config(config_path: Path) -> dict:
    if not config_path.is_file():
        return {"local_models": {}, "api_profiles": {}}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid LLM config (expected JSON object): {config_path}")
    data.setdefault("local_models", {})
    data.setdefault("api_profiles", {})
    return data


def resolve_llm_settings(
    model_alias: str | None,
    model_path_override: str | None,
    backend_override: str | None,
    config_path: Path,
) -> LlmSettings:
    alias = model_alias or os.environ.get("CFGEN_MODEL_ALIAS")
    path_override = model_path_override or os.environ.get("CFGEN_MODEL_PATH")
    backend_override = backend_override or os.environ.get("CFGEN_BACKEND")

    if path_override:
        use_alias = alias or Path(path_override).name or "local"
        return LlmSettings("vllm", use_alias, path_override)

    if not alias:
        raise ValueError(
            "Specify a model alias via `-m/--model-alias` or CFGEN_MODEL_ALIAS; "
            f"or use `--model-path` / CFGEN_MODEL_PATH for a local model. "
            f"See template at {REPO_ROOT / 'llm.local.json.example'}"
        )

    cfg = _load_llm_config(config_path)
    local_models = cfg.get("local_models") or {}
    api_profiles = cfg.get("api_profiles") or {}

    in_local = alias in local_models
    in_api = alias in api_profiles

    if backend_override == "vllm":
        if not in_local:
            raise ValueError(
                f"Alias {alias!r} not found under local_models in {config_path}. "
                f"Copy llm.local.json.example and set the local model path."
            )
        entry = local_models[alias]
        path = entry.get("path") if isinstance(entry, dict) else entry
        if not path:
            raise ValueError(f"local_models[{alias!r}] is missing the path field")
        return LlmSettings("vllm", alias, str(path))

    if backend_override == "api":
        if not in_api:
            raise ValueError(
                f"Alias {alias!r} not found under api_profiles in {config_path}. "
                f"Copy llm.local.json.example and fill in the API profile."
            )
        prof = api_profiles[alias]
        return _llm_settings_from_api_profile(alias, prof)

    if in_api and not in_local:
        return _llm_settings_from_api_profile(alias, api_profiles[alias])
    if in_local and not in_api:
        entry = local_models[alias]
        path = entry.get("path") if isinstance(entry, dict) else entry
        if not path:
            raise ValueError(f"local_models[{alias!r}] is missing the path field")
        return LlmSettings("vllm", alias, str(path))
    if in_local and in_api:
        raise ValueError(
            f"Alias {alias!r} appears in both local_models and api_profiles; "
            f"pass `--backend vllm` or `--backend api` explicitly."
        )

    raise ValueError(
        f"Model alias {alias!r} not found in {config_path}. "
        f"Copy llm.local.json.example to llm.local.json and configure it; "
        f"or use `--model-path` / CFGEN_MODEL_PATH for local vLLM."
    )


def _llm_settings_from_api_profile(alias: str, prof: dict) -> LlmSettings:
    if not isinstance(prof, dict):
        raise ValueError(f"api_profiles[{alias!r}] must be an object")
    base_url = str(prof.get("base_url", "")).rstrip("/")
    payload_model = prof.get("model") or prof.get("payload_model")
    api_key = prof.get("api_key") or os.environ.get("CFGEN_API_KEY")
    if not base_url or not payload_model:
        raise ValueError(f"api_profiles[{alias!r}] must include base_url and model (or payload_model)")
    if not api_key:
        raise ValueError(
            f"api_profiles[{alias!r}] is missing api_key; set it in config or CFGEN_API_KEY"
        )
    return LlmSettings("api", alias, str(payload_model), base_url, str(api_key))


def load_mapping(filepath: str) -> tuple[dict[int, str], dict[str, int]]:
    mapping = {}
    reverse_mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                name, idx = parts[0], int(parts[1])
                mapping[idx] = name
                reverse_mapping[name] = idx
    return mapping, reverse_mapping

def _parse_train_line_hrot(line: str) -> tuple[int, int, int, int] | None:
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None

ID2ENT, ENT2ID = load_mapping(str(ENTITY2ID_FILE))
ID2REL, REL2ID = load_mapping(str(RELATION2ID_FILE))

_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")

def _norm_symbolic_name(s: str) -> str:
    t = _NON_ALNUM_RE.sub("_", s.strip())
    t = _MULTI_UNDERSCORE_RE.sub("_", t).strip("_")
    return t.lower()

def _build_norm_lookup(name2id: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, idx in name2id.items():
        k = _norm_symbolic_name(name)
        if k and k not in out:
            out[k] = idx
    return out

ENT2ID_NORM = _build_norm_lookup(ENT2ID)
ENT2ID_NORM_KEYS = list(ENT2ID_NORM.keys())

REL2ID_NORM = _build_norm_lookup(REL2ID)

def init_cf_dataset(data_dir: str | Path) -> Path:
    global DATA_DIR, TRAIN_FILE, ENTITY2ID_FILE, RELATION2ID_FILE
    global ID2ENT, ENT2ID, ID2REL, REL2ID, ENT2ID_NORM, ENT2ID_NORM_KEYS, REL2ID_NORM

    DATA_DIR = Path(data_dir).expanduser().resolve()
    TRAIN_FILE = DATA_DIR / "train.txt"
    ENTITY2ID_FILE = DATA_DIR / "entity2id.txt"
    RELATION2ID_FILE = DATA_DIR / "relation2id.txt"
    for p in (TRAIN_FILE, ENTITY2ID_FILE, RELATION2ID_FILE):
        if not p.is_file():
            raise FileNotFoundError(f"Data file not found: {p}")
    ID2ENT, ENT2ID = load_mapping(str(ENTITY2ID_FILE))
    ID2REL, REL2ID = load_mapping(str(RELATION2ID_FILE))
    ENT2ID_NORM = _build_norm_lookup(ENT2ID)
    ENT2ID_NORM_KEYS = list(ENT2ID_NORM.keys())
    REL2ID_NORM = _build_norm_lookup(REL2ID)
    return TRAIN_FILE

def _fuzzy_ent_id(query: str, threshold: int) -> tuple[int, str, float] | None:
    q = _norm_symbolic_name(query)
    if not q:
        return None
    try:
        from rapidfuzz import fuzz, process  # type: ignore
    except Exception as e:
        raise RuntimeError("rapidfuzz is not installed: run `pip install rapidfuzz` before using --ent-fuzzy.") from e

    hit = process.extractOne(q, ENT2ID_NORM_KEYS, scorer=fuzz.WRatio)
    if not hit:
        return None
    matched_norm, score, _idx = hit
    if score < threshold:
        return None
    ent_id = ENT2ID_NORM.get(matched_norm)
    if ent_id is None:
        return None
    return ent_id, matched_norm, float(score)

def _fuzzy_ent_best(query: str) -> tuple[str, float] | None:
    q = _norm_symbolic_name(query)
    if not q:
        return None
    try:
        from rapidfuzz import fuzz, process  # type: ignore
    except Exception:
        return None
    hit = process.extractOne(q, ENT2ID_NORM_KEYS, scorer=fuzz.WRatio)
    if not hit:
        return None
    matched_norm, score, _idx = hit
    return str(matched_norm), float(score)

_LEGACY_PROMPT_TEMPLATE = """As an international-relations expert, perform counterfactual reasoning under the CAMEO event coding scheme.
The factual event is: entity [{s_name}] took action [{r_name}] toward entity [{o_name}].

Construct a plausible alternate branch: assume [{s_name}] shifts diplomatic strategy (e.g., from dialogue to conflict, or from confrontation to cooperation).
Output only the revised relation/action name using standard political/military/diplomatic vocabulary. No extra explanation."""

_MULTI_BRANCH_PROMPT_TEMPLATE = """As an international-relations expert, perform counterfactual reasoning under the CAMEO event coding scheme.
In the **real world**, subject [{s_name}] took action [{r_name}] toward object [{o_name}].

Imagine **multiple** (2–4 suggested) **distinct** yet politically and historically **plausible** alternate trajectories. Notes:
1) Branches must differ clearly; avoid repetitive phrasing;
2) Counterfactuals may change **relation/action**, and **subject** or **object** may also change (alliances, regime change, third-party involvement, misidentification, strategy shifts);
3) Use concise, formal actor names consistent with IR terminology; reuse original subject/object when unchanged.

**Key constraints (for valid output)**:
- Relation must be chosen exactly from the candidate set below (case and underscores preserved); do not invent new relation names;
- If subject/object change, use exact dataset entity strings; when unsure, keep the original subject/object.

[Relation candidates (must choose one)]:
{relation_candidates}

[Entity candidates (if subject/object change, choose exactly one; strings must match)]:
{entity_candidates}

**Output requirements (strict)**:
- Output only counterfactual **triple lines**; no numbering, titles, explanation, or Markdown;
- **One branch per line**, format: `subject||relation||object` (two `||` separators; no `||` inside fields);
- Relation must be an exact string from the candidate set above."""

_TCC_POLARITY_V3_TEMPLATE = """As a senior geopolitical intelligence analyst, use Trace–Correlate–Correct reasoning to infer a **polarity-reversed** counterfactual relation for the event below.

[Trace]
On the factual timeline, entity [{s_name}] took action [{r_name}] toward entity [{o_name}].

[Correlate]
Internally assess the geopolitical context and main tensions between [{s_name}] and [{o_name}] (do not write the analysis).

[Correct]
For a coherent alternate relation with **opposite polarity** to the factual action:
1) If [{r_name}] is cooperative, routine diplomacy, or neutral (consultation, statement, visit, negotiation intent, etc.), output a **confrontational, coercive, downgrade, or escalation** action.
2) If [{r_name}] is conflict, sanctions, coercion, or force, output **concession, reconciliation, aid, apology**, etc.
3) The action must be plausible for [{s_name}] as initiator; avoid absurd mismatches.
4) Output relation **must differ** from [{r_name}].

[Output constraints (strict)]
- Relation **must** be exactly one string from the list below (case and underscores preserved):
{relation_candidates}

- **No** analysis, decoration, quotes, or extra words; **one line only**: the relation name."""

_TRACER_IV_A_V4_TEMPLATE = """You are a TKG counterfactual trajectory expert. Follow TRACER IV.A Semantically Constrained Counterfactual History Construction to produce one on-disk counterfactual intervention for the pivot below.

[A. Intervention Pivot]
Treat this factual event as intervention pivot F_int in the history window:
- Time slice: {t_id}
- Fact: subject [{s_name}] — action [{r_name}] — object [{o_name}]

[B. Historical Context Retrieval]
Local evolution F_hist for the same (s,o) pair before F_int (distant to recent):
{historical_context}

Use this to understand F_int in the local event chain; **do not** rewrite in isolation from the evolution.

[C. High-Contrast Counterfactual Intervention]
Given F_hist and commonsense, produce counterfactual intervention F_cf:
1) **Contrastiveness**: substantively different from F_int (change at least one of subject, relation, object); never output the identical `subject||relation||object` as factual.
2) **Plausibility**: consistent with event logic and initiator capacity.
3) **Closure**: subject, object, relation must come from the candidate sets below (exact strings).

[Relation candidates (must choose one)]
{relation_candidates}

[Entity candidates (required if subject/object change; else keep originals)]
{entity_candidates}

[D. Output constraints]
- **One line only**: `subject||relation||object` (two `||` separators; all three fields non-empty).
- No analysis, titles, numbering, Markdown, or extra text."""

_TRACER_V5_PIVOT_GATE_TEMPLATE = """You are a TKG counterfactual construction expert. Decide whether the event below is a worthwhile intervention pivot F_int for TRACER (Intervention Pivot Discovery).

[Event under review] time slice {t_id}: subject [{s_name}] — action [{r_name}] — object [{o_name}]

[Local history F_hist (same subject–object pair, distant to recent)]
{historical_context}

[Auxiliary signals]
- Subject historical participation (out-degree + in-degree count): {s_degree}
- Object historical participation: {o_degree}

[Pivot criteria] (more satisfied → more likely YES)
1) May be a strategic turning point, escalation/de-escalation, or cascade potential in local evolution;
2) Relation/action is non-trivial; rewrite yields meaningful counterfactual contrast;
3) Avoid: background noise, redundant history, or rewrites that cannot plausibly chain forward.

[Output]
Output exactly one word: **YES** or **NO** (uppercase, no explanation)."""

_TRACER_V5_INTERVENTION_ROLLOUT_TEMPLATE = """You are a TRACER framework expert. The event below is accepted as intervention pivot F_int. Complete IV.A: high-contrast intervention + short-horizon rollout.

[Intervention pivot F_int] time slice {t_id}: {s_name} || {r_name} || {o_name}

[Local history F_hist]
{historical_context}

[High-contrast intervention] Produce F_cf: substantively different from F_int but plausible; subject/relation/object may change; vocabulary closed over candidate sets.

[Short-horizon rollout] For the **existing fact slots** below, provide counterfactual rewrites F_fut (time slices fixed; do not change time; subject/relation/object may change):
{rollout_slots}

If no rollout slots apply, output only the intervention line.

[Relation candidates]
{relation_candidates}

[Entity candidates]
{entity_candidates}

[Output format (strict)]
- Line 1: `INTERVENTION||subject||relation||object`
- Each following line per slot: `ROLLOUT||time_slice||subject||relation||object` (time slice must match the slot)
- No other text."""

def _normalize_prompt_version(version: str | None) -> str:
    if not version:
        return "v1"
    v = version.strip().lower()
    if v in ("v1", "legacy", "old"):
        return "v1"
    if v in ("v2", "multi", "branched", "new"):
        return "v2"
    if v in ("v3", "tcc", "polarity", "tcc_polarity", "hard_negative", "brsc"):
        return "v3"
    if v in ("v4", "tracer", "tracer_iv_a", "iv_a", "iv.a", "semantically_constrained"):
        return "v4"
    if v in ("v5", "tracer_full", "tracer_iv_a_full", "iv_a_full"):
        return "v5"
    raise ValueError(
        f"Unknown prompt version: {version!r}; use v1, v2, v3, v4, or v5 (full TRACER IV.A)"
    )

def _relation_candidates_text(r_name: str, seed_text: str, k: int = 80) -> str:
    rels = list(REL2ID.keys())
    if not rels:
        return r_name
    seed = zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF
    rng = random.Random(seed)
    pool = [x for x in rels if x != r_name]
    if len(pool) > (k - 1):
        picked = rng.sample(pool, k - 1)
    else:
        picked = pool
    picked.append(r_name)
    uniq = []
    seen = set()
    for x in picked:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    if r_name in uniq:
        uniq.remove(r_name)
        uniq.insert(0, r_name)
    return "\n".join(uniq)

def _clean_v3_relation_output(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    line = t.splitlines()[0].strip()
    if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'「」":
        line = line[1:-1].strip()
    line = line.strip("。．.!！?？:：;；,，")
    return line.strip()

def _build_entity_neighbors_from_lines(lines: list[str], top_per_ent: int = 80) -> tuple[dict[int, list[int]], list[int]]:
    cnt_by_ent: dict[int, Counter[int]] = defaultdict(Counter)
    deg: Counter[int] = Counter()
    for ln in lines:
        if not ln.strip():
            continue
        parsed = _parse_train_line_hrot(ln)
        if parsed is None:
            continue
        s_id, _r_id, o_id, _t_id = parsed
        cnt_by_ent[s_id][o_id] += 1
        cnt_by_ent[o_id][s_id] += 1
        deg[s_id] += 1
        deg[o_id] += 1

    neighbors: dict[int, list[int]] = {}
    for e, c in cnt_by_ent.items():
        items = list(c.items())
        items.sort(key=lambda x: (-x[1], x[0]))
        neighbors[e] = [nid for nid, _v in items[:top_per_ent]]

    hot_items = list(deg.items())
    hot_items.sort(key=lambda x: (-x[1], x[0]))
    global_hot = [eid for eid, _v in hot_items]
    return neighbors, global_hot

def _entity_candidates_text(
    s_id: int,
    o_id: int,
    k_total: int,
    neighbors: dict[int, list[int]],
    global_hot: list[int],
) -> str:
    k_total = max(10, int(k_total))
    k_s = max(0, min((k_total - 2) // 2, k_total))
    k_o = max(0, min((k_total - 2) // 2, k_total))
    k_g = max(0, k_total - 2 - k_s - k_o)

    ids: list[int] = []
    ids.extend([s_id, o_id])
    ids.extend(neighbors.get(s_id, [])[:k_s])
    ids.extend(neighbors.get(o_id, [])[:k_o])
    if k_g > 0:
        ids.extend(global_hot[:k_g])

    seen: set[int] = set()
    names: list[str] = []
    for eid in ids:
        if eid in seen:
            continue
        seen.add(eid)
        name = ID2ENT.get(eid)
        if name:
            names.append(name)
        if len(names) >= k_total:
            break
    return "\n".join(names)

def _build_v4_history_texts(lines: list[str], hist_top: int = CF_V4_HIST_TOP) -> list[str]:
    texts: list[str] = []
    pair_prior: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for ln in lines:
        parsed = _parse_train_line_hrot(ln)
        if parsed is None:
            texts.append("(no local history available)")
            continue
        s_id, r_id, o_id, t_id = parsed
        s_name = ID2ENT.get(s_id, str(s_id))
        o_name = ID2ENT.get(o_id, str(o_id))
        prior = pair_prior[(s_id, o_id)][-hist_top:]
        if prior:
            rows = [
                f"- time slice {tt}: {s_name} || {rname} || {o_name}"
                for rname, tt in prior
            ]
            hist = "\n".join(rows)
        else:
            hist = "(no other co-occurring events for this subject–object pair before intervention; use F_int and commonsense for a high-contrast intervention.)"
        texts.append(hist)
        r_name = ID2REL.get(r_id, str(r_id))
        pair_prior[(s_id, o_id)].append((r_name, t_id))
    return texts

def _load_all_quads(lines: list[str]) -> list[tuple[int, int, int, int] | None]:
    return [_parse_train_line_hrot(ln) for ln in lines]

def _build_entity_degrees(lines: list[str]) -> dict[int, int]:
    deg: Counter[int] = Counter()
    for ln in lines:
        parsed = _parse_train_line_hrot(ln)
        if parsed is None:
            continue
        s_id, _r, o_id, _t = parsed
        deg[s_id] += 1
        deg[o_id] += 1
    return dict(deg)

def _find_rollout_slot_indices(
    quads: list[tuple[int, int, int, int] | None],
    pivot_idx: int,
    pivot_s: int,
    pivot_o: int,
    pivot_t: int,
    lr: int = CF_V5_ROLLOUT_LR,
    max_slots: int = CF_V5_ROLLOUT_MAX_SLOTS,
) -> list[int]:
    pivot_ents = {pivot_s, pivot_o}
    slots: list[int] = []
    for j in range(pivot_idx + 1, len(quads)):
        q = quads[j]
        if q is None:
            continue
        sj, _rj, oj, tj = q
        if tj <= pivot_t or tj > pivot_t + lr:
            continue
        if pivot_ents.isdisjoint({sj, oj}):
            continue
        slots.append(j)
        if len(slots) >= max_slots:
            break
    return slots

def _format_rollout_slots_text(slot_indices: list[int], quads: list[tuple[int, int, int, int] | None]) -> str:
    if not slot_indices:
        return "(no short-horizon rollout slots; output INTERVENTION line only.)"
    rows: list[str] = []
    for k, j in enumerate(slot_indices, 1):
        q = quads[j]
        if q is None:
            continue
        sj, rj, oj, tj = q
        rows.append(
            f"- slot {k} | time slice {tj} | fact: {ID2ENT.get(sj, sj)} || {ID2REL.get(rj, rj)} || {ID2ENT.get(oj, oj)}"
        )
    return "\n".join(rows) if rows else "(no short-horizon rollout slots; output INTERVENTION line only.)"

def _build_v5_pivot_gate_prompt(
    s_name: str,
    r_name: str,
    o_name: str,
    t_id: int | str,
    historical_context: str,
    s_degree: int,
    o_degree: int,
) -> str:
    return _TRACER_V5_PIVOT_GATE_TEMPLATE.format(
        s_name=s_name,
        r_name=r_name,
        o_name=o_name,
        t_id=t_id,
        historical_context=historical_context or "(none)",
        s_degree=s_degree,
        o_degree=o_degree,
    ).strip()

def _build_v5_intervention_rollout_prompt(
    s_name: str,
    r_name: str,
    o_name: str,
    t_id: int | str,
    historical_context: str,
    rollout_slots: str,
    relation_candidates: str,
    entity_candidates: str,
) -> str:
    return _TRACER_V5_INTERVENTION_ROLLOUT_TEMPLATE.format(
        s_name=s_name,
        r_name=r_name,
        o_name=o_name,
        t_id=t_id,
        historical_context=historical_context or "(none)",
        rollout_slots=rollout_slots,
        relation_candidates=relation_candidates,
        entity_candidates=entity_candidates,
    ).strip()

def _parse_v5_pivot_yes(text: str) -> bool:
    t = (text or "").strip().upper()
    if not t:
        return False
    first = t.splitlines()[0].strip()
    if first == "YES" or first.startswith("YES"):
        return True
    if first == "NO" or first.startswith("NO"):
        return False
    return "YES" in first and "NO" not in first

def _parse_v5_intervention_rollout(
    text: str,
) -> tuple[tuple[str, str, str] | None, list[tuple[int, str, str, str]]]:
    intervention: tuple[str, str, str] | None = None
    rollouts: list[tuple[int, str, str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("||")]
        head = parts[0].upper() if parts else ""
        if head == "INTERVENTION" and len(parts) == 4:
            intervention = (parts[1], parts[2], parts[3])
        elif head == "ROLLOUT" and len(parts) == 5:
            try:
                rollouts.append((int(parts[1]), parts[2], parts[3], parts[4]))
            except ValueError:
                continue
    return intervention, rollouts

def _map_cf_triple_names(
    sn: str,
    rn: str,
    on: str,
    ent_fuzzy: bool,
    ent_fuzzy_threshold: int,
) -> tuple[int, int, int, int] | None:
    cs = ENT2ID.get(sn) or ENT2ID_NORM.get(_norm_symbolic_name(sn))
    co = ENT2ID.get(on) or ENT2ID_NORM.get(_norm_symbolic_name(on))
    cr = REL2ID.get(rn) or REL2ID_NORM.get(_norm_symbolic_name(rn))
    hits = 0
    if ent_fuzzy and cs is None:
        hit = _fuzzy_ent_id(sn, ent_fuzzy_threshold)
        if hit is not None:
            cs = hit[0]
            hits += 1
    if ent_fuzzy and co is None:
        hit = _fuzzy_ent_id(on, ent_fuzzy_threshold)
        if hit is not None:
            co = hit[0]
            hits += 1
    if cs is None or cr is None or co is None:
        return None
    return cs, cr, co, hits

def _wrap_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content.strip()}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_content.strip()

def _build_instruction_prompt(
    tokenizer,
    s_name,
    r_name,
    o_name,
    prompt_version: str = "v1",
    entity_candidates: str = "",
    historical_context: str = "",
    t_id: int | str = "?",
) -> str:
    pv = _normalize_prompt_version(prompt_version)
    if pv == "v1":
        user_content = _LEGACY_PROMPT_TEMPLATE.format(
            s_name=s_name, o_name=o_name, r_name=r_name
        )
    elif pv == "v3":
        rel_cand = _relation_candidates_text(
            r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}|v3", k=CF_REL_CAND_K
        )
        user_content = _TCC_POLARITY_V3_TEMPLATE.format(
            s_name=s_name, o_name=o_name, r_name=r_name, relation_candidates=rel_cand
        )
    elif pv == "v4":
        rel_cand = _relation_candidates_text(
            r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}|v4", k=CF_REL_CAND_K
        )
        user_content = _TRACER_IV_A_V4_TEMPLATE.format(
            s_name=s_name,
            o_name=o_name,
            r_name=r_name,
            t_id=t_id,
            historical_context=historical_context or "(no local history available)",
            relation_candidates=rel_cand,
            entity_candidates=entity_candidates,
        )
    else:
        rel_cand = _relation_candidates_text(
            r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}", k=CF_REL_CAND_K
        )
        user_content = _MULTI_BRANCH_PROMPT_TEMPLATE.format(
            s_name=s_name, o_name=o_name, r_name=r_name, relation_candidates=rel_cand, entity_candidates=entity_candidates
        )
    messages = [{"role": "user", "content": user_content.strip()}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_content.strip()

def _build_instruction_prompt_text(
    s_name,
    r_name,
    o_name,
    prompt_version: str = "v1",
    entity_candidates: str = "",
    historical_context: str = "",
    t_id: int | str = "?",
) -> str:
    pv = _normalize_prompt_version(prompt_version)
    if pv == "v1":
        return _LEGACY_PROMPT_TEMPLATE.format(s_name=s_name, o_name=o_name, r_name=r_name).strip()
    if pv == "v3":
        rel_cand = _relation_candidates_text(
            r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}|v3", k=CF_REL_CAND_K
        )
        return _TCC_POLARITY_V3_TEMPLATE.format(
            s_name=s_name, o_name=o_name, r_name=r_name, relation_candidates=rel_cand
        ).strip()
    if pv == "v4":
        rel_cand = _relation_candidates_text(
            r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}|v4", k=CF_REL_CAND_K
        )
        return _TRACER_IV_A_V4_TEMPLATE.format(
            s_name=s_name,
            o_name=o_name,
            r_name=r_name,
            t_id=t_id,
            historical_context=historical_context or "(no local history available)",
            relation_candidates=rel_cand,
            entity_candidates=entity_candidates,
        ).strip()
    rel_cand = _relation_candidates_text(
        r_name=r_name, seed_text=f"{s_name}|{r_name}|{o_name}", k=CF_REL_CAND_K
    )
    return _MULTI_BRANCH_PROMPT_TEMPLATE.format(
        s_name=s_name, o_name=o_name, r_name=r_name, relation_candidates=rel_cand, entity_candidates=entity_candidates
    ).strip()

def _parse_multi_branch_output(text: str) -> list[tuple[str, str, str]]:
    branches: list[tuple[str, str, str]] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) != 3:
            continue
        s_n, r_n, o_n = parts
        if s_n and r_n and o_n:
            branches.append((s_n, r_n, o_n))
    return branches

def _safe_int_len(x) -> int | None:
    if x is None:
        return None
    try:
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            return int(len(x))
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return None

def _extract_vllm_token_usage(request_output, generation_output) -> dict[str, int | None]:
    prompt_tokens = _safe_int_len(getattr(request_output, "prompt_token_count", None))
    if prompt_tokens is None:
        prompt_tokens = _safe_int_len(getattr(request_output, "prompt_tokens", None))
    if prompt_tokens is None:
        prompt_tokens = _safe_int_len(getattr(request_output, "prompt_token_ids", None))

    completion_tokens = _safe_int_len(getattr(request_output, "completion_token_count", None))
    if completion_tokens is None:
        completion_tokens = _safe_int_len(getattr(request_output, "completion_tokens", None))
    if completion_tokens is None:
        completion_tokens = _safe_int_len(getattr(generation_output, "token_count", None))
    if completion_tokens is None:
        completion_tokens = _safe_int_len(getattr(generation_output, "token_ids", None))

    total_tokens = _safe_int_len(getattr(request_output, "total_token_count", None))
    if total_tokens is None:
        total_tokens = _safe_int_len(getattr(request_output, "total_tokens", None))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def generate_counterfactual_raw(
    llm,
    tokenizer,
    sampling_params,
    s_name,
    r_name,
    o_name,
    prompt_version: str = "v1",
    entity_candidates: str = "",
    historical_context: str = "",
    t_id: int | str = "?",
) -> tuple[str, dict[str, int | None]]:
    prompt = _build_instruction_prompt(
        tokenizer,
        s_name,
        r_name,
        o_name,
        prompt_version=prompt_version,
        entity_candidates=entity_candidates,
        historical_context=historical_context,
        t_id=t_id,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    req_out = outputs[0]
    gen_out = req_out.outputs[0]
    text = gen_out.text.strip()
    usage = _extract_vllm_token_usage(req_out, gen_out)
    return text, usage

def generate_from_user_content_raw(
    llm,
    tokenizer,
    sampling_params,
    user_content: str,
) -> tuple[str, dict[str, int | None]]:
    prompt = _wrap_chat_prompt(tokenizer, user_content)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    req_out = outputs[0]
    gen_out = req_out.outputs[0]
    text = gen_out.text.strip()
    usage = _extract_vllm_token_usage(req_out, gen_out)
    return text, usage

_OPENAI_CLIENT_BY_BASE_URL: dict[str, object] = {}

def _get_openai_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed: run `pip install openai` before using API models.") from e

    base = base_url.rstrip("/")
    client = _OPENAI_CLIENT_BY_BASE_URL.get(base)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base)
        _OPENAI_CLIENT_BY_BASE_URL[base] = client
    return client

def _is_data_inspection_failed(exc: Exception) -> bool:
    s = str(exc)
    return ("data_inspection_failed" in s) or ("DataInspectionFailed" in s)

def _safe_retry_messages(user_content: str) -> list[dict[str, str]]:
    system = (
        "You are an academic text generation assistant. "
        "Avoid detailed depictions of violence, hate, illegal acts, sexual content, or other inappropriate material. "
        "Output results only; do not explain or expand background."
    )
    softened = user_content
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": softened},
    ]

def generate_counterfactual_api_raw(
    api_url: str,
    api_key: str,
    api_model: str,
    sampling_params: SamplingParams,
    s_name,
    r_name,
    o_name,
    prompt_version: str = "v1",
    entity_candidates: str = "",
    historical_context: str = "",
    t_id: int | str = "?",
) -> tuple[str, dict[str, int | None]]:
    user_content = _build_instruction_prompt_text(
        s_name=s_name,
        r_name=r_name,
        o_name=o_name,
        prompt_version=prompt_version,
        entity_candidates=entity_candidates,
        historical_context=historical_context,
        t_id=t_id,
    )
    client = _get_openai_client(base_url=api_url, api_key=api_key)

    temperature = getattr(sampling_params, "temperature", None)
    max_tokens = getattr(sampling_params, "max_tokens", None)

    def _call(messages: list[dict[str, str]]):
        return client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

    try:
        completion = _call([{"role": "user", "content": user_content}])
        retried = False
    except Exception as e:
        if not _is_data_inspection_failed(e):
            raise
        completion = _call(_safe_retry_messages(user_content))
        retried = True

    text = completion.choices[0].message.content or ""
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    return text.strip(), {
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
        "retried_safe_prompt": 1 if retried else 0,
    }

def generate_from_user_content_api_raw(
    api_url: str,
    api_key: str,
    api_model: str,
    sampling_params: SamplingParams,
    user_content: str,
) -> tuple[str, dict[str, int | None]]:
    client = _get_openai_client(base_url=api_url, api_key=api_key)
    temperature = getattr(sampling_params, "temperature", None)
    max_tokens = getattr(sampling_params, "max_tokens", None)

    def _call(messages: list[dict[str, str]]):
        return client.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

    try:
        completion = _call([{"role": "user", "content": user_content}])
        retried = False
    except Exception as e:
        if not _is_data_inspection_failed(e):
            raise
        completion = _call(_safe_retry_messages(user_content))
        retried = True

    text = completion.choices[0].message.content or ""
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    return text.strip(), {
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, int) else None,
        "retried_safe_prompt": 1 if retried else 0,
    }

def _sanitize_dir_token(s: str) -> str:
    s = s.strip()
    out: list[str] = []
    for c in s:
        if c.isalnum() or c in ".-_":
            out.append(c)
        else:
            out.append("_")
    t = "".join(out).strip("_")
    while "__" in t:
        t = t.replace("__", "_")
    return (t[:80] or "token").strip("_")

def _unique_run_dir(parent: Path, base_name: str) -> Path:
    candidate = parent / base_name
    if not candidate.exists():
        return candidate
    for i in range(2, 10_000):
        alt = parent / f"{base_name}__{i}"
        if not alt.exists():
            return alt
    return parent / f"{base_name}__{os.getpid()}"

def _make_llm(model_path: str) -> LLM:
    tp = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", os.environ.get("VLLM_TP", "1")))
    dtype = os.environ.get("VLLM_DTYPE", "bfloat16")
    mem = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    trust = os.environ.get("VLLM_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
    max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    return LLM(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        trust_remote_code=trust,
        max_model_len=max_len,
        disable_log_stats=True,
    )

def main(
    input_file: str,
    runs_parent: str | Path | None = None,
    sample_size=None,
    prompt_version: str | None = None,
    run_id: str | None = None,
    model_alias: str | None = None,
    model_path_override: str | None = None,
    backend_override: str | None = None,
    llm_config_path: str | Path | None = None,
    resume_dir: str | Path | None = None,
    test_mode: bool = False,
    ent_fuzzy: bool = False,
    ent_fuzzy_threshold: int = 97,
):
    t_all = time.perf_counter()
    started_at = datetime.now().astimezone()
    started_iso = started_at.isoformat(timespec="seconds")
    started_compact = started_at.strftime("%Y%m%d_%H%M%S")

    pv = _normalize_prompt_version(prompt_version or os.environ.get("CF_PROMPT_VERSION", "v1"))

    llm_cfg_path = Path(llm_config_path or os.environ.get("CFGEN_LLM_CONFIG", DEFAULT_LLM_CONFIG_PATH))
    llm_cfg_path = llm_cfg_path.expanduser().resolve()
    llm_settings = resolve_llm_settings(
        model_alias=model_alias,
        model_path_override=model_path_override,
        backend_override=backend_override,
        config_path=llm_cfg_path,
    )
    backend = llm_settings.backend
    model_alias = llm_settings.alias
    model_path = llm_settings.model_path

    sampling_params_gate = SamplingParams(temperature=0.3, max_tokens=8)
    sampling_params_v5_ir = SamplingParams(temperature=0.7, max_tokens=384)
    if pv in ("v1", "v3"):
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=48 if pv == "v3" else 10
        )
    elif pv == "v4":
        sampling_params = SamplingParams(temperature=0.7, max_tokens=96)
    elif pv == "v5":
        sampling_params = sampling_params_v5_ir
    else:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    llm = None
    tokenizer = None
    t0 = time.perf_counter()
    if backend == "vllm":
        llm = _make_llm(model_path)
        tokenizer = llm.get_tokenizer()
        api_url = None
        api_key = None
    else:
        api_url = llm_settings.api_url
        api_key = llm_settings.api_key
    t_load = time.perf_counter() - t0

    with open(input_file, 'r') as f:
        lines = f.readlines()

    if sample_size is not None:
        lines = lines[:sample_size]

    ent_neighbors, ent_global_hot = _build_entity_neighbors_from_lines(
        lines,
        top_per_ent=CF_ENT_NEI_TOP,
    )
    v4_histories: list[str] = []
    all_quads: list[tuple[int, int, int, int] | None] = []
    entity_degrees: dict[int, int] = {}
    if pv in ("v4", "v5"):
        v4_histories = _build_v4_history_texts(lines, hist_top=CF_V4_HIST_TOP)
    if pv == "v5":
        all_quads = _load_all_quads(lines)
        entity_degrees = _build_entity_degrees(lines)

    skipped = 0
    gen_lines = 0
    inference_calls = 0
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    token_calls_with_total = 0
    api_safe_retries = 0
    api_blocked_skips = 0
    v2_raw_lines_sum = 0
    v2_parsed_lines_sum = 0
    v2_kept_lines_sum = 0
    v2_drop_bad_format_sum = 0
    v2_drop_oov_sum = 0
    v2_drop_oov_ent_sum = 0
    v2_drop_oov_rel_sum = 0
    v2_drop_dup_sum = 0
    v2_fallback_used_sum = 0
    v2_ent_fuzzy_hits_sum = 0
    v5_pivot_yes_sum = 0
    v5_pivot_no_sum = 0
    v5_rollout_applied_sum = 0
    pending_rollout: dict[int, tuple[int, int, int, int]] = {}

    parent_raw = runs_parent if runs_parent is not None else os.environ.get(
        "CF_RUNS_PARENT", str(DEFAULT_RUNS_PARENT)
    )
    run_parent = Path(parent_raw).expanduser().resolve()
    run_parent.mkdir(parents=True, exist_ok=True)

    model_dir_token = _sanitize_dir_token(model_alias)

    if backend == "vllm":
        def gen_one(
            s_name,
            r_name,
            o_name,
            entity_candidates: str,
            historical_context: str = "",
            t_id: int | str = "?",
        ):
            return generate_counterfactual_raw(
                llm,
                tokenizer,
                sampling_params,
                s_name,
                r_name,
                o_name,
                prompt_version=pv,
                entity_candidates=entity_candidates,
                historical_context=historical_context,
                t_id=t_id,
            )

        def gen_user(user_content: str, sp: SamplingParams | None = None):
            sp_use = sp or sampling_params
            return generate_from_user_content_raw(
                llm, tokenizer, sp_use, user_content
            )
    else:
        api_model_for_call = model_path
        def gen_one(
            s_name,
            r_name,
            o_name,
            entity_candidates: str,
            historical_context: str = "",
            t_id: int | str = "?",
        ):
            return generate_counterfactual_api_raw(
                api_url=api_url,
                api_key=api_key,
                api_model=api_model_for_call,
                sampling_params=sampling_params,
                s_name=s_name,
                r_name=r_name,
                o_name=o_name,
                prompt_version=pv,
                entity_candidates=entity_candidates,
                historical_context=historical_context,
                t_id=t_id,
            )

        def gen_user(user_content: str, sp: SamplingParams | None = None):
            sp_use = sp or sampling_params
            return generate_from_user_content_api_raw(
                api_url, api_key, api_model_for_call, sp_use, user_content
            )

    if resume_dir is not None:
        run_dir = Path(resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume-dir does not exist or is not a directory: {run_dir}")
        quadruples_path = run_dir / "cf_train.txt"
        meta_path = run_dir / "meta.json"
        state_path = run_dir / "state.json"
        index_path = run_parent / "run_index.jsonl"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            processed_edges = int(state.get("processed_edges", 0))
            gen_lines = int(state.get("gen_lines", gen_lines))
            skipped = int(state.get("skipped", skipped))
            inference_calls = int(state.get("inference_calls", inference_calls))
            prompt_tokens_sum = int(state.get("prompt_tokens_sum", prompt_tokens_sum))
            completion_tokens_sum = int(state.get("completion_tokens_sum", completion_tokens_sum))
            total_tokens_sum = int(state.get("total_tokens_sum", total_tokens_sum))
            token_calls_with_total = int(state.get("token_calls_with_total", token_calls_with_total))
            api_safe_retries = int(state.get("api_safe_retries", api_safe_retries))
            api_blocked_skips = int(state.get("api_blocked_skips", api_blocked_skips))
            v2_raw_lines_sum = int(state.get("v2_raw_lines_sum", v2_raw_lines_sum))
            v2_parsed_lines_sum = int(state.get("v2_parsed_lines_sum", v2_parsed_lines_sum))
            v2_kept_lines_sum = int(state.get("v2_kept_lines_sum", v2_kept_lines_sum))
            v2_drop_bad_format_sum = int(state.get("v2_drop_bad_format_sum", v2_drop_bad_format_sum))
            v2_drop_oov_sum = int(state.get("v2_drop_oov_sum", v2_drop_oov_sum))
            v2_drop_oov_ent_sum = int(state.get("v2_drop_oov_ent_sum", v2_drop_oov_ent_sum))
            v2_drop_oov_rel_sum = int(state.get("v2_drop_oov_rel_sum", v2_drop_oov_rel_sum))
            v2_drop_dup_sum = int(state.get("v2_drop_dup_sum", v2_drop_dup_sum))
            v2_fallback_used_sum = int(state.get("v2_fallback_used_sum", v2_fallback_used_sum))
            v2_ent_fuzzy_hits_sum = int(state.get("v2_ent_fuzzy_hits_sum", v2_ent_fuzzy_hits_sum))
            v5_pivot_yes_sum = int(state.get("v5_pivot_yes_sum", v5_pivot_yes_sum))
            v5_pivot_no_sum = int(state.get("v5_pivot_no_sum", v5_pivot_no_sum))
            v5_rollout_applied_sum = int(state.get("v5_rollout_applied_sum", v5_rollout_applied_sum))
            pr = state.get("pending_rollout", {})
            if isinstance(pr, dict):
                pending_rollout = {int(k): tuple(v) for k, v in pr.items()}
        else:
            processed_edges = 0
    else:
        ds_token = _sanitize_dir_token(Path(input_file).resolve().parent.name)
        base_dir_name_tmp = "_".join([model_dir_token, ds_token, started_compact, pv])
        run_dir = _unique_run_dir(run_parent, base_dir_name_tmp)
        run_dir.mkdir(parents=False)
        quadruples_path = run_dir / "cf_train.txt"
        meta_path = run_dir / "meta.json"
        state_path = run_dir / "state.json"
        index_path = run_parent / "run_index.jsonl"
        processed_edges = 0

    flush_every = int(os.environ.get("CF_FLUSH_EVERY", "1000"))
    save_state_every_edges = int(os.environ.get("CF_SAVE_STATE_EVERY_EDGES", "1"))

    total_edges = len(lines)
    print(
        f"Generating counterfactual graph for {total_edges} edges… [prompt version: {pv}"
        f" (v5=full TRACER IV.A: pivot gate + intervention + rollout)]"
    )
    t_loop = time.perf_counter()
    pbar = tqdm(lines, total=total_edges, desc="counterfactual", unit="edge")
    t_w0 = time.perf_counter()
    open_mode = "a" if resume_dir is not None else "w"
    with open(quadruples_path, open_mode, encoding="utf-8") as out_f:
        for i, line in enumerate(pbar):
            if i < processed_edges:
                continue
            parsed = _parse_train_line_hrot(line)
            if parsed is None:
                skipped += 1
                pbar.set_postfix(gen=gen_lines, skip=skipped, refresh=False)
                continue
            s_id, r_id, o_id, t_id = parsed

            if s_id not in ID2ENT or r_id not in ID2REL or o_id not in ID2ENT:
                skipped += 1
                pbar.set_postfix(gen=gen_lines, skip=skipped, refresh=False)
                continue

            s_name = ID2ENT[s_id]
            r_name = ID2REL[r_id]
            o_name = ID2ENT[o_id]

            if pv == "v5":
                if i in pending_rollout:
                    cs, cr, co, tt = pending_rollout.pop(i)
                    out_f.write(f"{cs}\t{cr}\t{co}\t{tt}\n")
                    gen_lines += 1
                    v5_rollout_applied_sum += 1
                    processed_edges = i + 1
                    pbar.set_postfix(gen=gen_lines, skip=skipped, pivot=v5_pivot_yes_sum, refresh=False)
                    if save_state_every_edges > 0 and processed_edges % save_state_every_edges == 0:
                        state = {
                            "processed_edges": processed_edges,
                            "gen_lines": gen_lines,
                            "skipped": skipped,
                            "pending_rollout": {str(k): list(v) for k, v in pending_rollout.items()},
                            "v5_pivot_yes_sum": v5_pivot_yes_sum,
                            "v5_pivot_no_sum": v5_pivot_no_sum,
                            "v5_rollout_applied_sum": v5_rollout_applied_sum,
                            "inference_calls": inference_calls,
                            "prompt_tokens_sum": prompt_tokens_sum,
                            "completion_tokens_sum": completion_tokens_sum,
                            "total_tokens_sum": total_tokens_sum,
                            "token_calls_with_total": token_calls_with_total,
                            "prompt_version": pv,
                        }
                        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue

                try:
                    ent_cand = _entity_candidates_text(
                        s_id=s_id,
                        o_id=o_id,
                        k_total=CF_ENT_CAND_K,
                        neighbors=ent_neighbors,
                        global_hot=ent_global_hot,
                    )
                    hist_ctx = v4_histories[i] if i < len(v4_histories) else ""
                    gate_prompt = _build_v5_pivot_gate_prompt(
                        s_name=s_name,
                        r_name=r_name,
                        o_name=o_name,
                        t_id=t_id,
                        historical_context=hist_ctx,
                        s_degree=entity_degrees.get(s_id, 0),
                        o_degree=entity_degrees.get(o_id, 0),
                    )
                    gate_raw, u_gate = gen_user(gate_prompt, sampling_params_gate)
                    inference_calls += 1
                    api_safe_retries += int(u_gate.get("retried_safe_prompt") or 0)
                    for u in (u_gate,):
                        if u.get("prompt_tokens") is not None:
                            prompt_tokens_sum += int(u["prompt_tokens"])
                        if u.get("completion_tokens") is not None:
                            completion_tokens_sum += int(u["completion_tokens"])
                        if u.get("total_tokens") is not None:
                            total_tokens_sum += int(u["total_tokens"])
                            token_calls_with_total += 1

                    if not _parse_v5_pivot_yes(gate_raw):
                        out_f.write(f"{s_id}\t{r_id}\t{o_id}\t{t_id}\n")
                        gen_lines += 1
                        v5_pivot_no_sum += 1
                        if test_mode:
                            print(f"\n[TEST v5] edge={i} PIVOT=NO gate={gate_raw!r} -> copy factual")
                    else:
                        v5_pivot_yes_sum += 1
                        slot_indices = _find_rollout_slot_indices(
                            all_quads, i, s_id, o_id, t_id
                        )
                        slots_text = _format_rollout_slots_text(slot_indices, all_quads)
                        rel_cand = _relation_candidates_text(
                            r_name=r_name,
                            seed_text=f"{s_name}|{r_name}|{o_name}|v5",
                            k=CF_REL_CAND_K,
                        )
                        ir_prompt = _build_v5_intervention_rollout_prompt(
                            s_name=s_name,
                            r_name=r_name,
                            o_name=o_name,
                            t_id=t_id,
                            historical_context=hist_ctx,
                            rollout_slots=slots_text,
                            relation_candidates=rel_cand,
                            entity_candidates=ent_cand,
                        )
                        raw, u_ir = gen_user(ir_prompt, sampling_params_v5_ir)
                        inference_calls += 1
                        api_safe_retries += int(u_ir.get("retried_safe_prompt") or 0)
                        if u_ir.get("prompt_tokens") is not None:
                            prompt_tokens_sum += int(u_ir["prompt_tokens"])
                        if u_ir.get("completion_tokens") is not None:
                            completion_tokens_sum += int(u_ir["completion_tokens"])
                        if u_ir.get("total_tokens") is not None:
                            total_tokens_sum += int(u_ir["total_tokens"])
                            token_calls_with_total += 1

                        intervention, rollouts = _parse_v5_intervention_rollout(raw)
                        factual_key = (s_id, r_id, o_id)
                        wrote = False
                        if intervention:
                            mapped = _map_cf_triple_names(
                                intervention[0],
                                intervention[1],
                                intervention[2],
                                ent_fuzzy,
                                ent_fuzzy_threshold,
                            )
                            if mapped is not None:
                                cs, cr, co, fh = mapped
                                v2_ent_fuzzy_hits_sum += fh
                                if (cs, cr, co) != factual_key:
                                    out_f.write(f"{cs}\t{cr}\t{co}\t{t_id}\n")
                                    wrote = True
                        if not wrote:
                            cf_r_id = (r_id + 1) % len(ID2REL)
                            out_f.write(f"{s_id}\t{cf_r_id}\t{o_id}\t{t_id}\n")
                            v2_fallback_used_sum += 1
                        gen_lines += 1

                        for slot_j, rollout in zip(slot_indices, rollouts):
                            _rt, rs, rr, ro = rollout
                            qslot = all_quads[slot_j]
                            if qslot is None or slot_j <= i:
                                continue
                            slot_t = qslot[3]
                            mapped_r = _map_cf_triple_names(
                                rs, rr, ro, ent_fuzzy, ent_fuzzy_threshold
                            )
                            if mapped_r is None:
                                continue
                            cs, cr, co, fh = mapped_r
                            v2_ent_fuzzy_hits_sum += fh
                            if slot_j not in pending_rollout:
                                pending_rollout[slot_j] = (cs, cr, co, slot_t)

                        if test_mode:
                            print("\n" + "=" * 80)
                            print(f"[TEST v5] edge={i} PIVOT=YES gate={gate_raw!r}")
                            print(f"[TEST] slots={slot_indices}")
                            print("[TEST] ir_output:")
                            print(raw)
                except Exception as e:
                    if backend == "api" and _is_data_inspection_failed(e):
                        api_blocked_skips += 1
                        skipped += 1
                        out_f.write(f"{s_id}\t{r_id}\t{o_id}\t{t_id}\n")
                        gen_lines += 1
                        pbar.set_postfix(gen=gen_lines, skip=skipped, refresh=False)
                        processed_edges = i + 1
                        continue
                    raise

                processed_edges = i + 1
                pbar.set_postfix(
                    gen=gen_lines, skip=skipped, pivot=v5_pivot_yes_sum, refresh=False
                )
                if save_state_every_edges > 0 and processed_edges % save_state_every_edges == 0:
                    state = {
                        "processed_edges": processed_edges,
                        "gen_lines": gen_lines,
                        "skipped": skipped,
                        "pending_rollout": {str(k): list(v) for k, v in pending_rollout.items()},
                        "v5_pivot_yes_sum": v5_pivot_yes_sum,
                        "v5_pivot_no_sum": v5_pivot_no_sum,
                        "v5_rollout_applied_sum": v5_rollout_applied_sum,
                        "inference_calls": inference_calls,
                        "prompt_tokens_sum": prompt_tokens_sum,
                        "completion_tokens_sum": completion_tokens_sum,
                        "total_tokens_sum": total_tokens_sum,
                        "token_calls_with_total": token_calls_with_total,
                        "prompt_version": pv,
                    }
                    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                if flush_every > 0 and gen_lines % flush_every == 0:
                    out_f.flush()
                continue

            try:
                ent_cand = ""
                hist_ctx = ""
                if pv in ("v2", "v4"):
                    ent_cand = _entity_candidates_text(
                        s_id=s_id,
                        o_id=o_id,
                        k_total=CF_ENT_CAND_K,
                        neighbors=ent_neighbors,
                        global_hot=ent_global_hot,
                    )
                if pv == "v4":
                    hist_ctx = v4_histories[i] if i < len(v4_histories) else ""
                raw, usage = gen_one(
                    s_name, r_name, o_name, ent_cand, historical_context=hist_ctx, t_id=t_id
                )
            except Exception as e:
                if backend == "api" and _is_data_inspection_failed(e):
                    api_blocked_skips += 1
                    skipped += 1
                    pbar.set_postfix(gen=gen_lines, skip=skipped, refresh=False)
                    continue
                raise
            inference_calls += 1
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            api_safe_retries += int(usage.get("retried_safe_prompt") or 0)
            if pt is not None:
                prompt_tokens_sum += int(pt)
            if ct is not None:
                completion_tokens_sum += int(ct)
            if tt is not None:
                total_tokens_sum += int(tt)
                token_calls_with_total += 1

            if pv == "v1":
                cf_r_id = REL2ID.get(raw, (r_id + 1) % len(ID2REL))
                out_f.write(f"{s_id}\t{cf_r_id}\t{o_id}\t{t_id}\n")
                gen_lines += 1
            elif pv == "v3":
                rel_token = _clean_v3_relation_output(raw)
                cf_r_id = REL2ID.get(rel_token)
                if cf_r_id is None:
                    cf_r_id = REL2ID_NORM.get(_norm_symbolic_name(rel_token))
                if cf_r_id is None:
                    cf_r_id = (r_id + 1) % len(ID2REL)
                elif cf_r_id == r_id:
                    cf_r_id = (r_id + 1) % len(ID2REL)
                out_f.write(f"{s_id}\t{cf_r_id}\t{o_id}\t{t_id}\n")
                gen_lines += 1
                if test_mode:
                    print("\n" + "=" * 80)
                    print(f"[TEST] edge_idx={i} processed_edges={processed_edges} t_id={t_id}")
                    print(f"[TEST] src: {s_name} || {r_name} || {o_name}")
                    print(f"[TEST] token_usage: pt={pt} ct={ct} tt={tt} retried={usage.get('retried_safe_prompt')}")
                    print("[TEST] raw_output:")
                    print(raw)
                    print(
                        f"[TEST] v3: cleaned_rel={_clean_v3_relation_output(raw)!r} "
                        f"-> cf_r_id={cf_r_id} (factual r_id={r_id})"
                    )
            elif pv == "v4":
                raw_lines = [ln for ln in raw.splitlines() if ln.strip()]
                v2_raw_lines = len(raw_lines)
                v2_raw_lines_sum += v2_raw_lines
                parsed: list[tuple[str, str, str]] = []
                bad_format = 0
                for ln in raw_lines:
                    parts = [p.strip() for p in ln.strip().split("||")]
                    if len(parts) != 3:
                        bad_format += 1
                        continue
                    sn, rn, on = parts
                    if not (sn and rn and on):
                        bad_format += 1
                        continue
                    parsed.append((sn, rn, on))
                v2_parsed_lines = len(parsed)
                v2_parsed_lines_sum += v2_parsed_lines
                v2_drop_bad_format_sum += bad_format
                kept = 0
                dropped_oov = 0
                dropped_oov_ent = 0
                dropped_oov_rel = 0
                dropped_dup = 0
                ent_fuzzy_hits = 0
                factual_key = (s_id, r_id, o_id)
                for sn, rn, on in parsed:
                    cs = ENT2ID.get(sn)
                    cr = REL2ID.get(rn)
                    co = ENT2ID.get(on)
                    if cs is None:
                        cs = ENT2ID_NORM.get(_norm_symbolic_name(sn))
                    if co is None:
                        co = ENT2ID_NORM.get(_norm_symbolic_name(on))
                    if cr is None:
                        cr = REL2ID_NORM.get(_norm_symbolic_name(rn))
                    if ent_fuzzy and cs is None:
                        hit = _fuzzy_ent_id(sn, ent_fuzzy_threshold)
                        if hit is not None:
                            cs = hit[0]
                            ent_fuzzy_hits += 1
                    if ent_fuzzy and co is None:
                        hit = _fuzzy_ent_id(on, ent_fuzzy_threshold)
                        if hit is not None:
                            co = hit[0]
                            ent_fuzzy_hits += 1
                    if cs is None or cr is None or co is None:
                        dropped_oov += 1
                        if cs is None or co is None:
                            dropped_oov_ent += 1
                        if cr is None:
                            dropped_oov_rel += 1
                        continue
                    if (cs, cr, co) == factual_key:
                        dropped_dup += 1
                        continue
                    out_f.write(f"{cs}\t{cr}\t{co}\t{t_id}\n")
                    gen_lines += 1
                    kept = 1
                    break
                if kept == 0:
                    cf_r_id = (r_id + 1) % len(ID2REL)
                    out_f.write(f"{s_id}\t{cf_r_id}\t{o_id}\t{t_id}\n")
                    gen_lines += 1
                    v2_fallback_used_sum += 1
                v2_kept_lines_sum += kept
                v2_drop_oov_sum += dropped_oov
                v2_drop_oov_ent_sum += dropped_oov_ent
                v2_drop_oov_rel_sum += dropped_oov_rel
                v2_drop_dup_sum += dropped_dup
                v2_ent_fuzzy_hits_sum += ent_fuzzy_hits
                if test_mode:
                    print("\n" + "=" * 80)
                    print(f"[TEST] edge_idx={i} t_id={t_id}")
                    print(f"[TEST] F_int: {s_name} || {r_name} || {o_name}")
                    print(f"[TEST] F_hist:\n{hist_ctx}")
                    print("[TEST] raw_output:")
                    print(raw)
                    print(
                        f"[TEST] v4: kept={kept} parsed={v2_parsed_lines} "
                        f"drop_oov={dropped_oov} drop_dup={dropped_dup} "
                        f"fallback={1 if kept == 0 else 0}"
                    )
            else:
                raw_lines = [ln for ln in raw.splitlines() if ln.strip()]
                v2_raw_lines = len(raw_lines)
                v2_raw_lines_sum += v2_raw_lines

                parsed: list[tuple[str, str, str]] = []
                bad_format = 0
                for ln in raw_lines:
                    parts = [p.strip() for p in ln.strip().split("||")]
                    if len(parts) != 3:
                        bad_format += 1
                        continue
                    sn, rn, on = parts
                    if not (sn and rn and on):
                        bad_format += 1
                        continue
                    parsed.append((sn, rn, on))

                v2_parsed_lines = len(parsed)
                v2_parsed_lines_sum += v2_parsed_lines
                v2_drop_bad_format_sum += bad_format

                branches = parsed
                seen_branch: set[tuple[int, int, int]] = set()
                kept = 0
                dropped_oov = 0
                dropped_oov_ent = 0
                dropped_oov_rel = 0
                dropped_dup = 0
                ent_fuzzy_hits = 0
                for sn, rn, on in branches:
                    cs = ENT2ID.get(sn)
                    cr = REL2ID.get(rn)
                    co = ENT2ID.get(on)
                    if cs is None:
                        cs = ENT2ID_NORM.get(_norm_symbolic_name(sn))
                    if co is None:
                        co = ENT2ID_NORM.get(_norm_symbolic_name(on))
                    fuzzy_cs = None
                    fuzzy_co = None
                    if ent_fuzzy and cs is None:
                        fuzzy_cs = _fuzzy_ent_id(sn, ent_fuzzy_threshold)
                        if fuzzy_cs is not None:
                            cs = fuzzy_cs[0]
                            ent_fuzzy_hits += 1
                    if ent_fuzzy and co is None:
                        fuzzy_co = _fuzzy_ent_id(on, ent_fuzzy_threshold)
                        if fuzzy_co is not None:
                            co = fuzzy_co[0]
                            ent_fuzzy_hits += 1
                    if test_mode and ent_fuzzy and (fuzzy_cs is None or fuzzy_co is None) and (cs is None or co is None):
                        if cs is None:
                            best = _fuzzy_ent_best(sn)
                            if best is not None:
                                print(f"[TEST] ent_fuzzy_best(subject) score={best[1]:.1f} best_norm={best[0]} query={sn}")
                        if co is None:
                            best = _fuzzy_ent_best(on)
                            if best is not None:
                                print(f"[TEST] ent_fuzzy_best(object) score={best[1]:.1f} best_norm={best[0]} query={on}")
                    if cs is None or cr is None or co is None:
                        dropped_oov += 1
                        if cs is None or co is None:
                            dropped_oov_ent += 1
                        if cr is None:
                            dropped_oov_rel += 1
                        continue
                    key = (cs, cr, co)
                    if key in seen_branch:
                        dropped_dup += 1
                        continue
                    seen_branch.add(key)
                    out_f.write(f"{cs}\t{cr}\t{co}\t{t_id}\n")
                    gen_lines += 1
                    kept += 1
                if not seen_branch:
                    cf_r_id = (r_id + 1) % len(ID2REL)
                    out_f.write(f"{s_id}\t{cf_r_id}\t{o_id}\t{t_id}\n")
                    gen_lines += 1
                    v2_fallback_used_sum += 1

                v2_kept_lines_sum += kept
                v2_drop_oov_sum += dropped_oov
                v2_drop_oov_ent_sum += dropped_oov_ent
                v2_drop_oov_rel_sum += dropped_oov_rel
                v2_drop_dup_sum += dropped_dup
                v2_ent_fuzzy_hits_sum += ent_fuzzy_hits

                if test_mode:
                    print("\n" + "=" * 80)
                    print(f"[TEST] edge_idx={i} processed_edges={processed_edges} t_id={t_id}")
                    print(f"[TEST] src: {s_name} || {r_name} || {o_name}")
                    print(f"[TEST] token_usage: pt={pt} ct={ct} tt={tt} retried={usage.get('retried_safe_prompt')}")
                    print("[TEST] raw_output:")
                    print(raw)
                    print(
                        f"[TEST] v2_stats: raw_lines={v2_raw_lines} parsed_ok={v2_parsed_lines} "
                        f"kept={kept} drop_bad_format={bad_format} drop_oov={dropped_oov} "
                        f"(ent_oov={dropped_oov_ent}, rel_oov={dropped_oov_rel}) "
                        f"drop_dup={dropped_dup} ent_fuzzy_hits={ent_fuzzy_hits} "
                        f"fallback_used={1 if kept == 0 else 0}"
                    )

            if flush_every > 0 and gen_lines % flush_every == 0:
                out_f.flush()

            pbar.set_postfix(gen=gen_lines, skip=skipped, refresh=False)

            processed_edges = i + 1
            if save_state_every_edges > 0 and processed_edges % save_state_every_edges == 0:
                state = {
                    "processed_edges": processed_edges,
                    "gen_lines": gen_lines,
                    "skipped": skipped,
                    "inference_calls": inference_calls,
                    "prompt_tokens_sum": prompt_tokens_sum,
                    "completion_tokens_sum": completion_tokens_sum,
                    "total_tokens_sum": total_tokens_sum,
                    "token_calls_with_total": token_calls_with_total,
                    "api_safe_retries": api_safe_retries,
                    "api_blocked_skips": api_blocked_skips,
                    "v2_raw_lines_sum": v2_raw_lines_sum,
                    "v2_parsed_lines_sum": v2_parsed_lines_sum,
                    "v2_kept_lines_sum": v2_kept_lines_sum,
                    "v2_drop_bad_format_sum": v2_drop_bad_format_sum,
                    "v2_drop_oov_sum": v2_drop_oov_sum,
                    "v2_drop_oov_ent_sum": v2_drop_oov_ent_sum,
                    "v2_drop_oov_rel_sum": v2_drop_oov_rel_sum,
                    "v2_drop_dup_sum": v2_drop_dup_sum,
                    "v2_fallback_used_sum": v2_fallback_used_sum,
                    "v2_ent_fuzzy_hits_sum": v2_ent_fuzzy_hits_sum,
                    "model_alias": model_alias,
                    "model": model_path,
                    "prompt_version": pv,
                    "started_at_local": started_iso,
                    "updated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        out_f.flush()
    t_write = time.perf_counter() - t_w0

    t_gen = time.perf_counter() - t_loop

    n_out = gen_lines

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    t_total = time.perf_counter() - t_all

    meta = {
        "model_alias": model_alias,
        "model": model_path,
        "llm_backend": backend,
        "prompt_version": pv,
        "total_generated_quadruples": n_out,
        "token_usage": {
            "inference_calls": inference_calls,
            "prompt_tokens_sum": prompt_tokens_sum,
            "completion_tokens_sum": completion_tokens_sum,
            "total_tokens_sum": total_tokens_sum,
            "token_calls_with_total": token_calls_with_total,
            "tokens_per_inference_call": round(total_tokens_sum / token_calls_with_total, 3)
            if token_calls_with_total else None,
            "tokens_per_generated_quadruple": round(total_tokens_sum / n_out, 3)
            if n_out else None,
        },
        "api_guardrail": {
            "safe_retries": api_safe_retries,
            "blocked_skips": api_blocked_skips,
        },
        "v2_quality": {
            "raw_lines_sum": v2_raw_lines_sum,
            "parsed_lines_sum": v2_parsed_lines_sum,
            "kept_lines_sum": v2_kept_lines_sum,
            "drop_bad_format_sum": v2_drop_bad_format_sum,
            "drop_oov_sum": v2_drop_oov_sum,
            "drop_oov_ent_sum": v2_drop_oov_ent_sum,
            "drop_oov_rel_sum": v2_drop_oov_rel_sum,
            "drop_dup_sum": v2_drop_dup_sum,
            "fallback_used_sum": v2_fallback_used_sum,
            "ent_fuzzy_hits_sum": v2_ent_fuzzy_hits_sum,
        },
        "v5_quality": {
            "pivot_yes_sum": v5_pivot_yes_sum,
            "pivot_no_sum": v5_pivot_no_sum,
            "rollout_applied_sum": v5_rollout_applied_sum,
            "rollout_lr": CF_V5_ROLLOUT_LR,
            "rollout_max_slots": CF_V5_ROLLOUT_MAX_SLOTS,
        },
        "wall_time_seconds": round(t_total, 3),
        "model_load_seconds": round(t_load, 3),
        "generation_loop_seconds": round(t_gen, 3),
        "write_file_seconds": round(t_write, 3),
        "input_file": str(Path(input_file).resolve()),
        "dataset_dir": str(Path(input_file).resolve().parent),
        "runs_parent": str(run_parent),
        "run_directory": str(run_dir.resolve()),
        "quadruples_file": str(quadruples_path.resolve()),
        "meta_file": str(meta_path.resolve()),
        "skipped_source_lines": skipped,
        "started_at_local": started_iso,
        "finished_at_local": finished_at,
        "run_id_user": run_id,
    }
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as mf:
        mf.write(meta_json)

    index_row = {
        "run_directory": meta["run_directory"],
        "model_alias": model_alias,
        "model": model_path,
        "llm_backend": backend,
        "prompt_version": pv,
        "total_generated_quadruples": n_out,
        "started_at_local": started_iso,
        "finished_at_local": finished_at,
        "wall_time_seconds": meta["wall_time_seconds"],
        "run_id_user": run_id,
    }
    with open(index_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(index_row, ensure_ascii=False) + "\n")

    print(f"Counterfactual dataset saved to {quadruples_path} ({n_out} lines, {skipped} skipped)")
    print(f"Run directory: {run_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Run index (append): {index_path}")
    print(
        f"[summary] alias={model_alias} | model={model_path} | prompt={pv} | generated={n_out} | "
        f"wall_time={t_total:.2f}s"
    )
    print(
        f"[timing] model_load {t_load:.2f}s | generation {t_gen:.2f}s "
        f"({(t_gen / n_out):.3f}s/line, generated only)" if n_out else f"[timing] model_load {t_load:.2f}s | generation {t_gen:.2f}s"
    )
    print(f"[timing] write {t_write:.3f}s | total {t_total:.2f}s")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRACER counterfactual graph generation (IV.A V5: pivot gate + intervention + rollout)"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dataset directory with train.txt, entity2id.txt, relation2id.txt. Default ../data/ICEWS14",
    )
    p.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=None,
        help="Process only the first n input edges; omit for full run. CF_SAMPLE_SIZE env var also supported (CLI wins)",
    )
    p.add_argument(
        "-m",
        "--model-alias",
        default=None,
        metavar="ALIAS",
        help=(
            "Model alias (for output folder name); must appear in llm.local.json local_models or api_profiles; "
            "or set CFGEN_MODEL_ALIAS"
        ),
    )
    p.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Local vLLM model path or Hugging Face repo ID (bypasses local_models in llm.local.json); "
        "or set CFGEN_MODEL_PATH",
    )
    p.add_argument(
        "--backend",
        choices=("vllm", "api"),
        default=None,
        help="Force vllm or api; if omitted, inferred from which llm.local.json section contains the alias",
    )
    p.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        metavar="FILE",
        help=f"LLM config path, default {DEFAULT_LLM_CONFIG_PATH}; or set CFGEN_LLM_CONFIG",
    )
    p.add_argument(
        "--runs-parent",
        default=None,
        metavar="DIR",
        help=f"Root directory for counterfactual runs (creates a subfolder per run). Default {DEFAULT_RUNS_PARENT}; or CF_RUNS_PARENT",
    )
    p.add_argument(
        "-r",
        "--run-id",
        default=None,
        metavar="ID",
        help="Optional experiment tag written to meta.json / run_index only (not used in folder names). Or CF_RUN_ID",
    )
    p.add_argument(
        "--resume-dir",
        default=None,
        metavar="DIR",
        help="Resume from a previous run dir (needs cf_train.txt; state.json recommended). Continues from processed_edges and appends to cf_train.txt",
    )
    p.add_argument(
        "--test-mode",
        action="store_true",
        help="Print raw output and parse/map stats per sample (use with small -n)",
    )
    p.add_argument(
        "--ent-fuzzy",
        action="store_true",
        help="Enable entity fuzzy matching (rapidfuzz) when exact/normalized match fails; reduces ent_oov (validate with --test-mode first)",
    )
    p.add_argument(
        "--ent-fuzzy-threshold",
        type=int,
        default=97,
        help="Entity similarity threshold 0–100 (higher = stricter). Default 97",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.data_dir is not None:
        train_path = init_cf_dataset(args.data_dir)
    else:
        train_path = TRAIN_FILE

    env_sample = os.environ.get("CF_SAMPLE_SIZE")
    sample = (
        args.sample_size
        if args.sample_size is not None
        else (int(env_sample) if env_sample else None)
    )

    prompt_ver = "v5"

    run_id = args.run_id or os.environ.get("CF_RUN_ID")
    runs_parent = args.runs_parent or os.environ.get("CF_RUNS_PARENT")
    model_alias = args.model_alias or os.environ.get("CFGEN_MODEL_ALIAS")
    model_path_override = args.model_path or os.environ.get("CFGEN_MODEL_PATH")
    llm_config_path = args.llm_config or os.environ.get("CFGEN_LLM_CONFIG")

    main(
        str(train_path),
        runs_parent=runs_parent,
        sample_size=sample,
        prompt_version=prompt_ver,
        run_id=run_id,
        model_alias=model_alias,
        model_path_override=model_path_override,
        backend_override=args.backend,
        llm_config_path=llm_config_path,
        resume_dir=args.resume_dir,
        test_mode=args.test_mode,
        ent_fuzzy=args.ent_fuzzy,
        ent_fuzzy_threshold=args.ent_fuzzy_threshold,
    )
