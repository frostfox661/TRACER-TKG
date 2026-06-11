# TRACER V5 — High-Contrast Intervention + Short-Horizon Rollout (IV.A.2)

Called after the pivot is `YES`. Structured multi-line output:

- Line 1: `INTERVENTION||subject||relation||object`
- Following lines: `ROLLOUT||time_slice||subject||relation||object` (time slice must match the predefined slot)

```
You are a TRACER framework expert. The event below has been accepted as intervention pivot F_int. Complete IV.A: high-contrast intervention + short-horizon rollout.

[Intervention pivot F_int] time slice {t_id}: {s_name} || {r_name} || {o_name}

[Local history F_hist]
{historical_context}

[High-contrast intervention] Produce F_cf: substantively different from F_int in event semantics but plausible; subject/relation/object may change; vocabulary must be closed over the candidate sets.

[Short-horizon rollout] For the **existing fact slots** below, provide counterfactual rewrites F_fut in the short post-intervention window (time slices are fixed; do not change time; subject/relation/object may change):
{rollout_slots}

If no rollout slots apply, output only the intervention line.

[Relation candidates]
{relation_candidates}

[Entity candidates]
{entity_candidates}

[Output format (strict)]
- Line 1: `INTERVENTION||subject||relation||object`
- Each following line per slot: `ROLLOUT||time_slice||subject||relation||object` (time slice must match the slot)
- No other text.
```
