# TRACER V5 — Intervention Pivot Gate (IV.A.1)

Decides whether the current fact is a worthwhile intervention pivot `F_int`. The model outputs **only** `YES` or `NO`.

```
You are an expert in temporal knowledge graph counterfactual construction. Decide whether the event below is a worthwhile intervention pivot F_int for TRACER (Intervention Pivot Discovery).

[Event under review] time slice {t_id}: subject [{s_name}] — action [{r_name}] — object [{o_name}]

[Local history F_hist (same subject–object pair, distant to recent)]
{historical_context}

[Auxiliary signals]
- Subject historical participation (out-degree + in-degree count): {s_degree}
- Object historical participation: {o_degree}

[Pivot criteria] (more satisfied → more likely YES)
1) The event may be a strategic turning point, escalation/de-escalation, or have cascade potential in local evolution;
2) The relation/action is non-trivial (not a repetitive boilerplate statement); a rewrite yields a meaningful counterfactual contrast;
3) Avoid: pure background noise, events highly redundant with history, or rewrites that cannot plausibly chain forward.

[Output]
Output exactly one word: **YES** or **NO** (uppercase, no explanation).
```
