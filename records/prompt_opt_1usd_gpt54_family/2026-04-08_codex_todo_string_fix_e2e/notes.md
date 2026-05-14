Prompt-opt candidate for the Craftax "Todo Tool" strategy with short-string todo items and a tighter end-to-end action batch.

Hypothesis:
- The existing todo-refresh candidate improves loop handling, but the private plan can still bloat into verbose internal prose or split a 3-4 action batch across unrelated subgoals.
- Forcing each todo item to stay a short string and asking for one end-to-end batch aligned to the first todo item should preserve the todo structure under GEPA reflection with less prompt drift.

What changed:
- Tightened the shared `TODO_SCRATCHPAD_REQUIREMENTS` in `src/nanohorizon/baselines/prompt_opt.py` so reflection and rollout feedback now preserve short-string todo items and one end-to-end batch aligned to the active first item.
- Added a prompt-opt candidate config that keeps the same model, optimizer budget, seed split, and rollout shape as the prior todo-refresh candidate while adding the short-string and end-to-end wording to the seed prompt.
- Packaged a matching candidate record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/metrics.json` shows the reference prompt-opt run regressed from `0.6` bootstrap reward to `0.35`, so the safest next step remains a narrow prompt-only correction rather than a harness change.
- `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml` already improved refresh and stale-target replacement, making prompt compactness and batch alignment the smallest remaining todo-only gap.
- `src/nanohorizon/baselines/reflexion.py` already treats useful memory outputs as short strings, which supports mirroring that compactness constraint in the todo scratchpad.

Validation performed in this task:
- Review pass against the candidate diff found no blocking scope or consistency issues; protected Craftax harness files stayed untouched.
- Structural source/config/record validation only.
- No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- The short-string instruction could make the private todo items too terse in states where richer internal grounding would help.
- The end-to-end batch wording could overcommit to one subgoal when a mixed tactical batch would be safer.
- Because no live rollout run was executed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.
