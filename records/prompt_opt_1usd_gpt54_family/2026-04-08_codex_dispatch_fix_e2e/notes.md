Prompt-opt candidate for the Craftax "Todo Tool" strategy with an end-to-end dispatch fix.

Hypothesis:
- The todo-refresh candidates already encourage tracking the active blocker and replacing stale targets, but the shared prompt-opt contract still lets GEPA reflection drift away from an explicit action-dispatch rule.
- Preserving one extra rule that keeps the full 3-4 action batch committed to the current first todo item, while explicitly naming the compact scratchpad slots, should reduce mixed-purpose batches and repeated loops without touching the shared Craftax runtime or eval harness.

What changed:
- Added the action-dispatch rule plus the compact-scratchpad slot wording to `src/nanohorizon/baselines/prompt_opt.py` so rollout feedback and reflection preserve them as part of the shared todo-tool contract.
- Added a prompt-opt candidate config that asks the policy to keep spending actions on the first todo item until it is satisfied, blocked, or unsafe before using the rest of the batch on lower-priority work, while keeping the hidden scratchpad in a compact ordered three-slot form.
- Packaged a matching candidate record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` frames Craftax as a multi-step decision task where small planning improvements are valuable under fixed runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` already centralizes todo refresh and loop-break guidance, which made the missing action-dispatch clause the smallest end-to-end gap left in the preserved contract.
- `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml` already tested "follow the first todo item" in a candidate seed prompt, but that rule was not yet preserved in the shared source contract used by GEPA reflection and rollout feedback.
- `tests/test_codex_durable_intent_candidate.py` also expects the shared prompt-opt source to keep compact-scratchpad wording, so the candidate now restores that signal instead of letting the reflection contract drift.

Validation performed in this task:
- Structural source/config/record validation only. No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- The stronger first-item dispatch rule could overconstrain short tactical batches that should pivot early for safety or opportunistic resource pickup.
- This change improves contract consistency, not the underlying reward signal; if GEPA search noise dominates, the practical gain may be small.
- Because no live rollout run was executed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.
