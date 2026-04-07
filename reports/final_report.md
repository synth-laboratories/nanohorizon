# Craftax Todo Refresh Gate Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the todo-tool idea without changing the protected shared harness surfaces, while making the prompt-opt reflection path preserve the same scratchpad contract used by the candidate prompt.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the prior prompt-opt record documents a regression, so a compact seed-prompt correction is a lower-risk change than editing shared runtime code.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does prompt optimization preserve a stable todo-tool contract during GEPA reflection?
   - Outcome: supporting.
   - Evidence: the source now centralizes the private three-item scratchpad requirements in `TODO_SCRATCHPAD_REQUIREMENTS` and reuses them in reflection instructions and rollout feedback.

3. `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
   - Question: does the candidate add a compact but stricter loop-break / action-gating variant?
   - Outcome: supporting.
   - Evidence: the prompt now refreshes todo items every turn, replaces stale targets after no-progress loops, and asks the short action batch to follow the current first todo item.

4. `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `run_config.yaml`, `notes.md`, `metrics.json`, `metadata.json`, `system_info.json`, and `command.txt`.

## Insights

1. The narrowest honest improvement here is still prompt and reflection shaping, not a harness edit.
2. The useful part of the todo strategy is not just naming subgoals, but preserving one exact private three-item contract across seed prompt, GEPA reflection, and rollout feedback.
3. A small extra constraint that ties the 3-4 action batch to the active first todo item is worth packaging as a separate candidate because it is reviewable and easy to measure later.
4. Reward impact is still unmeasured because this task only performed structural validation.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/`
- Structural regression test: `tests/test_codex_todo_refresh_gate_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `uv run pytest tests/test_codex_todo_refresh_gate_candidate.py`
- Result: 3 tests passed.
- Executed: `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
- Result: `{ "ok": true, "warnings": [] }`
- Reviewable commit: finalized via the required `workspace_push` flow outside this static report body; inspect the run handoff for the exact pushed commit outcome.
- Push flow: this report intentionally records the code and validation state only; the backend-tracked push result is reported separately in the run handoff.
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search output.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Main risk: the stronger "follow the first todo item" wording could overconstrain otherwise good short tactical action batches.
- Push artifact: inspect the run handoff for the final backend-tracked branch and commit outcome.
- Recommended verifier focus:
  - confirm the centralized todo contract remains present in reflection instructions
  - inspect whether the follow-the-first-item wording is compact enough to avoid overlong reasoning
  - if infrastructure is available, run the candidate config against the reference baseline for a real reward comparison
