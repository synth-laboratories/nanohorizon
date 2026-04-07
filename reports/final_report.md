# Craftax Todo-Tool Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the durable-intent / todo-list idea without changing the protected shared harness surfaces.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the prior prompt-opt record documents a regression, so a compact seed-prompt correction is a lower-risk change than editing shared runtime code.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_durable_intent_fix.yaml`
   - Question: does the candidate now express durable intent more explicitly?
   - Outcome: supporting.
   - Evidence: the prompt now asks for a private three-item todo list, refreshes completed items every turn, and swaps targets after repeated no-progress loops.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_durable_intent_fix`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `run_config.yaml`, `notes.md`, `metrics.json`, `metadata.json`, `system_info.json`, and `command.txt`.

## Insights

1. The narrowest durable-intent implementation here is a dedicated prompt-opt config refinement plus matching record-bundle updates.
2. The useful part of the todo strategy is not just naming subgoals, but explicitly refreshing completed items and replacing stale targets when loops repeat.
3. Reward impact is still unmeasured because this task only performed structural validation.

## Research artifacts produced

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_durable_intent_fix.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_durable_intent_fix/`
- Structural regression test: `tests/test_codex_durable_intent_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `uv run pytest tests/test_codex_durable_intent_candidate.py`
- Result: 3 tests passed.
- Executed: `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_durable_intent_fix`
- Result: `{ "ok": true, "warnings": [] }`
- Push flow: `workspace_push` failed with `workspace_push requires a live worker session; no live worker handle was found`.
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search output.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_durable_intent_fix.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Main risk: the hidden todo list could add reasoning overhead without lifting reward.
- Operational blocker: backend-tracked branch push could not be completed from this run because the required worker push handle was unavailable.
- Recommended verifier focus:
  - confirm the candidate bundle remains self-consistent
  - inspect whether the refresh/swap wording is compact enough to avoid overlong reasoning
  - if infrastructure is available, run the candidate config against the reference baseline for a real reward comparison
