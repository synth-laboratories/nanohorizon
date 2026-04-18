# NanoHorizon Submission Eval Report

## Context

This run targeted a single-file leaderboard candidate in `submission/agent.py`.
The goal was to turn the submission into a stronger Craftax policy candidate and
run a lightweight honest eval on the curated train seeds.

## Change summary

- Updated the submission prompt to use the stronger private three-item todo
  contract already used in the prompt-opt candidate configs.
- Aligned the submission defaults with the prompt-opt rollout shape:
  - `enable_thinking = True`
  - `thinking_budget_tokens = 2000`
  - `max_new_tokens = 3072`
  - `target_action_batch_size = 4`
  - `min_action_batch_size = 3`
- Added submission-specific score aliases in the eval result payload:
  - `submission_mean_outcome_reward`
  - `submission_achievement_frequencies`

## Validation

Validation was **resource-blocked / inconclusive**.

Observed issues:

- The first direct eval attempt failed because `submission/agent.py` passed
  unsupported `evaluate_model()` kwargs for action batch sizes.
- After removing those kwargs, local rollout execution was blocked because the
  workspace did not have the `craftax` dependency group installed.
- I verified that `uv` could materialize `craftax` in an isolated env, but the
  next local eval pass hit additional dependency/runtime friction and was
  stopped per operator instruction.

What was confirmed:

- `submission/agent.py` now parses and compiles.
- The candidate change stays inside the allowed submission surface.
- The train-seed eval path is wired to the curated seed manifest in
  `data/craftax/craftax_prompt_opt_starter_seeds.json`.

What was not validated:

- No completed Craftax rollout score was produced in this workspace.
- No held-out leaderboard uplift was measured here.

## Residual risk

The candidate is a prompt-and-runtime-shape improvement, but its actual score
impact remains unmeasured in this run because local Craftax rollout execution
was blocked before a successful eval completed.
