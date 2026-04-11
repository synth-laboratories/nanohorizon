# Final Report

## Context & Objective

This run targeted the NanoHorizon Craftax prompt-opt path. The objective was the smallest honest improvement for the leaderboard candidate, centered on a compact todo scratchpad contract that stays private to the policy and is refreshed every turn.

## Experiments Cited

1. `tests/test_auto_push_e2e_candidate.py`
   - Question: Does the candidate package the todo-scratchpad strategy without changing the shared harness surfaces?
   - Outcome: Supporting.
   - Evidence: `configs/craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml`, `src/nanohorizon/baselines/prompt_opt.py`, `records/prompt_opt_1usd_gpt54_family/2026-04-11_auto_push_e2e/`.

## Insights

1. Centralizing the scratchpad contract in `prompt_opt.py` keeps the candidate change narrow and easy to review.
2. The candidate remains a packaging and prompt-shaping change only; no SFT, RL, or live reward run was performed.

## Research Artifacts Produced

- `src/nanohorizon/baselines/prompt_opt.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/http_shim.py`
- `scripts/run_craftax_model_eval.sh`
- `scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- `configs/craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml`
- `records/prompt_opt_1usd_gpt54_family/2026-04-11_auto_push_e2e/`

## Quality & Validation

- Structural validation only through `tests/test_auto_push_e2e_candidate.py`.
- The prompt-opt wrapper command is now present as a dry-run/config inspection script.
- No live Craftax rollout, Modal execution, or GEPA optimization run was executed.

## Reproduction & Handoff

- Candidate config command:
  - `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
  - `uv run --with pytest pytest -q tests/test_auto_push_e2e_candidate.py`
  - `./scripts/run_craftax_model_eval.sh`
- Residual risk:
  - The added end-to-end handoff wording may be slightly restrictive for some short tactical batches, but it keeps the candidate honest and localized.
- Publication blocker:
  - The branch is committed and pushed, but `create_github_pr` rejected every attempted repo slug as not present in the configured GitHub repos list, so a real PR could not be opened from this session.
