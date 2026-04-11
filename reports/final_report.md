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
3. The recorded wrapper confirms the candidate config still advertises the todo contract: `SEED_PROMPT_HAS_TODO=True`.

## Research Artifacts Produced

- `src/nanohorizon/baselines/prompt_opt.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/http_shim.py`
- `scripts/run_craftax_model_eval.sh`
- `configs/craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml`
- `records/prompt_opt_1usd_gpt54_family/2026-04-11_auto_push_e2e/`

## Quality & Validation

- Structural validation only through `tests/test_auto_push_e2e_candidate.py`, which passed with `4 passed`.
- Recorded wrapper validation through `uv run bash scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`, which reported `SEED_PROMPT_HAS_TODO=True`.
- No live Craftax rollout, Modal execution, or GEPA optimization run was executed.

## Reproduction & Handoff

- Candidate verification command:
  - `uv run --with pytest pytest -q tests/test_auto_push_e2e_candidate.py`
- Recorded wrapper command:
  - `uv run bash scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Preserved surface:
  - `./scripts/run_craftax_model_eval.sh`
- Residual risk:
  - No live rollout was executed, so score impact remains unmeasured.
