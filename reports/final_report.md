# Craftax Prompt Improvement Report

## Context and objective

Task: improve the NanoHorizon Craftax prompt so the policy reasons before emitting the final action-array tool call, while keeping shared Craftax harness surfaces unchanged.

This run stayed scoped to the prompt-optimization lane and did not modify the preserved harness files:

- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

## Files changed

- `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`
- `configs/craftax_prompt_opt_qwen35_4b_gepa_smoke.yaml`
- `configs/craftax_prompt_opt_qwen35_4b_gepa_eval20.yaml`
- `configs/craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml`
- `src/nanohorizon/baselines/prompt_opt.py`
- `tests/test_craftax_prompt_opt_config.py`
- `findings.txt`
- `reports/final_report.md`

## What changed

The prompt contract was normalized across the Craftax prompt-optimization configs so the policy now explicitly:

- reasons briefly and privately before acting
- checks the nearest useful affordance
- reads recent action history before repeating a move
- checks whether the trajectory is looping
- favors early-game progress and nearby resource collection
- keeps JSON/prose out of assistant content outside the final tool call

The prompt-optimizer reflection template and rollout-feedback text were updated to preserve that same contract during GEPA prompt rewrites.

## Validation performed

Successful validation:

1. `PYTHONPATH=/workspace/src uv --no-config tool run --isolated --with pytest --with pyyaml pytest /workspace/tests/test_craftax_prompt_opt_config.py`
   Result: `1 passed in 0.05s`
2. `PYTHONPATH=/workspace/src uv --no-config tool run --isolated --with pyyaml python - <<'PY' ... PY`
   Confirmed all four prompt-opt configs now contain both `recent action history` and `outside the tool call`.

Blocked validation:

1. `uv run pytest /workspace/tests/test_craftax_prompt_opt_config.py /workspace/tests/test_pivot_verifier_baseline.py`
2. `uv run --no-default-groups --group dev ...`

Both `uv run` attempts failed before test execution because the repo lockfile resolves an unavailable local-path dependency from `pyproject.toml`:

- `synth-ai @ file:///Users/joshpurtell/Documents/GitHub/synth-ai`

## Verifier feedback consulted

Used the repo-native pivot verifier rubric in `src/nanohorizon/baselines/pivot_verifier.py` as the acceptance lens before declaring the prompt ready. The relevant verifier criteria are:

- reward reasoning that is present
- reward explicit plans or justifications
- reward reasoning grounded in observed state / nearby affordances / chosen actions
- reward reasoning that references the proposed action
- penalize passive or strategically disconnected behavior
- prefer actions that unlock achievements, increase prerequisite resources, or advance tool progression

That verifier guidance is consistent with the prompt update: the new wording explicitly asks the agent to reason about affordances, recent action history, looping, and shortest concrete progress before the final action tool call.

## Assumptions and caveats

- The repo-local `docs/task-craftax.md` copy was used because the absolute path in the task statement was not mounted in this workspace.
- No live Craftax rollout evaluation was run in this turn.
- The `uv run` path is currently unreliable in this checkout until the missing local `synth-ai` source is made available or removed from the resolved environment.

## Git / review

- Commit: pending
- PR: pending
