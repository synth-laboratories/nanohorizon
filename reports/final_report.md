# Final report: Craftax Spark v4 E2E

## Context & objective
- Implement a compact TODO/scratchpad mechanism for Craftax subgoal tracking (`Spark v4 E2E`) with minimal surface changes.
- Preserve shared harness surfaces unless required; implement rollout-side scratchpad support and package a reproducible candidate record.

## Experiments cited
1. `tests/test_codex_spark_v4_candidate.py::test_spark_v4_candidate_config_uses_compact_todo_scratchpad`
   - Question: is the candidate prompt carrying private, compact todo constraints and no-op safety wording?
   - Outcome: supporting.
   - Evidence: `configs/craftax_prompt_opt_qwen35_4b_spark_v4_e2e.yaml`, `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/`.

2. `tests/test_codex_spark_v4_candidate.py::test_rollout_source_tracks_todo_scratchpad_state`
   - Question: does harness source expose scratchpad refresh/state support?
   - Outcome: supporting.
   - Evidence: `src/nanohorizon/craftax_core/rollout.py`.

3. `tests/test_craftax_core_contract.py::test_rollout_includes_private_todo_scratchpad_and_refreshes_on_stagnation`
   - Question: does rollout inject and track `todo_scratchpad` over turns under repeated no-progress paths?
   - Outcome: supporting.
   - Evidence: synthetic turn-level assertions in the test plus updated rollout helper logic.

4. `python -m pytest tests/test_craftax_core_contract.py tests/test_codex_spark_v4_candidate.py`
   - Question: do related tests pass end-to-end and what is still blocking?
   - Outcome: 9 passed, 1 failed.
   - Evidence: test output from this run.
   - Remaining failure: `test_http_shim_health_and_task_info` in `tests/test_craftax_core_contract.py` due missing local `craftax` dependency.

## Insights
1. Scratchpad tracking can be integrated in rollout generation without editing protected files (`http_shim.py`, `runner.py`, `metadata.py`, `scripts/run_craftax_model_eval.sh`, `docs/task-craftax.md`).
2. Turn-level `todo_scratchpad` in trace enables explicit stagnation feedback and gives verifiable control over subgoal refresh behavior.
3. Verifier-driven validation is currently constrained by environment dependencies; structure and prompt/rollout contracts are passing while runtime-shim health checks require install-time fixes.

## Research artifacts produced
- Environment
  - Local regression verification via `python -m pytest` (direct Python path required because `uv run` in this workspace resolves to a stale local path and exits with "Distribution not found at: file:///Users/joshpurtell/Documents/GitHub/synth-ai").
- Data
  - No new training data created.
- Candidate and model artifacts
  - Candidate config: `configs/craftax_prompt_opt_qwen35_4b_spark_v4_e2e.yaml`
  - Record bundle:
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/metadata.json`
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/metrics.json`
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/notes.md`
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/run_config.yaml`
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/command.txt`
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/system_info.json`

## Quality & validation
- Passed checks:
  - `python -m pytest tests/test_craftax_core_contract.py -k "todo_scratchpad" tests/test_codex_spark_v4_candidate.py` -> 3 passed, 7 deselected.
  - `python -m pytest tests/test_craftax_core_contract.py tests/test_codex_spark_v4_candidate.py` -> 9 passed, 1 failed.
- Explicit blocker:
  - `test_http_shim_health_and_task_info` still fails in this environment because importing `craftax` is unavailable.

## Reproduction & handoff
- Commit: `bd9c1b4c7ff682b7569c860ca97e94be9925c199`
- Push branch: `spark-v4-e2e-todo-scratchpad`
- PR: https://github.com/synth-laboratories/nanohorizon/pull/9
- Candidate execution entrypoint retained as not-yet-run marker bundle for future scorer runs (status `not_run`).
- Open risks:
  - missing `craftax` package in the run environment can block `/health` contract verification and full rollout smoke tests;
  - candidate leaderboard uplift is not yet measured in this run.
