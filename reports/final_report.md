# Final report: Craftax Spark v4 E2E

## Context & objective
- Task: implement a focused, higher-confidence Craftax candidate (`Spark v4 E2E`) centered on compact TODO/scratchpad tracking in the harness workflow.
- Constraints followed: keep changes minimal, avoid editing shared harness files unless required, no SFT/RL, and keep candidate in prompt/rollout strategy with verifier-driven checks.
- Candidate state was packaged as not-yet-run (`candidate_not_run`) for reproducible follow-up scoring.

## Experiments cited
1. `tests/test_codex_spark_v4_candidate.py::test_spark_v4_candidate_config_uses_compact_todo_scratchpad`
   - Question: Is the Spark v4 candidate spec carrying the intended compact TODO/scratchpad constraints?
   - Outcome: Supporting. Assertions pass on seed prompt wording and record metadata integrity.
   - Evidence: `configs/craftax_prompt_opt_qwen35_4b_spark_v4_e2e.yaml`, `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/`.

2. `tests/test_craftax_core_contract.py::test_rollout_includes_private_todo_scratchpad_and_refreshes_on_stagnation`
   - Question: Is scratchpad state injected into the rollout prompt and tracked across turns with no-progress refresh behavior?
   - Outcome: Supporting. Test passes with synthetic stale-trajectory path.
   - Evidence: `src/nanohorizon/craftax_core/rollout.py`, test output logs from `python -m pytest ...`.

3. `python -m pytest tests/test_craftax_core_contract.py tests/test_codex_spark_v4_candidate.py`
   - Question: Does the change pass current regression surface and what are remaining blockers?
   - Outcome: 9 passed, 1 failed.
   - Evidence: test run output in this run notes.
   - Failure was in an existing shim dependency path (`craftax` module missing in env), not introduced by Spark v4 files.

## Insights
1. Scratchpad support can be added in rollback-safe fashion inside rollout logic without touching the protected shared files listed in the task contract.
2. Persisting `todo_scratchpad` per turn in trace improves auditability and enables verifiable stagnation handling.
3. In this workspace the strongest immediate verifier signal is structural consistency and contract tests; live rollout scoring is still blocked by environment/package availability for Craftax tests.

## Research artifacts produced
- Environments
  - Local repo tests executed via Python (`python -m pytest`) due `uv` project-resolution failure in this workspace.
- Data
  - No new training data generated in this task; candidate is prompt/rollout logic and record packaging.
- Models / checkpoints
  - No model weights produced in this run.
  - Candidate record references run command at:
    - `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/run_config.yaml`

## Quality & validation
- Controlled checks passed:
  - `python -m pytest tests/test_craftax_core_contract.py -k "todo_scratchpad" tests/test_codex_spark_v4_candidate.py` → 3 passed, 7 deselected.
  - `python -m pytest tests/test_craftax_core_contract.py tests/test_codex_spark_v4_candidate.py` → 9 passed, 1 failed.
- Failure details:
  - `test_http_shim_health_and_task_info` fails because environment lacks `craftax` (`ModuleNotFoundError: No module named 'craftax'`) when `/health` calls texture cache initialization.

## Reproduction & handoff
- Reviewable commit and PR creation are pending after workspace push step.
- Artifact locations:
  - Candidate config: `configs/craftax_prompt_opt_qwen35_4b_spark_v4_e2e.yaml`
  - Rollout source changes: `src/nanohorizon/craftax_core/rollout.py`
  - Tests: `tests/test_craftax_core_contract.py`, `tests/test_codex_spark_v4_candidate.py`
  - Record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/*`
  - Execution notes: `findings.txt`
- Command bundle stored in record: `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e/command.txt`
