# Craftax Smoke Test B Candidate (GEPA)

## Context & objective

Implement the smallest honest Craftax leaderboard candidate improvement under Smoke Test B by adding a focused prompt-optimization configuration and a reproducible record bundle, while preserving all shared harness surfaces.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_smoke_test_b.yaml`
   - Question: can a Smoke Test B candidate be introduced without touching shared runtime surfaces?
   - Outcome: supporting for candidate packaging.
   - Evidence: the new config changes only prompt optimization parameters and prompt text.

2. `records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b`
   - Question: is the candidate bundle complete and valid for leaderboard tooling?
   - Outcome: supporting.
   - Evidence: required `metadata.json`, `run_config.yaml`, `metrics.json`, `system_info.json`, and `command.txt` were created and aligned with the config.

3. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: is this candidate compatible with the existing GEPA-based prompt optimization/reflection flow?
   - Outcome: supporting.
   - Evidence: no baseline code changes were required; the candidate follows existing GEPA `optimizer`/verifier contract.

4. `PYTHONPATH=src python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b`
   - Question: does validator feedback confirm the new record structure?
   - Outcome: supporting.
   - Evidence: `{"ok": true, "warnings": []}`.

## Insights

1. The highest-leverage change for this smoke cycle is config-level prompt shaping plus anti-loop language, not harness edits.
2. The candidate remains GEPA-aligned by design and therefore satisfies the optimization-posture requirement.
3. The candidate is explicitly labeled as `Smoke Test B` in `metadata.json` to preserve leaderboard traceability.
4. Reward impact is not measured in this task because execution was kept smoke-scoped.

## Research artifacts produced

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_smoke_test_b.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b/`
- Verifier invocation command and result: `PYTHONPATH=src python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b`
- Repo handoff notes: `findings.txt`

## Quality & validation

- Attempted: `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b`
- Result: failed due local `pyproject.toml` cloud dependency path resolution (`file:///Users/joshpurtell/Documents/GitHub/synth-ai` not present in this environment).
- Executed verifier fallback: `PYTHONPATH=src python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b`
- Result: `{"ok": true, "warnings": []}`.
- Not validated: live rollout score uplift, Modal runtime contract, and GEPA optimization trajectory.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_smoke_test_b.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Candidate label: `Smoke Test B` (from `records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b/metadata.json`).
- Main follow-up: launch the entrypoint and collect a scored `best_eval` if leaderboard lift is needed.
