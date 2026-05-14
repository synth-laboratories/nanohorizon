Verifier review for `codex_todo_string_fix_e2e`

Scope checks:
- Confirmed the protected Craftax harness files were not edited: `docs/task-craftax.md`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`, `src/nanohorizon/craftax_core/metadata.py`, and `scripts/run_craftax_model_eval.sh`.
- Confirmed the candidate stays inside the prompt-opt surface: `src/nanohorizon/baselines/prompt_opt.py`, one new config, one new record bundle, one focused test, and handoff docs.

Consistency checks:
- The shared `TODO_SCRATCHPAD_REQUIREMENTS` now encodes the same short-string and end-to-end constraints used in the candidate config.
- The record bundle reproduces the candidate prompt, cites the reference baseline regression (`0.6 -> 0.35`), and remains marked `candidate_not_run`.
- Focused regression test coverage matches the intended candidate contract.

Validation checks:
- `PYTHONPATH=/workspace/src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_todo_string_fix_e2e_candidate.py`
- `PYTHONPATH=/workspace/src uv run --no-project --with pyyaml python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e`
- `git diff --check -- src/nanohorizon/baselines/prompt_opt.py configs/craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e tests/test_codex_todo_string_fix_e2e_candidate.py findings.txt reports/final_report.md`

Findings:
- No blocking review findings.

Residual risk for follow-up evaluation:
- The short-string and end-to-end wording may overconstrain tactical flexibility, so the next real reward comparison should be against `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`.
