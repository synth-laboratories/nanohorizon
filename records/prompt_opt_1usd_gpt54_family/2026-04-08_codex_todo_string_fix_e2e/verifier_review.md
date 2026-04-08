Verifier review for `codex_todo_string_fix_e2e`

Scope checks:
- Confirmed the protected Craftax harness files were not edited: `docs/task-craftax.md`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`, `src/nanohorizon/craftax_core/metadata.py`, and `scripts/run_craftax_model_eval.sh`.
- Confirmed the candidate stays inside the prompt-opt surface: `src/nanohorizon/baselines/prompt_opt.py`, one new config, one new record bundle, one focused test, and handoff docs.

Consistency checks:
- The shared `TODO_SCRATCHPAD_REQUIREMENTS` now encodes the same short-string and end-to-end constraints used in the candidate config.
- The record bundle reproduces the candidate prompt, cites the reference baseline regression (`0.6 -> 0.35`), and remains marked `candidate_not_run`.
- The missing config and regression test were added in this run so the record bundle is now self-consistent with its `command.txt` and report claims.
- Focused regression test coverage matches the intended candidate contract.

Validation checks:
- `PYTHONPATH=/workspace/src uv tool run --with pytest --with pyyaml pytest tests/test_codex_todo_string_fix_e2e_candidate.py tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py`
- `PYTHONPATH=/workspace/src uv run --no-sync python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e`
- `git diff --check -- src/nanohorizon/baselines/prompt_opt.py configs/craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e tests/test_codex_todo_string_fix_e2e_candidate.py findings.txt reports/final_report.md`
- `curl -I -m 10 https://synth-laboratories--nanohorizon-craftax-prompt-opt-p-a5e20a-dev.modal.run/v1/models`
- `curl -sS -m 20 https://synth-laboratories--nanohorizon-craftax-prompt-opt-p-a5e20a-dev.modal.run/v1/chat/completions -H 'Authorization: Bearer nanohorizon-prompt-opt-key' ...`

Findings:
- Packaging bug fixed: the record bundle previously referenced a config and regression test that were not present in the repo.
- Remaining blocker: no live runtime or reward comparison was verified in this workspace. The previously recorded Modal endpoint returned `HTTP 404` on `/v1/models` and `modal-http: invalid function call` on `/v1/chat/completions`, so the candidate is still `not_ready` for leaderboard submission.

Residual risk for follow-up evaluation:
- The short-string and end-to-end wording may overconstrain tactical flexibility, so the next real reward comparison should be against `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`.
