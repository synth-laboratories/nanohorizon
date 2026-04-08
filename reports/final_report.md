# Craftax Todo String Fix E2E

## Context & objective

Implement the smallest honest NanoHorizon change for the Craftax todo-tool strategy without editing the protected shared harness surfaces. The target for this run was the `_string Fix E2E` candidate: make the compact private todo/scratchpad contract survive the full prompt-optimization path, not just the reflection instructions.

The required external task doc path `/Users/joshpurtell/Documents/GitHub/nanohorizon/docs/task-craftax.md` was not mounted in this workspace. I used the in-repo [`docs/task-craftax.md`](../docs/task-craftax.md) plus direct source inspection and record this as an environment caveat rather than pretending the read succeeded.

## Experiments cited

1. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does the prompt-opt runtime actually enforce the todo scratchpad string end-to-end?
   - Outcome: supporting.
   - Evidence: added `enforce_todo_scratchpad_contract(...)` and applied it to the seed prompt, the prompt passed into rollout evaluation, and the persisted best candidate.

2. `tests/test_codex_todo_refresh_gate_candidate.py`
   - Question: is the string-preservation fix anchored by a focused regression test?
   - Outcome: supporting.
   - Evidence: the test now checks that the source contains the canonical todo contract helper and that the evaluation and seed-prompt paths call it.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
   - Question: does the pre-existing candidate bundle still validate after the code-side E2E fix?
   - Outcome: supporting for packaging.
   - Evidence: record validation succeeded with no warnings.

4. Verifier-style checks on the current diff
   - Question: is the narrow diff structurally sound?
   - Outcome: mixed.
   - Evidence:
     - `git diff --check -- src/nanohorizon/baselines/prompt_opt.py tests/test_codex_todo_refresh_gate_candidate.py` passed.
     - `uv tool run --with ruff ruff check src/nanohorizon/baselines/prompt_opt.py tests/test_codex_todo_refresh_gate_candidate.py` reported pre-existing import-order / duplicate-import issues in `src/nanohorizon/baselines/prompt_opt.py`.

## Insights

1. The prior todo-refresh candidate relied on reflection instructions to preserve the private todo contract, but the runtime still evaluated and persisted raw candidate strings. This was the actual end-to-end gap.
2. The smallest reviewable fix is to normalize prompts at the prompt-opt boundary, not to edit the Craftax runtime or model-eval harness.
3. This fix strengthens the GEPA-style posture by preserving the intended prompt contract during candidate evolution, while staying entirely inside the prompt-opt baseline.
4. Reward impact is still unmeasured. The evidence from this run is structural and reproducible, not leaderboard performance.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Regression test update: `tests/test_codex_todo_refresh_gate_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed:
  `PYTHONPATH=/workspace/src uv tool run --with pytest --with pyyaml pytest tests/test_codex_todo_refresh_gate_candidate.py`
  Result: `3 passed`.
- Executed:
  `PYTHONPATH=/workspace/src uv tool run --with pyyaml python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
  Result: `{ "ok": true, "warnings": [] }`.
- Executed:
  `git diff --check -- src/nanohorizon/baselines/prompt_opt.py tests/test_codex_todo_refresh_gate_candidate.py`
  Result: passed.
- Verifier-style feedback:
  `uv tool run --with ruff ruff check src/nanohorizon/baselines/prompt_opt.py tests/test_codex_todo_refresh_gate_candidate.py`
  Result: failed on pre-existing `prompt_opt.py` import-layout / duplicate-import issues outside this narrow fix.
- Not validated:
  live Craftax reward, GEPA search output quality, Modal runtime behavior, or leaderboard score.

## Reproduction & handoff

- Change intent: ensure the canonical todo scratchpad requirements survive seed prompt creation, candidate evaluation, and persisted best-prompt output.
- Workspace caveat: plain `uv run ...` is blocked here by the unresolved local dependency `synth-ai @ file:///Users/joshpurtell/Documents/GitHub/synth-ai` in [`pyproject.toml`](../pyproject.toml); used `uv tool run` for targeted reproducible verification instead.
- Review focus:
  - confirm the E2E helper is only applied to the prompt-opt `system_prompt` path
  - confirm the helper does not duplicate the contract when the candidate already preserves it
  - run a real prompt-opt comparison against the reference baseline when the full environment is available
