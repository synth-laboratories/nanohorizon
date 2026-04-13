# Craftax Rollout Evidence Candidate

## Context & Objective

The task was to make a small, reviewable Craftax improvement that strengthens planning clarity and records explicit rollout evidence without changing the leaderboard scoring contract.

The candidate needed to stay inside the existing Craftax rollout path, use `uv` for Python commands, and leave behind verifier evidence plus a reviewable commit/PR.

## Experiments Cited

1. `src/nanohorizon/craftax_core/metadata.py`
   - Question: can the prompt payload carry explicit rollout evidence without changing reward computation?
   - Outcome: supporting.
   - Evidence: `PromptContext.to_prompt_payload()` now emits `rollout_evidence` from `RewardHistoryWindow.evidence_summary()`.

2. `src/nanohorizon/craftax_core/http_shim.py`
   - Question: can the HTTP shim expose the rollout contract cleanly and remain monkeypatchable for verifier tests?
   - Outcome: supporting.
   - Evidence: `create_app()` now serves `/health`, `/task_info`, and `/rollout`; `run_rollout_request()` is a lazy wrapper so tests can stub the rollout path without circular imports.

3. `src/nanohorizon/craftax_core/rollout.py`
   - Question: does the policy-facing prompt include explicit rollout evidence and does the trace store the same context?
   - Outcome: supporting.
   - Evidence: each user prompt now includes a `Recent rollout evidence:` block, and each turn stores a structured `planning_context`.

4. `tests/test_craftax_core_contract.py`
   - Question: do the HTTP shim and rollout prompt carry the new evidence without breaking the contract?
   - Outcome: supporting.
   - Evidence: the verifier test now checks the prompt text and the recorded `planning_context`.

5. `tests/test_craftax_core_runtime_guarantees.py`
   - Question: does the rollout path still consume all model actions and preserve turn semantics after the evidence change?
   - Outcome: supporting.
   - Evidence: the existing rollout tests passed after adding assertions that the evidence block appears in the first and repaired prompts.

6. Baseline vs candidate seeded smoke
   - Question: does the candidate preserve the scoring contract while adding traceable evidence?
   - Outcome: supporting for contract preservation, inconclusive for real reward lift.
   - Evidence: a repeated-seed smoke on seeds `7`, `8`, and `9` executed the checked-in `HEAD` rollout source and the edited rollout source side by side under the same fake policy; both returned reward `0.0`, while only the candidate recorded `planning_context`.

## Insights

1. The smallest useful improvement here is additive: surface evidence to the model and trace it in the rollout artifact, rather than changing reward logic.
2. The prompt is now more readable because the evidence block uses action names instead of raw action indices.
3. The HTTP shim needed a lazy wrapper so the new `/rollout` endpoint stayed testable without a circular import.
4. Reward behavior was unchanged in the seeded smoke, which is the expected result for a traceability-only candidate.

## Research Artifacts Produced

- Code:
  - `src/nanohorizon/craftax_core/http_shim.py`
  - `src/nanohorizon/craftax_core/metadata.py`
  - `src/nanohorizon/craftax_core/rollout.py`
- Verifier coverage:
  - `tests/test_craftax_core_contract.py`
  - `tests/test_craftax_core_runtime_guarantees.py`
  - `tests/test_craftax_interface.py`
- Durable handoff:
  - `findings.txt`
  - `reports/final_report.md`

## Quality & Validation

- Command: `PYTHONPATH=src:. uv run --no-project --with pytest --with fastapi --with httpx --with pillow --with pyyaml --with numpy pytest tests/test_craftax_interface.py tests/test_craftax_core_contract.py tests/test_craftax_core_runtime_guarantees.py`
- Result: `18 passed`
- Command: `PYTHONPATH=src:. uv run --no-project --with pytest --with fastapi --with httpx --with pillow --with pyyaml --with numpy python - <<'PY' ...`
- Result: repeated seeded rollouts for `7`, `8`, and `9` returned `planning_context` with empty prior evidence and reward `0.0` for each seed.
- Command: `PYTHONPATH=src:. uv run --no-project --with pyyaml python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
- Result: `{ "ok": true, "warnings": [] }`
- Not validated: live Craftax reward improvement on a real model endpoint.

## Reproduction & Handoff

- The candidate behavior is in the three Craftax core files listed above.
- The baseline comparison was done by executing `git show HEAD:src/nanohorizon/craftax_core/rollout.py` and `exec`-ing that source side by side with the edited module under the same fake policy and repeated seeds.
- The candidate preserves the outcome contract: `reward_info` is unchanged, and the new evidence only augments prompt text and turn metadata.
- Remaining caveat: the workspace still contains unrelated pre-existing dirty-tree changes outside the files above; they were left untouched.
