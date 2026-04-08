# Craftax Dispatch Fix E2E Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the todo-tool idea without changing the protected shared harness surfaces, while fixing the missing end-to-end dispatch rule between the private todo list and the final short Craftax action batch.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the prior prompt-opt record documents a held-out reward regression, so a compact seed-prompt correction is lower risk than editing shared runtime code.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
   - Question: what gap remained after the refresh-gate candidate?
   - Outcome: supporting.
   - Evidence: the prompt already asked the policy to follow the current first todo item, which exposed that the shared prompt-opt source contract still lacked the same dispatch clause.

3. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does prompt optimization preserve a stable todo-tool dispatch contract during GEPA reflection?
   - Outcome: supporting.
   - Evidence: `TODO_SCRATCHPAD_REQUIREMENTS` now includes both the compact internal scratchpad wording and the end-to-end first-item dispatch clause, and `build_reflection_system_directive()` preserves the same framing.

4. `configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml`
   - Question: does the candidate config match the new shared dispatch contract?
   - Outcome: supporting.
   - Evidence: the seed prompt keeps the same model, budget, and rollout settings while instructing the policy to stay on the first todo item until it is satisfied, blocked, or unsafe.

5. `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `run_config.yaml`, `notes.md`, `metrics.json`, `metadata.json`, `system_info.json`, and `command.txt`.

## Insights

1. The narrowest honest improvement here is still prompt and reflection shaping, not a harness edit.
2. The useful part of the todo strategy is preserving one exact private contract across seed prompt, GEPA reflection, rollout feedback, and packaged record.
3. The remaining gap after the refresh-gate candidate was dispatch consistency, not loop-refresh behavior.
4. Reward impact remains unmeasured because this task only performed structural validation.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e/`
- Structural regression test: `tests/test_codex_dispatch_fix_e2e_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `PYTHONPATH=/workspace/src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_dispatch_fix_e2e_candidate.py tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py`
- Result: 9 tests passed.
- Executed: `PYTHONPATH=/workspace/src uv run --no-project python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e`
- Result: `{ "ok": true, "warnings": [] }`
- Executed review check: `git status --short docs/task-craftax.md src/nanohorizon/craftax_core/http_shim.py src/nanohorizon/craftax_core/runner.py src/nanohorizon/craftax_core/metadata.py scripts/run_craftax_model_eval.sh`
- Result: no output; the protected Craftax harness files remained unchanged.
- Verifier-driven review note: the runtime `request_review` tool was not exposed in this session, so readiness review used repo-local verifiers only: targeted pytest, record validation, and protected-file diff checks.
- Validation dead end: plain project-scoped `uv run ...` failed before running tests because the repo references a missing local `synth-ai` checkout at `/Users/joshpurtell/Documents/GitHub/synth-ai`; `uv --no-project` was required for truthful validation in this container.
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search output.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Main risk: the stronger first-item dispatch rule could still overconstrain short tactical batches that should pivot earlier for safety or opportunistic resource pickup.
- Follow-up: if infrastructure is available, run the candidate config against the reference baseline for a real reward comparison.
