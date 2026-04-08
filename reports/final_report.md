# Craftax Dispatch Fix E2E Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the todo-tool idea without changing the protected shared harness surfaces, while fixing one remaining end-to-end gap: the shared prompt-opt contract did not yet preserve the dispatch rule that keeps a short action batch committed to the current first todo item.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the prior prompt-opt record documents a regression, so a compact seed-prompt correction is a lower-risk change than editing shared runtime code.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
   - Question: what gap remained after the refresh-gate candidate?
   - Outcome: supporting.
   - Evidence: the candidate already asked the policy to follow the current first todo item, which exposed that the shared source contract still lacked the same dispatch clause.

3. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does prompt optimization preserve a stable todo-tool contract during GEPA reflection?
   - Outcome: supporting.
   - Evidence: the source centralizes the private three-item scratchpad requirements in `TODO_SCRATCHPAD_REQUIREMENTS` and now includes the end-to-end dispatch clause reused in reflection instructions and rollout feedback.

4. `configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml`
   - Question: does the candidate add the smallest config-level change that matches the new shared dispatch contract?
   - Outcome: supporting.
   - Evidence: the prompt keeps the same todo refresh behavior but asks the policy to dispatch the full 3-4 action batch from the first todo item until it is satisfied, blocked, or unsafe.

5. `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `run_config.yaml`, `notes.md`, `metrics.json`, `metadata.json`, `system_info.json`, and `command.txt`.

## Insights

1. The narrowest honest improvement here is still prompt and reflection shaping, not a harness edit.
2. The useful part of the todo strategy is not just naming subgoals, but preserving one exact private three-item contract across seed prompt, GEPA reflection, and rollout feedback.
3. The smallest remaining issue after the refresh-gate candidate was not loop refresh but dispatch consistency between prompt seed, shared GEPA contract, and packaged record.
4. Preserving the first-item dispatch clause inside the shared source contract is more defensible than editing the runtime or adding broader search changes.
5. Reward impact is still unmeasured because this task only performed structural validation.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e/`
- Structural regression test: `tests/test_codex_dispatch_fix_e2e_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_dispatch_fix_e2e_candidate.py tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py`
- Result: 9 tests passed.
- Executed: `PYTHONPATH=src uv run --no-project python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_dispatch_fix_e2e`
- Result: `{ "ok": true, "warnings": [] }`
- Executed review check: `git diff --name-only -- docs/task-craftax.md src/nanohorizon/craftax_core/http_shim.py src/nanohorizon/craftax_core/runner.py src/nanohorizon/craftax_core/metadata.py scripts/run_craftax_model_eval.sh`
- Result: no output; the protected Craftax harness files remained unchanged.
- Verifier-driven review note: the runtime `request_review` tool was not exposed in this session, so readiness review used repo-local verifiers only: targeted pytest, record validation, and protected-file diff checks.
- Validation dead end: plain project-scoped `uv run ...` failed before running tests because the repo references a missing local `synth-ai` checkout at `/Users/joshpurtell/Documents/GitHub/synth-ai`; `uv --no-project` was required for truthful validation in this container.
- Reviewable commit: finalized via the required `workspace_push` flow outside this static report body; inspect the run handoff for the exact pushed commit outcome.
- Push flow: this report intentionally records the code and validation state only; the backend-tracked push result is reported separately in the run handoff.
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search output.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Main risk: the stronger first-item dispatch rule could still overconstrain short tactical batches that should pivot earlier for safety or opportunistic resource pickup.
- Push artifact: inspect the run handoff for the final backend-tracked branch and commit outcome.
- Recommended verifier focus:
  - confirm the centralized todo contract remains present in reflection instructions
  - inspect whether the first-item dispatch wording is compact enough to avoid overlong reasoning
  - if infrastructure is available, run the candidate config against the reference baseline for a real reward comparison
