# Craftax Todo String Fix E2E Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the todo-tool idea without changing the protected shared harness surfaces, while tightening the prompt-opt todo contract so GEPA reflection preserves compact todo items and one end-to-end action batch. In this run, the concrete repo-task outcome was to make the 2026-04-08 candidate package self-consistent and leave an honest readiness verdict.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the prior prompt-opt record documents a regression from `bootstrap_score=0.6` to `primary_score=0.35`, so a compact seed-prompt correction is a lower-risk change than editing shared runtime code.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does prompt optimization preserve a stable todo-tool contract during GEPA reflection?
   - Outcome: supporting.
   - Evidence: the source now centralizes the private three-item scratchpad requirements in `TODO_SCRATCHPAD_REQUIREMENTS`, including short-string todo items and one end-to-end action batch aligned to the first todo item, and reuses them in reflection instructions and rollout feedback.

3. `configs/craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml`
   - Question: does the candidate add a compact but stricter todo-string / end-to-end batch variant?
   - Outcome: supporting.
   - Evidence: the prompt now keeps each todo item as a short string fragment, refreshes todo items every turn, replaces stale targets after no-progress loops, and asks for one short end-to-end action batch aligned to the current first todo item.

4. `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `run_config.yaml`, `notes.md`, `metrics.json`, `metadata.json`, `system_info.json`, `command.txt`, `verifier_review.md`, and the newly materialized config/test files the bundle references.

5. Live runtime probe of the previously recorded Modal Qwen endpoint
   - Question: can this workspace prove runtime availability or reward readiness for the candidate?
   - Outcome: negative.
   - Evidence: `/v1/models` returned `HTTP 404`, and `/v1/chat/completions` with the repo-default key returned `modal-http: invalid function call`.

## Insights

1. The narrowest honest improvement here is still prompt and reflection shaping, not a harness edit.
2. The useful part of the todo strategy is not just naming subgoals, but preserving one exact private three-item contract across seed prompt, GEPA reflection, and rollout feedback.
3. The smallest remaining todo-only gap after the refresh-gate candidate was prompt compactness: keeping todo items as short strings reduces the chance that private planning drifts into verbose prose.
4. Asking for one end-to-end batch aligned to the first todo item is a reviewable extension of the refresh-gate idea, but reward impact is still unmeasured because this task only performed structural validation.
5. The candidate package is now reviewable, but the honest verifier verdict is still `not_ready` until a live Qwen/Craftax comparison is run.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e/`
- Structural regression test: `tests/test_codex_todo_string_fix_e2e_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `PYTHONPATH=/workspace/src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_todo_string_fix_e2e_candidate.py`
- Result: superseded by the broader post-fix run below.
- Executed: `PYTHONPATH=/workspace/src uv tool run --with pytest --with pyyaml pytest tests/test_codex_todo_string_fix_e2e_candidate.py tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py`
- Result: 9 tests passed.
- Executed: `PYTHONPATH=/workspace/src uv run --no-project --with pyyaml python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e`
- Result: superseded by the no-sync project-module run below.
- Executed: `PYTHONPATH=/workspace/src uv run --no-sync python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e`
- Result: `{ "ok": true, "warnings": [] }`
- Verifier review: `records/prompt_opt_1usd_gpt54_family/2026-04-08_codex_todo_string_fix_e2e/verifier_review.md`
- Verifier result: packaging is consistent, but the candidate remains `not_ready` because no live runtime or reward comparison was available.
- Default `uv run ...` project sync is blocked here because `uv.lock` resolves `synth-ai` to a missing local path (`../synth-ai`).
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search output.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Main risk: the stronger short-string and end-to-end wording could overconstrain otherwise good short tactical action batches.
- Push artifact: inspect the run handoff for the final backend-tracked branch and commit outcome.
- Recommended verifier focus:
  - confirm the newly added config/test stay aligned with the 2026-04-08 record bundle
  - confirm the centralized todo contract remains present in reflection instructions
  - inspect whether the short-string wording reduces prompt drift without under-specifying state
  - if infrastructure is available, run the candidate config against the reference baseline for a real reward comparison
