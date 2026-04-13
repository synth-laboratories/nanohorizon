# Craftax Candidate Run

## Context & Objective

This run aimed to produce the smallest honest NanoHorizon Craftax candidate that could plausibly improve performance without changing the shared Craftax harness surfaces.

The candidate hypothesis was narrow:

- keep the private todo scratchpad contract
- tighten the policy prompt to an exact 4-action batch
- make the prompt more state-driven: `do` for adjacent useful targets, craft/upgrade when inventory supports it, otherwise explore

The protected surfaces named in the task were preserved:

- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

## Experiments Cited

1. [`configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`](../configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml)
   - Question: does tightening the Craftax prompt to an exact 4-action batch and a more state-driven progression heuristic still preserve the intended todo contract?
   - Outcome: supporting at the prompt-shaping level.
   - Evidence: the seed prompt now requires exactly three private todo items, exact 4-action output, and explicit progression rules.

2. [`tests/test_codex_todo_refresh_gate_candidate.py`](../tests/test_codex_todo_refresh_gate_candidate.py)
   - Question: does the repo still encode the candidate prompt/record expectations after the wording update?
   - Outcome: supporting.
   - Evidence: the focused regression test passed after the prompt and record status updates.

3. [`experiments/craftax_candidate_impl/experiment_log.txt`](../experiments/craftax_candidate_impl/experiment_log.txt)
   - Question: can the existing prompt-opt eval path run a baseline-vs-candidate comparison in this workspace?
   - Outcome: blocked.
   - Evidence: the direct prompt-opt CLI reached GEPA, but reflection repeatedly failed with `404 modal-http: invalid function call`.

4. [`records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/metrics.json`](../records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/metrics.json)
   - Question: what is the prior reference score for this prompt-opt family?
   - Outcome: supporting baseline context.
   - Evidence: `primary_score` is `0.35`, which is the last completed reference score in the repo.

## Insights

1. The smallest plausible candidate here is still a prompt-shaping change, not a harness rewrite.
2. The exact-4-action rule is a useful sharpening because the rollout harness already targets 4-action batches.
3. The live comparison path is currently blocked by the rollout/reflection endpoint, not by the candidate prompt itself.
4. Because no candidate reward run completed, there is no honest score improvement to report.

## Research Artifacts Produced

- Candidate config: [`configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`](../configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml)
- Candidate regression test: [`tests/test_codex_todo_refresh_gate_candidate.py`](../tests/test_codex_todo_refresh_gate_candidate.py)
- Candidate record bundle: [`records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/`](../records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/)
- Durable experiment log: [`experiments/craftax_candidate_impl/experiment_log.txt`](../experiments/craftax_candidate_impl/experiment_log.txt)
- Record status updates: [`records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/metadata.json`](../records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/metadata.json), [`records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/metrics.json`](../records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/metrics.json), [`records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/notes.md`](../records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/notes.md)

## Quality & Validation

- `uv run pytest tests/test_codex_todo_refresh_gate_candidate.py`
- `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
- The live prompt-opt comparison attempt failed during GEPA reflection with `404 modal-http: invalid function call`

## Reproduction & Handoff

- Candidate prompt validation: `uv run pytest tests/test_codex_todo_refresh_gate_candidate.py`
- Record validation: `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
- Comparison attempt:
  - `NANOHORIZON_PROMPT_OPT_CONTAINER_URL=direct://local uv run --group modal --group classic python -m nanohorizon.baselines.prompt_opt --config configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml --output-dir experiments/craftax_candidate_impl/baseline_correct_2026-04-13 --inference-url https://synth-laboratories--nanohorizon-craftax-prompt-opt-p-a5e20a-dev.modal.run --request-model qwen35-4b-prompt-opt`
- Open risk: the prompt-opt eval path needs a reachable rollout/reflection endpoint or Modal bootstrap/auth to finish a real reward comparison.

