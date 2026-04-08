# Smoke smoke-1775661501

## Context and objective

The task was to make the smallest reviewable Craftax improvement in the NanoHorizon repo by changing the Craftax prompt so the agent is explicitly guided to reason before producing the final action array/tool call, while leaving shared harness surfaces stable.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: what is the checked-in prompt-opt reference baseline and what measured result does it already claim?
   - Outcome: supporting context. The reference prompt is simple and its checked-in metrics show `bootstrap_score=0.6` versus `primary_score=0.35`, which means prompt changes can regress and should not be oversold.
   - Evidence: `metrics.json`, `prompt_bundle.json`.
2. `records/prompt_opt_1usd_gpt54_family/2026-04-08_smoke_1775661501_reasoning_prompt`
   - Question: does the new candidate prompt explicitly require reasoning before the action array while preserving the tool-call contract?
   - Outcome: supporting for structural correctness, not for reward improvement.
   - Evidence: `prompt_bundle.json`, `artifacts/prompt_audit.json`, `run_config.yaml`.

## Insights

1. The narrowest candidate surface is the prompt-opt seed prompt config, not the shared Craftax harness. Supported by the new config and the fact that protected files were left unchanged.
2. The baseline prompt says "Think carefully" but does not explicitly require private reasoning before choosing the action array, nor does it ground that reasoning in state/resource/action-history context. Supported by `artifacts/prompt_audit.json`.
3. This candidate is structurally stronger but performance-unverified. Supported by the reference regression in `2026-03-21_reference_baseline/metrics.json` and the absence of a fresh rollout/eval bundle for this run.

## Research artifacts produced

- Environment: repo-local workspace on commit base `c1d840d`, validated with `uv --no-project` because the checked-in workspace environment references a missing local `synth-ai` path dependency.
- Config: `configs/craftax_prompt_opt_qwen35_4b_reasoning_smoke_1775661501.yaml`.
- Record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-08_smoke_1775661501_reasoning_prompt/`.
- Handoff notes: `findings.txt`.

## Quality and validation

- Validated record shape with `uv run python -m nanohorizon.shared.validate_record ...`.
- Used a deterministic prompt audit to compare baseline and candidate prompt contract coverage.
- Did not run live Craftax rollouts, held-out eval, or model-serving checks in this task.

## Reproduction and handoff

- Validate the candidate record:
  `uv run --no-project env PYTHONPATH=src python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-08_smoke_1775661501_reasoning_prompt`
- Inspect the prompt delta:
  `records/prompt_opt_1usd_gpt54_family/2026-04-08_smoke_1775661501_reasoning_prompt/prompt_bundle.json`
- Main risk: real reward impact is unknown until the candidate is run through the Craftax eval path.
