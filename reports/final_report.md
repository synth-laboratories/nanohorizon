# Daytona Stack Validation

## Context & objective

The task was to implement a NanoHorizon Craftax leaderboard candidate called `Daytona Stack Validation` with the smallest honest change. The intended improvement was not a broad harness rewrite; it was a compact prompt-side optimization that keeps a private todo scratchpad aligned with the Craftax action loop.

The checkout in this run did not contain the Craftax source tree that the task contract referenced. The only committed file at `HEAD` was `README.md`, and the protected Craftax surfaces were absent. I therefore reconstructed the candidate from an archived NanoHorizon prompt-optimization baseline and kept the scope limited to candidate packaging.

## Experiments cited

1. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: can the compact three-item todo scratchpad be centralized as a reusable candidate helper?
   - Outcome: supporting.
   - Evidence: `TODO_SCRATCHPAD_REQUIREMENTS`, `todo_scratchpad_directive()`, and `build_daytona_stack_validation_seed_prompt()`.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_daytona_stack_validation.yaml`
   - Question: does the candidate config reflect the Daytona Stack Validation idea with a compact scratchpad and a short action-batch policy?
   - Outcome: supporting.
   - Evidence: the seed prompt names the candidate, preserves the three-item todo scratchpad, and keeps the `craftax_interact` contract.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-11_daytona_stack_validation/`
   - Question: is the candidate packaged with an honest run status and reproduction pointer?
   - Outcome: supporting.
   - Evidence: `command.txt`, `metadata.json`, `metrics.json`, `notes.md`, `run_config.yaml`, and `system_info.json`.

## Insights

1. A compact todo scratchpad is the only mechanism added here; no shared Craftax harness surface had to be edited in this checkout.
2. The candidate remains truthful about its status: it is packaged for review, but not executed in this workspace.
3. Because the source tree was missing, the most defensible improvement was to preserve the candidate contract and durable handoff artifacts rather than fabricate an unvalidated runtime change.

## Research artifacts produced

### Environments

- The run happened in the SMR workspace checkout at the initial commit only.
- Archived NanoHorizon workspaces were used as read-only references for the prompt-opt baseline shape.

### Data

- No new training or evaluation data was generated.
- The candidate references the existing prompt-opt starter seed file path in the config.

### Models / checkpoints

- No model weights or checkpoints were trained or promoted in this run.

## Quality & validation

- Validated locally by checking that the candidate helper centralizes the todo contract and that the config includes the Daytona Stack Validation prompt.
- Not validated end-to-end against the Craftax runtime, because the required source tree and harness files were absent from the checkout.

## Reproduction & handoff

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_daytona_stack_validation.yaml`
- Candidate helper: `src/nanohorizon/baselines/prompt_opt.py`
- Record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-11_daytona_stack_validation/`
- Caveat: this is a reconstructed candidate from an archived baseline, not a live verified Craftax rollout in this checkout.

