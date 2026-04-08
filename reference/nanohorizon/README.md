# NanoHorizon Unique-Achievement GEPA

This lane is the simpler first step for NanoHorizon Craftax GEPA research.

The core question is:

- With the baseline script, how many unique achievements do we get on the
  held-out set?
- If we make the optimization target "maximize unique achievements," can local
  GEPA using `gemini-2.5-flash-lite` as the policy improve that held-out result
  within 500 training rollouts?

## Primary goals

1. Lift
- Measure the baseline held-out unique-achievement outcome.
- Run GEPA with `unique achievements` as the target objective.
- Report baseline vs optimized held-out unique-achievement performance.

2. Scientific quality
- Preserve the experiment trail and show what prompt or algorithm changes were
  explored.
- Distinguish real held-out lift from GEPA internal search-score movement.

3. Throughput / engineering
- Measure how long the 500-rollout GEPA run took.
- Judge whether this is a practical first hill-climb objective.

## Canonical source roots

- `nanohorizon/src/nanohorizon/baselines/prompt_opt.py`
- `nanohorizon/scripts/run_craftax_prompt_opt_gemini25_flash_lite_local.sh`
- `nanohorizon/configs/craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml`
- `nanohorizon/docs/tracks/prompt_opt_1usd_gpt54_family.md`

The staged packet includes these references, and the full `nanohorizon` repo is
also available in the writable workspace.

## Workspace contract

- Treat the current checkout as the canonical writable workspace.
- Treat the literal staged packet path provided by the task, such as
  `{dataset_ref}/...`, as the canonical packet location.
- Do not assume the packet lives at a hardcoded `/workspace/...` path.
- Do not try to recover a missing packet by cloning a repo manually.
- Do not borrow artifacts from another run directory under `/synth/state/.out/...`.
- If the prompt-opt packet or checkout is unavailable after inspecting the
  current checkout root truthfully, fail the task rather than inventing a
  substitute workspace.

## Required experiment shape

1. Run the local Gemini prompt-opt baseline path first.
2. Record the baseline held-out unique-achievement outcome and achievement
   frequency summary.
3. Run one GEPA optimization pass with a training budget capped at 500
   rollouts, using unique achievements as the main target objective.
4. Re-evaluate the optimized result on the same held-out set.
5. Write a final report comparing baseline vs optimized performance and runtime.

## Baseline validity rule

The only valid baseline for this lane is a fresh local Gemini prompt-opt baseline.

Accepted baseline evidence:

- a fresh baseline measurement produced from:
  - `nanohorizon/scripts/run_craftax_prompt_opt_gemini25_flash_lite_local.sh`
  - `nanohorizon/configs/craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml`

Invalid baseline evidence:

- `offline_*` records
- `rlvr_*` records
- checked-in prompt-opt reference bundles from other model families
- any non-prompt-opt Craftax track
- any baseline borrowed from another lane just because it contains achievement
  frequencies

## What counts as success

A strong final bundle should show:

- grounded baseline unique-achievement measurements
- a real local GEPA attempt with a 500-rollout cap
- positive held-out lift in unique-achievement outcome
- supporting achievement-frequency evidence
- a clear runtime / throughput story
