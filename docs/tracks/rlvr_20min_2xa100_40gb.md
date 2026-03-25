# RLVR 20min 2xA100 40GB

This is the online RLVR-style benchmark track for NanoHorizon.

**Track ID:** `rlvr_20min_2xa100_40gb` (used in `records/<track_id>/…`)

## Contract

- base model: `Qwen/Qwen3.5-4B`
- environment: Craftax
- hardware: 2x A100 40GB
- wall-clock budget: 20 minutes

## Allowed During The Budget Window

- live rollout collection
- model-in-the-loop environment interaction
- online reward computation
- RLVR or RL-style adapter updates

## Reference Stack

- one Modal app with two runtime surfaces:
  - Craftax service on CPU
  - one clustered learner-plus-inference runtime spanning 2x A100 40GB
- single-script RLVR logic in `src/nanohorizon/baselines/rlvr.py`
- track-owned Modal runtime lives in `src/nanohorizon/baselines/rlvr.py`
- public runner in `./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh`
- default GPU: `A100-40GB` via `NANOHORIZON_MODAL_GPU_RLVR`

Reference algorithm shape:

- grouped on-policy rollout generation
- environment-computed rewards
- group-relative reward normalization
- sequence-level clipped GRPO-style objective
- LoRA adapter reload between rollout waves

Current record status:

- implementation is in repo
- checked-in clustered reference smoke run:
  [../../records/rlvr_20min_2xa100_40gb/2026-03-21_reference_baseline/](../../records/rlvr_20min_2xa100_40gb/2026-03-21_reference_baseline/)
- checked-in score: `0.0`
- longer canonical scoring runs are still expected to improve on this baseline

Historical design notes:

- [rlvr_modal_craftax_baseline_plan.md](rlvr_modal_craftax_baseline_plan.md)

## Expected Starter Script

```bash
./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh
```

Config override pattern:

```bash
NANOHORIZON_RLVR_CONFIG=configs/craftax_rlvr_qwen35_4b_validation_smoke.yaml \
./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh
```

## Edit Surface

Change only:

- `src/nanohorizon/baselines/rlvr.py`

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- optional large artifacts referenced by manifest

## How To Read A Run

- final benchmark score: `metrics.json -> final_mean_outcome_reward`
- bootstrap eval: `periodic_eval/step_000/summary.json`
- post-update evals: `periodic_eval/step_001/summary.json`, `step_002`, ...
- training-wave detail: `iteration_summaries.json`
- raw rollout traces: `iterations/iter_XXX/rollouts.jsonl` when included

Interpretation rule:

- use periodic eval summaries to judge improvement
- do not infer hillclimbing from training rollout rewards alone
- a good RLVR run should keep later periodic eval means above `step_000` on the same held-out seeds

Current reference interpretation:

- the checked-in clustered smoke record proves the runtime and transport path
- it does not prove reward lift
- a later 10-step exploratory run reached non-zero periodic eval reward (`0.5`) before falling back to `0.0`
