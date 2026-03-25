# Offline 20min 1xA100 40GB

This is the filtered behavior cloning benchmark track for NanoHorizon.

**Track ID:** `offline_20min_1xa100_40gb` (used in `records/<track_id>/…`)

## Contract

- base model: `Qwen/Qwen3.5-4B`
- environment: Craftax
- hardware: 1x A100 40GB
- wall-clock budget: 20 minutes

## Allowed During The Budget Window

- SFT on fixed data
- offline RL on fixed data
- reward-weighted filtering
- deterministic preprocessing of precomputed data
- generating fresh rollout traces and converting the high-reward traces into SFT rows during the same 20-minute budget window, as long as the final trained policy remains `Qwen/Qwen3.5-4B`

## Reference Stack

- local Craftax rollout collection and filtering
- Modal teacher inference with `Qwen/Qwen3.5-9B`
- Modal SFT for `Qwen/Qwen3.5-4B`
- async parallel rollout collection with explicit concurrency and permit caps
- reward-based filtering before training
- TRL `SFTTrainer` for `Qwen/Qwen3.5-4B`
- local held-out evaluation against the local Craftax container using remote Modal inference
- held-out base-vs-finetuned compare defaults to `20` rollouts at concurrency `10`
- default example GPU: `A100-40GB` via `NANOHORIZON_MODAL_GPU_OFFLINE`

## Not Allowed During The Budget Window

- calling arbitrary live teachers to label fresh data outside the declared teacher allowance above

## Expected Starter Script

```bash
./scripts/run_offline_training.sh
```

## Main Reference Run

If you want the full intended end-to-end user flow, run:

```bash
./scripts/run_offline_training.sh
```

Edit [src/nanohorizon/baselines/offline_sft.py](/Users/joshpurtell/Documents/GitHub/nanohorizon/src/nanohorizon/baselines/offline_sft.py) to change the learning logic. Run [run_offline_training.sh](/Users/joshpurtell/Documents/GitHub/nanohorizon/scripts/run_offline_training.sh) to handle the full end-to-end flow: local Craftax service, local rollout collection, Modal inference and SFT, final evals, and base-vs-finetuned comparison.

The reference script runs async parallel Craftax rollouts, filters to high-reward tool-calling traces, fine-tunes the student, and compares base versus fine-tuned evaluation on held-out seeds with `thinking_budget_tokens = 2000`.

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- optional large artifacts referenced by manifest
