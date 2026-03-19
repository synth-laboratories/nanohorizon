# Offline 20min 1xA100 40GB

This is the fixed-data benchmark track for NanoHorizon.

## Contract

- base model: `Qwen/Qwen3.5-0.8B`
- environment: Crafter
- hardware: 1x A100 40GB
- wall-clock budget: 20 minutes

## Allowed During The Budget Window

- SFT on fixed data
- offline RL on fixed data
- reward-weighted filtering
- deterministic preprocessing of precomputed data
- generating fresh SFT rows with `Qwen/Qwen3.5-27B`, as long as that generation happens inside the same 20-minute budget window and the final trained policy remains `Qwen/Qwen3.5-0.8B`

## Reference Stack

- RunPod for execution
- `vllm serve Qwen/Qwen3.5-27B` for teacher inference
- TRL `SFTTrainer` for `Qwen/Qwen3.5-0.8B`

## Not Allowed During The Budget Window

- new environment interaction
- new rollout generation
- calling arbitrary live teachers to label fresh data outside the declared `Qwen/Qwen3.5-27B` allowance above

## Expected Starter Script

```bash
./scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh
```

## Main Reference Run

If you want the full intended end-to-end user flow, edit and run:

```bash
./scripts/run_crafter_offline_reference.sh
```

That file is the main surface a competitor should tweak for the reference baseline.

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- optional large artifacts referenced by manifest
