# RLVR 20min 2xA100 40GB

This is the online RLVR-style benchmark track for NanoHorizon.

## Contract

- base model: `Qwen/Qwen3.5-0.8B`
- environment: Crafter
- hardware: 2x A100 40GB
- wall-clock budget: 20 minutes

## Allowed During The Budget Window

- live rollout collection
- model-in-the-loop environment interaction
- online reward computation
- RLVR or RL-style adapter updates

## Expected Starter Script

```bash
./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh
```

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- optional large artifacts referenced by manifest
