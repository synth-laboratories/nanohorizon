# Classic

This is the `classic` benchmark track for NanoHorizon.

**Track ID:** `classic` (used in `records/<track_id>/...`)

## Contract

- task: Craftax-Classic via the JAX `craftax` package
- benchmark tier: `1M`
- training regime: RL from random initialization
- policy size: strictly under `100M` parameters
- methods/hardware: separate from the Crafter Modal/Qwen tracks
- runtime shape: no container abstractions and no OpenAI-compatible policy serving layer

## Required Constraints

- do not use the in-repo Crafter-RS runtime
- do not treat this as a variant of `rlvr_20min_2xa100_40gb`
- use a random-init RL policy rather than a pretrained Qwen policy
- keep the tracked model under `100M` parameters
- do not depend on Modal, container tunnels, or the repo's LLM rollout plumbing

## Starter Assets

- config: [../../configs/classic_craftax_1m_random_init.yaml](../../configs/classic_craftax_1m_random_init.yaml)
- local runner: [../../scripts/run_classic_craftax_1m.sh](../../scripts/run_classic_craftax_1m.sh)
- Modal runner: [../../scripts/run_classic_craftax_1m_modal.sh](../../scripts/run_classic_craftax_1m_modal.sh)
- baseline: [../../src/nanohorizon/baselines/classic.py](../../src/nanohorizon/baselines/classic.py)
- Modal wrapper: [../../src/nanohorizon/baselines/classic_modal.py](../../src/nanohorizon/baselines/classic_modal.py)

## Current Repo Status

- the track contract is now checked in
- the starter runner includes a PPO-RNN `1M` baseline config
- the baseline writes a checkpoint and runs a compiled chunked parallel eval pass
- the Modal path now prewarms the Craftax texture cache and reuses a persistent JAX compilation cache
- no checked-in reference baseline record exists yet

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- optional training artifacts referenced by manifest
