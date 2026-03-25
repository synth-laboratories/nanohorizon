# Craftax-Classic Task

The `classic` NanoHorizon track uses the JAX `craftax` package rather than the in-repo Craftax runtime.

This task is intentionally separate from the Craftax tracks:

- environment family: `Craftax-Classic`
- package/runtime: `craftax` on top of JAX
- training regime: RL from random initialization
- model regime: small policies only, capped below `100M` parameters
- runtime shape: no container abstraction, no OpenAI-compatible policy server, no Modal requirement

## Why A Separate Task

Craftax-Classic is close enough to Craftax to be familiar, but different enough that it should not share a leaderboard, runtime, or method contract with the existing Craftax/Qwen tracks.

The track is for:

- random-init RL methods
- small-model experimentation
- JAX-native environment throughput
- methods and hardware choices that do not fit the Modal + Qwen + Craftax runtime stack

The track is not for:

- the repo's container abstractions
- the in-repo Craftax rollout runtime
- LLM-in-the-loop rollouts
- OpenAI-compatible inference endpoints

## Environment Contract

- use the upstream `craftax` package
- use the Craftax-Classic environment, not the in-repo Python Craftax rollout runtime
- target the published `1M` benchmark regime
- keep the training loop local and JAX-native rather than routing through container services

The official Craftax docs describe Craftax as a JAX RL environment, note that `pip install craftax` is the standard install path, and distinguish Craftax-Classic from the full Craftax environment.

## Policy Contract

- initialize from scratch
- use an RL training loop
- keep the policy under `100M` parameters

## Starter Surface

- config: [configs/classic_craftax_1m_random_init.yaml](/Users/joshpurtell/Documents/GitHub/nanohorizon/configs/classic_craftax_1m_random_init.yaml)
- local runner: [scripts/run_classic_craftax_1m.sh](/Users/joshpurtell/Documents/GitHub/nanohorizon/scripts/run_classic_craftax_1m.sh)
- Modal runner: [scripts/run_classic_craftax_1m_modal.sh](/Users/joshpurtell/Documents/GitHub/nanohorizon/scripts/run_classic_craftax_1m_modal.sh)
- Python entrypoint: [src/nanohorizon/baselines/classic.py](/Users/joshpurtell/Documents/GitHub/nanohorizon/src/nanohorizon/baselines/classic.py)
- Modal entrypoint: [src/nanohorizon/baselines/classic_modal.py](/Users/joshpurtell/Documents/GitHub/nanohorizon/src/nanohorizon/baselines/classic_modal.py)

The current in-repo starter now includes the classic PPO-RNN baseline plus the same fast-path ideas that proved out in the Craftax reportbench reference work:

- honest training timing via `jax.block_until_ready`
- compiled chunked evaluation rather than a Python step loop
- Modal image prewarm for the Craftax texture cache
- persistent JAX compilation cache on the shared Modal artifact volume

It still stays outside the repo's containerized environment abstractions.
