# NanoLong

NanoLong is a small long-horizon training and evaluation lab.

The goal is to facilitate iterating on long-horizon RL algorithms

The repo is organized around one idea:

- environments expose a common container contract
- models interact with those environments through OpenAI-compatible inference
- rollouts are saved in a trainer-ready JSONL format
- post-training code consumes those rollouts for SFT or RL

## Quick Start

If you want one entrypoint that runs the real Crafter long-horizon RL job end to end on `Qwen/Qwen3.5-0.8B`, use:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/crafter_multistep/run_non_stacked_qwen08b_full_job.sh
```

That script:

- starts the local native Crafter runtime
- runs the non-stacked NanoLong long-horizon CISPO-style RL loop on `Qwen/Qwen3.5-0.8B`
- saves the full run result
- runs a final eval batch against the final trained model before exit

If you want the smaller smoke path instead, use:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/crafter_multistep/run_modal_qwen9b_rl.sh
```

## Repo Map

- [containers](/Users/joshpurtell/Documents/GitHub/nanolong/containers)
  - First-class environment backends and the container contract.
- [examples](/Users/joshpurtell/Documents/GitHub/nanolong/examples)
  - Runnable end-to-end examples for each environment.
- [training](/Users/joshpurtell/Documents/GitHub/nanolong/training)
  - Inference, SFT, and RL code.
- [go_explore](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore)
  - Canonical home for Go-Explore, including the active Crafter-first runner and the migrated legacy multi-environment stack.
- [reference](/Users/joshpurtell/Documents/GitHub/nanolong/reference)
  - Vendored/reference runtimes and upstream source material.
- [tests](/Users/joshpurtell/Documents/GitHub/nanolong/tests)
  - Focused verification for the training and serving substrate.
- [evals](/Users/joshpurtell/Documents/GitHub/nanolong/evals)
  - Throughput-oriented evaluation work.

## First-Class Containers

NanoLong currently supports these first-class rollout backends:

- Crafter
- Netter
- ALFWorld
- NLE
- Tic-Tac-Toe

The contract for a first-class container is documented in [containers/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/containers/README.md).

In practice, a container is the environment runtime. It is responsible for:

- exposing `GET /health`, `GET /task_info`, and `POST /rollout`
- stepping the environment
- calling a provided OpenAI-compatible model endpoint during rollout
- computing rewards inside the environment
- returning trainer-ready rollout JSONL records

## Where To Hack

If you want to change training behavior, these are the main files to edit:

- RL trainer core: [training/posttrain/rl/rl.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/rl.py)
- RL orchestration and async prototypes:
  - [training/posttrain/rl/train_rl_modal.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/train_rl_modal.py)
  - [training/posttrain/rl/stacked_modal_rl.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/stacked_modal_rl.py)
  - [training/posttrain/rl/non_stacked_modal_rl.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/non_stacked_modal_rl.py)
- SFT data path: [training/posttrain/sft/common.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/sft/common.py)
- SFT trainer entrypoint: [training/posttrain/sft/train_sft_modal.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/sft/train_sft_modal.py)
- Rollout-to-SFT export: [training/posttrain/sft/export_rollouts_to_sft.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/sft/export_rollouts_to_sft.py)
- Modal inference serving:
  - [training/inference/serve_vllm_modal.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/serve_vllm_modal.py)
  - [training/inference/serve_vllm_modal_qwen35_9b.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/serve_vllm_modal_qwen35_9b.py)
- LoRA publish/load flow:
  - [training/inference/publish_lora_modal.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/publish_lora_modal.py)
  - [training/inference/modal_handle.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/modal_handle.py)

If you want to change environment behavior, edit the container itself:

- Crafter: [reference/crafter/crafter_rs_container](/Users/joshpurtell/Documents/GitHub/nanolong/reference/crafter/crafter_rs_container)
- Netter: [reference/netter/src/bin/netter-container.rs](/Users/joshpurtell/Documents/GitHub/nanolong/reference/netter/src/bin/netter-container.rs)
- ALFWorld: [containers/alfworld/alfworld_container.py](/Users/joshpurtell/Documents/GitHub/nanolong/containers/alfworld/alfworld_container.py)
- NLE: [containers/nle/nle_container.py](/Users/joshpurtell/Documents/GitHub/nanolong/containers/nle/nle_container.py)
- Tic-Tac-Toe: [containers/tictactoe/tictactoe_container.py](/Users/joshpurtell/Documents/GitHub/nanolong/containers/tictactoe/tictactoe_container.py)

## Main Entrypoints

The easiest way to use the repo is through the example folders in [examples](/Users/joshpurtell/Documents/GitHub/nanolong/examples).

Each example typically includes:

- `run_eval.sh`
- `run_openai_eval.sh` or a provider-specific eval script
- `run_rl.sh`
- sometimes a provider-specific RL script

Useful starting points:

- Crafter: [examples/crafter_multistep/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/crafter_multistep/README.md)
- Go-Explore Crafter: [examples/go_explore_crafter/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/go_explore_crafter/README.md)
- Netter: [examples/netter_rlvr/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/netter_rlvr/README.md)
- ALFWorld: [examples/alfworld_multistep/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/alfworld_multistep/README.md)
- NLE: [examples/nle_multistep/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/nle_multistep/README.md)
- Tic-Tac-Toe: [examples/tictactoe_smoke/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/tictactoe_smoke/README.md)

## Common Workflows

### 1. Run an environment eval

Use one of the example runners, for example:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/tictactoe_smoke/run_openai_eval.sh
```

That flow:

- starts or talks to the local environment service
- points it at a model endpoint
- saves rollout artifacts under that example's `artifacts/` directory

### 2. Run a tiny RL smoke

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/tictactoe_smoke/run_rl.sh
```

This uses saved rollout JSONL and feeds it into the RL trainer.

For Crafter specifically, the main RL entrypoints are:

- full `Qwen/Qwen3.5-0.8B` long-horizon job: [examples/crafter_multistep/run_non_stacked_qwen08b_full_job.sh](/Users/joshpurtell/Documents/GitHub/nanolong/examples/crafter_multistep/run_non_stacked_qwen08b_full_job.sh)
- hosted-model smoke path: [examples/crafter_multistep/run_modal_qwen9b_rl.sh](/Users/joshpurtell/Documents/GitHub/nanolong/examples/crafter_multistep/run_modal_qwen9b_rl.sh)
- local mock-policy path: [examples/crafter_multistep/run_rl.sh](/Users/joshpurtell/Documents/GitHub/nanolong/examples/crafter_multistep/run_rl.sh)

### 3. Run a Go-Explore Crafter search

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
./examples/go_explore_crafter/run_smoke.sh
./examples/go_explore_crafter/run_prototype.sh
```

That flow:

- starts the existing local mock OpenAI-compatible policy server
- starts the local `crafter-rs` container
- runs the NanoLong-local Go-Explore search loop
- writes candidate rankings, evaluation results, and summaries under `examples/go_explore_crafter/artifacts/`

The full migrated Go-Explore codebase now lives in [go_explore](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore). The old multi-environment implementation is preserved under [go_explore/legacy](/Users/joshpurtell/Documents/GitHub/nanolong/go_explore/legacy), but the recommended path for new work is the Crafter-first local runner at the top of that package.

### 4. Run the core RL trainer directly

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
uv run --python 3.11 --group training python training/posttrain/rl/rl.py \
  --config tests/hello_world_train_infer/configs/rl_tiny_train.yaml
```

The RL trainer details live in [training/posttrain/rl/README.txt](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/README.txt).

### 5. Serve a model on Modal

```bash
cd /Users/joshpurtell/Documents/GitHub/nanolong
uv run --python 3.11 --group cloud python training/inference/serve_vllm_modal_qwen35_9b.py
```

For local code changes, the main serving entrypoints are:

- [training/inference/serve_vllm_modal.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/serve_vllm_modal.py)
- [training/inference/serve_vllm_modal_qwen35_9b.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/inference/serve_vllm_modal_qwen35_9b.py)

## How The Pieces Fit

The normal loop looks like this:

1. A first-class container receives a rollout request.
2. The container calls an OpenAI-compatible model endpoint for each decision step.
3. The environment computes rewards and writes a rollout artifact.
4. Example scripts save those rollouts under `examples/*/artifacts/`.
5. Training code reads the rollout JSONL for SFT or RL.
6. Updated LoRA adapters can be served again through the inference layer.

## If You Are New Here

Start in this order:

1. Read [containers/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/containers/README.md)
2. Read [examples/README.md](/Users/joshpurtell/Documents/GitHub/nanolong/examples/README.md)
3. Run one small example like [examples/tictactoe_smoke](/Users/joshpurtell/Documents/GitHub/nanolong/examples/tictactoe_smoke)
4. Read [training/posttrain/rl/README.txt](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/README.txt)
5. Edit [training/posttrain/rl/rl.py](/Users/joshpurtell/Documents/GitHub/nanolong/training/posttrain/rl/rl.py) or one of the container files depending on what you want to change
