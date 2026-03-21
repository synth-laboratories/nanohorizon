# RLVR Modal Crafter Baseline Plan

This document is now historical context. The current implementation lives in:

- `src/nanohorizon/rlvr_training.py`
- `src/nanohorizon/modal_rlvr.py`
- `scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh`

This document scoped the initial reference baseline for the `rlvr_20min_2xa100_40gb`
track.

## Goal

Build a real online Crafter RLVR baseline for `Qwen/Qwen3.5-4B` that:

- runs live Crafter rollouts during the budget window
- serves the Crafter runtime from Modal instead of a local process
- keeps the Crafter runtime and learner stack in the same Modal app
- satisfies the Synth container contract used elsewhere in the repo
- produces a reproducible record bundle under `records/rlvr_20min_2xa100_40gb/...`

## Required Architecture

### 1. Crafter service as a Modal-hosted container runtime

Current state:

- the Crafter runtime is implemented in
  `containers/crafter_rs/src/main.rs`
- the offline baseline runs it locally and talks to it over HTTP

Target state:

- package the Crafter Rust runtime into a Modal image
- expose the same HTTP surface from Modal
- keep the service Synth-compatible:
  - `GET /health`
  - `GET /task_info`
  - `POST /rollout`
  - checkpoint / resume routes already present in the runtime

This preserves compatibility with the Synth container expectations while removing
the local-process dependency from the RLVR track.

### 2. RLVR runner in the same Modal app

Current state:

- `src/nanohorizon/modal_rlvr.py` just shells out to
  `python -m nanohorizon.baselines.rlvr`
- `src/nanohorizon/baselines/rlvr.py` consumes a static rollout JSONL

Target state:

- define a Crafter service function/class in `modal_rlvr.py`
- define a learner function/class in `modal_rlvr.py`
- have the learner talk to the colocated Crafter service over the in-app Modal
  address instead of a local tunnel

This should make rollout traffic simpler and remove the tunnel requirement for the
RLVR reference path.

### 3. Simple TRL-based online baseline

Initial baseline should stay simple:

- single student model: `Qwen/Qwen3.5-4B`
- LoRA adapters only
- collect live rollouts
- convert them into weighted training examples
- train with a straightforward TRL/SFT-style update loop or a simple
  reward-weighted objective
- publish/evaluate the updated adapter periodically or at the end of the run

The first baseline does not need NanoLong-style distributed RL. It needs a clear,
defensible online reference that really interacts with Crafter during the run.

## Concrete File Targets

- `src/nanohorizon/modal_rlvr.py`
  - stop treating RLVR as a static-JSONL job
  - own the Modal app layout for both Crafter service and learner
- `src/nanohorizon/baselines/rlvr.py`
  - replace static JSONL-only flow with live rollout collection + online updates
- `configs/crafter_rlvr_qwen35_4b_2xa100_20min.yaml`
  - add rollout concurrency, service config, eval config, and publish cadence
- `scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh`
  - keep the public entrypoint stable
- `docs/tracks/rlvr_20min_2xa100_40gb.md`
  - update the reference stack description once implemented

## Constraints

- keep the public track script stable
- keep the Crafter service Synth-compatible
- no local tunnel requirement for the RLVR reference runner
- no dependency on `nanolong`; use it only as a comparison/reference
- default target remains `Qwen/Qwen3.5-4B`

## First Milestones

1. Build and launch `containers/crafter_rs` inside Modal.
2. Verify the Modal-hosted Crafter service responds to `health`, `task_info`, and
   `rollout`.
3. Replace static rollout JSONL in the RLVR baseline with live rollout collection.
4. Add a simple TRL-based update loop over collected weighted examples.
5. Add final held-out evaluation and record bundle writing.
