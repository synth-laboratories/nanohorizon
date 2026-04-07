# CPT -> RLVR Qwen3.5 0.8B Plan

This document proposes a new NanoHorizon track family built around:

- true dense continued pretraining (CPT) on `Qwen/Qwen3.5-0.8B`
- fixed-budget RLVR with LoRA on top of the dense CPT checkpoint
- clean comparison against a no-CPT RLVR baseline

This is a planning document, not a finalized public track contract yet.

## Framework Recommendation

Recommended first implementation:

- dense CPT in an external pretraining stack
- RLVR stays in-repo in `nanohorizon`
- handoff artifact is a dense model checkpoint that `nanohorizon` can treat as
  the RL base model

Recommended pretraining framework for the first version:

- `NeMo` / `Megatron Bridge`

Reasoning:

- we need true dense CPT, not LoRA adaptation
- current official PyTorch docs position `TorchTitan` as the PyTorch pretraining
  framework, but the examples we could verify are Llama-centric
- current NVIDIA docs already expose Qwen-family pretraining configs and data
  pipeline guidance, which makes the first end-to-end CPT implementation less
  risky

Practical conclusion:

- prefer `NeMo` / `Megatron Bridge` for baseline CPT implementation
- keep `TorchTitan` as a follow-on option if we want a more PyTorch-native
  stack after the baseline is working

## High-Level Design

Stage 1: data generation

- generate Crafter rollout text using `Qwen/Qwen3.5-27B`
- serve generation with `vLLM` on `B200`
- serialize rollouts into pretraining text records
- upload records to Hugging Face datasets in rolling shards

Stage 2: dense CPT

- start from `Qwen/Qwen3.5-0.8B`
- run dense next-token training on Crafter rollout text
- export dense checkpoints at defined milestones
- first milestone is a tiny bootstrap run over the first `100k` tokens only

Stage 3: fixed RLVR

- point `nanohorizon` RLVR at a CPT checkpoint as the base model
- keep the RL recipe fixed across experiments
- run LoRA RLVR on top of the CPT checkpoint

Stage 4: evaluation

- compare fixed-RL baseline vs CPT-plus-fixed-RL
- preserve record bundles and exact training/eval configs

## Experiment Contract

We want one simple comparison before adding extra complexity.

Baseline A:

- base `Qwen/Qwen3.5-0.8B`
- no CPT
- fixed RLVR recipe

Baseline B:

- CPT checkpoint only
- no RLVR
- offline eval and/or held-out Crafter eval only

Main comparison:

- CPT checkpoint from the same base `Qwen/Qwen3.5-0.8B`
- same fixed RLVR recipe as Baseline A

Interpretation rule:

- if `CPT + RLVR` beats `base + RLVR`, CPT helped downstream RL
- if `CPT only` improves loss but `CPT + RLVR` does not improve reward, then
  the CPT data or recipe likely improved modeling without improving the policy
  search regime

## Data Notes

Target proposal:

- create up to `5B` tokens of Crafter rollouts for CPT
- use `Qwen/Qwen3.5-27B` generation on `B200` with `vLLM`
- upload generated data to a Hugging Face dataset

Initial shard plan:

- bootstrap by writing the first shard at `100k` tokens
- use that shard to validate format, tokenizer behavior, upload plumbing, and
  CPT job wiring before scaling out

Scaling note:

- `100k` token shards are good for bootstrap and debugging
- they are too small to be the steady-state storage format for a `5B` token
  dataset
- after the first bootstrap shard works, move to much larger logical shards
  while keeping deterministic manifests

Required dataset properties:

- every record must be plain pretraining text or a clearly documented chat/text
  projection
- tokenizer must match the CPT base model
- data generation must be reproducible from saved prompts, seeds, and model IDs
- manifests should track token counts per shard, source rollout count, and
  upload status

## Track Additions

Proposed new track family:

- `cpt_rlvr_qwen35_0p8b`

Likely public-facing pieces:

- new track doc in `docs/tracks/`
- one CPT bootstrap runner
- one dataset generation runner
- one CPT-to-RLVR runner
- one or more configs for:
  - bootstrap CPT on `100k` tokens
  - full CPT
  - fixed RLVR baseline
  - CPT plus fixed RLVR

Likely in-repo ownership split:

- data generation helpers: `scripts/` plus `src/nanohorizon/shared/`
- CPT launcher/config adapters: `scripts/` and possibly `dev/`
- RLVR handoff logic: `src/nanohorizon/baselines/rlvr.py`
- public track notes and record expectations: `docs/tracks/` and `README.md`

## Implementation Sequence

We should do this in strict stages.

### Phase 0: planning and scaffolding

1. Add this planning document.
2. Decide the checkpoint handoff format from CPT into RLVR.
3. Keep the existing RLVR public script stable where possible.

### Phase 1: baseline CPT code on tiny data

Goal:

- prove dense CPT wiring before large-scale data generation

Deliverables:

- baseline CPT launcher/config using the chosen pretraining framework
- local or remote bootstrap run over the first `100k` tokens only
- saved dense checkpoint and training logs

Success criteria:

- CPT job starts cleanly
- tokenization path is correct
- checkpoint save/load works

### Phase 2: generation pipeline

Goal:

- create the first real Crafter rollout shard

Deliverables:

- script to generate Crafter rollouts with `Qwen/Qwen3.5-27B` via `vLLM`
- text projection format for CPT
- upload path to Hugging Face datasets
- first `100k` token shard uploaded successfully

Success criteria:

- generation is resumable
- token counts are deterministic
- upload manifest is written

### Phase 3: RLVR handoff

Goal:

- run current RLVR from a dense CPT checkpoint

Deliverables:

- config support for arbitrary dense base model path/checkpoint in RLVR
- fixed RLVR baseline config for `Qwen/Qwen3.5-0.8B`
- CPT plus fixed-RLVR config using the CPT checkpoint as base model

Success criteria:

- `nanohorizon` RLVR can load a CPT checkpoint as its base model
- LoRA updates still reload cleanly into the inference runtime

### Phase 4: full pipeline

Goal:

- run end-to-end `generate -> CPT -> RLVR`

Deliverables:

- automated script or documented sequence for full flow
- baseline and CPT-plus-RLVR records
- run notes on cost, token counts, and wall time

## First Coding Milestones

The first milestones should be executed in this order:

1. write baseline CPT code that trains only on the first `100k` tokens
2. write the data generation script
3. generate the first `100k` token dataset shard
4. run the first CPT checkpoint
5. wire RLVR to start from that CPT checkpoint
6. run the first end-to-end `CPT -> RLVR` baseline

## Risks

Main engineering risks:

- framework-model mismatch for `Qwen/Qwen3.5-0.8B`
- tokenization or chat-to-text projection mismatch between generated rollouts and
  CPT input
- RLVR loader assumptions about Hugging Face model layout and local checkpoint
  paths
- storage and upload inefficiency if shard sizing stays too small

Main experimental risks:

- Crafter-only CPT may over-specialize and hurt general action quality
- generated text may be too narrow or too repetitive to produce strong RL gains
- CPT loss improvements may not transfer to downstream reward

## Immediate Next Step

Implement the smallest real artifact:

- baseline dense CPT bootstrap code on a `100k` token shard

After that works, build the data generation and upload path, then run the first
end-to-end `CPT -> RLVR` experiment.
