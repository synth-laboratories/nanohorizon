# Craftax Task

The first NanoHorizon task is Craftax.

The objective is to improve Qwen-based models on multi-step Craftax decision making under strict runtime and hardware budgets.

## Why Craftax

Craftax is a compact long-horizon environment:

- it exposes clear progress and survival dynamics
- it is rich enough to reward planning
- it is small enough to support rapid iteration

## Evaluation Shape

The intended public evaluation loop is:

1. start a Craftax runtime that exposes a stable rollout HTTP contract
2. query an OpenAI-compatible policy endpoint during each step
3. save trainer-ready rollout JSONL
4. score held-out episodes with a pinned eval harness

The in-repo Craftax runtime lives under [src/nanohorizon/craftax_core](/Users/joshpurtell/Documents/GitHub/nanohorizon/src/nanohorizon/craftax_core).

## Official Model

- Base model target: `Qwen/Qwen3.5-4B` unless a track doc states otherwise.
- Reference offline baseline: `Qwen/Qwen3.5-4B`

## Starter Assets

The repo now includes two benchmark starter assets:

- FT starter dataset:
  - [craftax_ft_starter.jsonl](../data/craftax/craftax_ft_starter.jsonl)
  - intended as the starting fixed-data bundle for the offline baseline
- FT seed prompts for teacher generation:
  - [craftax_ft_seed_prompts.jsonl](../data/craftax/craftax_ft_seed_prompts.jsonl)
  - intended for the reference offline script to expand into fresh SFT rows with `Qwen/Qwen3.5-9B` during the budget window
- RLVR starter seeds:
  - [craftax_rlvr_starter_seeds.json](../data/craftax/craftax_rlvr_starter_seeds.json)
  - intended as the starting rollout seed set for the RLVR baseline

## Track-Specific Policy

RLVR track:
- live environment interaction is allowed during the budget window

Offline track:
- no environment interaction after the budget timer starts
- training must consume fixed precomputed data only
- exception: the reference FBC path may generate teacher rollouts during the same budget window, filter high-reward traces into SFT rows, and then train `Qwen/Qwen3.5-4B`

Prompt optimization track:
- the final deployed policy remains `Qwen/Qwen3.5-4B`
- prompt search and prompt refinement may use up to $1 total spend across `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`
- the budget applies to prompt optimization only, not final Craftax evaluation runs

Separate classic track:
- not part of this Craftax task contract
- uses Craftax-Classic via the upstream JAX `craftax` package instead
- has its own task doc at [task-craftax-classic.md](task-craftax-classic.md)

## Open Questions To Lock Before OSS Launch

- exact held-out seed set
- exact final score definition
- whether any teacher besides `Qwen/Qwen3.5-9B` should be allowed for the offline reference path
- whether precomputed reward labels are part of the allowed offline bundle

## Current Candidate Note

- `Test Candidate` uses the smallest harness-side change that keeps the shared rollout contract stable: a compact working-memory buffer that feeds prior turn plan/state summaries back into the next Craftax prompt.
- The buffer is intentionally narrow and does not change the tool contract, action schema, or rollout HTTP surface.
