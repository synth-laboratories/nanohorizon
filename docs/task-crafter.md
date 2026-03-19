# Crafter Task

The first NanoHorizon task is Crafter.

The objective is to improve `Qwen/Qwen3.5-0.8B` on multi-step Crafter decision making under strict runtime and hardware budgets.

## Why Crafter

Crafter is a compact long-horizon environment:

- it exposes clear progress and survival dynamics
- it is rich enough to reward planning
- it is small enough to support rapid iteration

## Evaluation Shape

The intended public evaluation loop is:

1. start a Crafter runtime that exposes a stable rollout HTTP contract
2. query an OpenAI-compatible policy endpoint during each step
3. save trainer-ready rollout JSONL
4. score held-out episodes with a pinned eval harness

The in-repo Crafter runtime lives at [runtime/crafter_rs](../runtime/crafter_rs).

## Official Model

- `Qwen/Qwen3.5-0.8B`

## Starter Assets

The repo now includes two benchmark starter assets:

- FT starter dataset:
  - [crafter_ft_starter.jsonl](../data/crafter/crafter_ft_starter.jsonl)
  - intended as the starting fixed-data bundle for the offline baseline
- FT seed prompts for teacher generation:
  - [crafter_ft_seed_prompts.jsonl](../data/crafter/crafter_ft_seed_prompts.jsonl)
  - intended for the reference offline script to expand into fresh SFT rows with `Qwen/Qwen3.5-27B` during the budget window
- RLVR starter seeds:
  - [crafter_rlvr_starter_seeds.json](../data/crafter/crafter_rlvr_starter_seeds.json)
  - intended as the starting rollout seed set for the RLVR baseline

## Track-Specific Policy

RLVR track:
- live environment interaction is allowed during the budget window

Offline track:
- no environment interaction after the budget timer starts
- training must consume fixed precomputed data only
- exception: the reference offline path may generate SFT data with `Qwen/Qwen3.5-27B` during the same budget window before training `Qwen/Qwen3.5-0.8B`

Prompt optimization track:
- the final deployed policy remains `Qwen/Qwen3.5-0.8B`
- prompt search and prompt refinement may use up to $1 total spend across `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`
- the budget applies to prompt optimization only, not final Crafter evaluation runs

## Open Questions To Lock Before OSS Launch

- exact held-out seed set
- exact final score definition
- whether any teacher besides `Qwen/Qwen3.5-27B` should be allowed for the offline reference path
- whether precomputed reward labels are part of the allowed offline bundle
