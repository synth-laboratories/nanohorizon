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

The in-repo Craftax runtime lives under `src/nanohorizon/craftax_core`.

## Official Model

- Base model target: `Qwen/Qwen3.5-4B` unless a track doc states otherwise.
- Reference offline baseline: `Qwen/Qwen3.5-4B`

## Track-Specific Policy

Prompt optimization track:
- the final deployed policy remains `Qwen/Qwen3.5-4B`
- prompt search and prompt refinement may use up to $1 total spend across `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`
- the budget applies to prompt optimization only, not final Craftax evaluation runs

