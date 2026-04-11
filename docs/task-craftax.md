# Craftax Task

This workspace is focused on the Craftax leaderboard candidate for NanoHorizon.

## Objective

Improve the Craftax policy with the smallest honest prompt-side change that
helps long-horizon planning.

## Candidate

- label: `Full Auto E2E`
- strategy: `Todo Tool`
- model family: `Qwen/Qwen3.5-4B`
- optimization budget: `1 USD` across `gpt-5.4`, `gpt-5.4-mini`, and
  `gpt-5.4-nano`

## Prompt contract

The candidate keeps a private three-item todo list before each action choice:

1. the most urgent blocker or danger
2. the next tile, object, or resource to reach
3. a fallback action that breaks a loop if progress stalls

The prompt also:

- refreshes completed todo items every turn
- replaces stale targets after repeated no-progress loops
- keeps the todo list private
- returns exactly one `craftax_interact` tool call with a short valid action
  batch

## Stability rule

The shared harness surfaces should stay stable unless a direct change is
required for the candidate:

- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `src/nanohorizon/craftax_core/runner.py`
- `scripts/run_craftax_model_eval.sh`

## Verification stance

This run is a prompt-shaping and packaging pass. It does not claim a live
Craftax score improvement unless an actual rollout or verifier run is recorded
in the workspace artifacts.
