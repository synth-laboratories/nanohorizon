# Craftax Candidate Contract

This workspace is a minimal NanoHorizon Craftax candidate harness.

## Candidate

- Candidate label: `Video Validation Run`
- Goal: keep the harness changes small, reviewable, and evidence-based.

## What to preserve

Keep the shared harness surfaces stable unless a change is clearly justified:

- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

## Workflow

Use a compact todo/scratchpad tool in the agent workflow:

- keep the list short
- prefer explicit subgoals over narrative notes
- mark items done as evidence accumulates
- keep the scratchpad bounded so it stays readable during video validation runs

## Constraints

- No SFT.
- No RL.
- Prefer the smallest honest change that improves reliability or performance.
