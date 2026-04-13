# Craftax Candidate Report

## Context & Objective

This run repaired the Craftax harness surface and made the rollout prompt more informative without changing the action contract or adding SFT/RL. The target was the smallest reviewable change that improves reproducibility and scorer-observable rollout behavior.

## Experiments Cited

1. Baseline Craftax contract test before the fix
   - Command: `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with pyyaml --with pillow --with uvicorn --with numpy -- python -m pytest tests/test_craftax_core_contract.py -q`
   - Outcome: failed during collection because `nanohorizon.craftax_core.http_shim` no longer exported `create_app`.
   - Evidence: the ImportError from `tests/test_craftax_core_contract.py`.

2. Candidate Craftax test suite after the fix
   - Command: `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with pyyaml --with pillow --with uvicorn --with numpy -- python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_interface.py tests/test_craftax_core_runtime_guarantees.py -q`
   - Outcome: passed.
   - Evidence: `18 passed in 1.64s`.

3. Repeated-seed proxy benchmark
   - Artifact: `experiments/craftax_candidate/results/prompt_context_proxy_eval.json`
   - Script: `experiments/craftax_candidate/scripts/craftax_prompt_context_proxy_eval.py`
   - Outcome: supporting for the prompt change.
   - Evidence: repeated seeds produced identical prompt hashes, candidate prompts included structured context and reward history, and the candidate proxy policy achieved higher mean native env reward total than baseline (`22.0` vs `19.0`).

4. Fresh verification pass
   - Artifact: `experiments/craftax_candidate/results/prompt_context_proxy_eval_fresh.json`
   - Script: `experiments/craftax_candidate/scripts/craftax_prompt_context_proxy_eval.py`
   - Outcome: supporting and repeatable.
   - Evidence: the fresh run matched the earlier repeated-seed summary exactly, including deterministic hashes and the same `22.0` vs `19.0` native env reward totals.

## Insights

1. Restoring `create_app`, `app`, and `main` is a real reproducibility fix, not just a style cleanup. The prior workspace state could not even import the Craftax HTTP contract.
2. Using `render_prompt_turn(...)` inside `run_rollout(...)` makes the model-facing prompt carry stable structured state and bounded reward history instead of only a raw text observation.
3. The new prompt assembly is deterministic across repeated seeds in the proxy benchmark, which matters for reproducible rollout debugging.
4. The proxy benchmark is only a proxy. It supports the prompt change, but it is not a live-model score and should not be treated as a final leaderboard result.

## Research Artifacts Produced

### Environments

- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/rollout.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/texture_cache.py`

### Data

- Proxy benchmark summary: `experiments/craftax_candidate/results/prompt_context_proxy_eval.json`
- Experiment log: `experiments/craftax_candidate/experiment_log.txt`

### Models / Checkpoints

- None produced in this run.

## Quality & Validation

- Baseline failure captured before the fix: missing `create_app` import.
- Candidate validation passed with the Craftax-focused test set.
- `uv run` in normal project mode was blocked by a stale cloud dependency path in `pyproject.toml`; the verification commands therefore used `uv run --no-project` with explicit packages and `PYTHONPATH=src`.
- Not validated: any live Craftax container rollout, any external model endpoint, or any real leaderboard score.

## Reproduction & Handoff

- Re-run the Craftax tests:
  - `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with pyyaml --with pillow --with uvicorn --with numpy -- python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_interface.py tests/test_craftax_core_runtime_guarantees.py -q`
- Re-run the proxy benchmark:
  - `PYTHONPATH=src uv run --no-project --with fastapi --with httpx --with pyyaml --with pillow --with uvicorn --with numpy -- python experiments/craftax_candidate/scripts/craftax_prompt_context_proxy_eval.py --output experiments/craftax_candidate/results/prompt_context_proxy_eval.json`
  - Fresh replay: `PYTHONPATH=src uv run --no-project --with fastapi --with httpx --with pyyaml --with pillow --with uvicorn --with numpy -- python experiments/craftax_candidate/scripts/craftax_prompt_context_proxy_eval.py --output experiments/craftax_candidate/results/prompt_context_proxy_eval_fresh.json`
- Pushed commit:
  - `e2cbb174f3e2843b52ff0c08f2d0f5a4599459cf`
- PR:
  - `https://github.com/synth-laboratories/nanohorizon/pull/42`
- Open risk:
  - the reported reward lift is from a deterministic proxy policy on a fake Craftax loop, not from the live evaluation service.
