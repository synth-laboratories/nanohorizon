# Craftax Prompt History Candidate

## Context & objective

The objective was to make the smallest honest NanoHorizon Craftax change that could plausibly improve leaderboard behavior without changing the protected harness surface more than necessary.

Success meant:

1. the rollout prompt should carry more useful decision context than the baseline
2. the change should be validated against a controlled baseline-vs-candidate slice
3. the work should be published as a reviewable commit and GitHub PR
4. verifier feedback should be consulted before calling the candidate ready

## Experiments cited

1. [`src/nanohorizon/craftax_core/rollout.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/src/nanohorizon/craftax_core/rollout.py)
   - Question: does adding bounded recent action/reward history to the prompt change Craftax behavior in a useful direction?
   - Outcome: supporting.
   - Evidence: the prompt builder now appends a compact recent-history section before the action request, and `run_rollout()` records reward history across turns.

2. [`src/nanohorizon/craftax_core/http_shim.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/src/nanohorizon/craftax_core/http_shim.py)
   - Question: can the local Craftax HTTP entrypoint be restored without changing the rollout contract?
   - Outcome: supporting.
   - Evidence: `create_app()` now exposes `/health`, `/task_info`, `/rollout`, and `/rollouts`, which matches the repo’s eval/tunnel expectations and unblocks local verification.

3. [`tests/test_craftax_core_contract.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/tests/test_craftax_core_contract.py), [`tests/test_craftax_core_runtime_guarantees.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/tests/test_craftax_core_runtime_guarantees.py), [`tests/test_craftax_interface.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/tests/test_craftax_interface.py), [`tests/test_craftax_core_rollout_state_view.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/tests/test_craftax_core_rollout_state_view.py)
   - Question: do the harness changes preserve the existing prompt and rollout expectations?
   - Outcome: supporting.
   - Evidence: 19 targeted tests passed in the isolated `uv` environment.

4. [`experiments/craftax_candidate_20260412/scripts/proxy_eval.py`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/experiments/craftax_candidate_20260412/scripts/proxy_eval.py) and [`experiments/craftax_candidate_20260412/results/proxy_eval_comparison.json`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/experiments/craftax_candidate_20260412/results/proxy_eval_comparison.json)
   - Question: does the candidate outperform a baseline slice when the model can actually use the added history?
   - Outcome: supporting on the proxy slice, inconclusive for live model quality.
   - Evidence: fixed-seed comparison on seeds `0..3` produced baseline mean `outcome_reward = 1.0` and candidate mean `outcome_reward = 2.0` with `delta = +1.0`.

## Insights

1. The missing signal in the original rollout path was not just state text, but prior turn context. The candidate now carries a bounded recent action/reward window into the next prompt, which is the smallest direct behavioral change that could plausibly reduce repeated loops.
2. Restoring the HTTP shim surface was necessary to keep the existing Craftax entrypoint and local verification path alive. Without it, the rollout code could not be exercised as intended.
3. On the proxy evaluation slice, the candidate clearly dominated the baseline on the chosen metric. That supports retaining the prompt-history change, while still leaving live-model uplift unproven.
4. The evidence is honest but limited: the local workspace lacks `craftax` and `craftaxlm`, so the comparison used the fake Craftax runner plus a deterministic mock policy endpoint rather than a live Qwen rollout.

## Research artifacts produced

### Environments

- Local isolated `uv` test environment created with:
  - `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml --with uvicorn ...`
- Proxy eval environment:
  - `experiments/craftax_candidate_20260412/scripts/proxy_eval.py`
  - fixed seeds, fake runner, deterministic mock policy

### Data

- Proxy comparison output:
  - [`experiments/craftax_candidate_20260412/results/proxy_eval_comparison.json`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/experiments/craftax_candidate_20260412/results/proxy_eval_comparison.json)
- Durable experiment log:
  - [`experiments/craftax_candidate_20260412/experiment_log.txt`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/experiments/craftax_candidate_20260412/experiment_log.txt)
- Handoff note:
  - [`findings.txt`](/synth/state/.out/smr/projects/2afc6cd5-d5f7-4b26-b81c-7acd5bfd62ed/runs/caeffd64-dbf3-42f5-a270-b82109f8bed1/workspace/findings.txt)

### Models / checkpoints

- No model checkpoint was trained or changed.
- This run only changed prompt shaping and the local shim surface.

## Quality & validation

- Targeted validation command:
  - `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml --with uvicorn python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_core_runtime_guarantees.py tests/test_craftax_interface.py tests/test_craftax_core_rollout_state_view.py -q`
- Result:
  - `19 passed`
- Proxy evaluation command:
  - `PYTHONPATH=src:. uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml --with uvicorn python experiments/craftax_candidate_20260412/scripts/proxy_eval.py --output-dir experiments/craftax_candidate_20260412/results --seed-start 0 --num-seeds 4`
- Result:
  - baseline mean `outcome_reward = 1.0`
  - candidate mean `outcome_reward = 2.0`
  - delta `+1.0`
- Explicitly not validated:
  - live Craftax rollouts against `craftax` / `craftaxlm`
  - a real model rollout on Qwen 3.5

## Reproduction & handoff

- Workspace commit from `workspace_push`:
  - `cd300d0b41cd8ab55a7f65e196bb43c9715edfed`
- GitHub push / review branch:
  - `pr/worker/run-caeffd64-dbf3-42f5-a270-b82109f8bed1`
  - GitHub commit `b431bb8246a9a292474a2591ee112f9385fbd016`
- PR:
  - [synth-laboratories/nanohorizon#36](https://github.com/synth-laboratories/nanohorizon/pull/36)
- Main risk:
  - the gain is measured on a proxy slice because the live Craftax dependency stack is absent locally
- Current blocker:
  - external verifier feedback has been requested but is not yet available in the run artifacts, so the candidate is published and reproducibly measured but not fully externally verified yet
