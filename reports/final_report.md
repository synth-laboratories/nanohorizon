# Craftax Recent-Trajectory Prompt Candidate

## Context & Objective

Implement the smallest honest Craftax improvement in the NanoHorizon repo that could plausibly improve policy quality, while keeping the shared harness surfaces stable unless a compatibility repair was genuinely required.

The chosen lever was prompt shaping in the rollout loop: expose a short recent-trajectory summary to the model on each turn so it can break repetition loops using actual action history instead of only the current observation text.

## Experiments Cited

1. `tests/test_craftax_core_contract.py` and `tests/test_craftax_interface.py`
   - Question: did the rollout prompt change preserve the contract and expose the new recent-trajectory context?
   - Outcome: supporting.
   - Evidence: the focused verifier run passed after restoring the missing FastAPI shim and adding the new prompt-history test.

2. `records/prompt_opt_1usd_gpt54_family/2026-04-12_recent_trajectory_proxy/`
   - Question: does the prompt-history addition improve a rollout proxy when the policy can see the recent trajectory block?
   - Outcome: supporting for the prompt-shape hypothesis, with the caveat that this is a fake-runner proxy rather than a live Craftax score.
   - Evidence: `comparison.json`, `baseline_summary.json`, `candidate_summary.json`, `command.txt`, `notes.md`, `metrics.json`, `metadata.json`.

## Insights

1. Recent action history is useful when it is actually provided to the model. In the three-seed proxy slice, the baseline mean outcome reward was `1.0` and the candidate mean outcome reward was `2.0`, a delta of `+1.0`.
2. The candidate achieved `collect_sapling` in addition to `collect_wood` in all three proxy rollouts, while the baseline only reached `collect_wood`.
3. The compatibility repair in `src/nanohorizon/craftax_core/http_shim.py` was necessary because the workspace snapshot was missing `create_app`, and the tests depended on that surface.
4. The live Craftax runtime could not be exercised in this checkout because the `craftax` dependency was unavailable, so the benchmark result is a deterministic prompt-shape proxy, not a leaderboard score.

## Research Artifacts Produced

### Environments

- `src/nanohorizon/craftax_core/rollout.py`
  - prompt shaping now includes recent trajectory context
- `src/nanohorizon/craftax_core/http_shim.py`
  - restored FastAPI app factory and preserved the rollout route seam

### Data

- `records/prompt_opt_1usd_gpt54_family/2026-04-12_recent_trajectory_proxy/`
  - baseline and candidate summaries
  - per-seed rollouts
  - command note and proxy-eval notes

### Models / Checkpoints

- No weights or checkpoints were produced in this run.

## Quality & Validation

- Executed: `PYTHONPATH=src python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_interface.py`
- Result: `10 passed`
- Executed: direct rollout proxy on seeds `10000`, `10001`, `10002` with the repo rollout path and a fake Craftax runner
- Result:
  - baseline mean outcome reward: `1.0`
  - candidate mean outcome reward: `2.0`
  - delta: `+1.0`
- Not validated:
  - live Craftax runtime
  - real Qwen leaderboard score
  - Modal / remote container execution

## Reproduction & Handoff

- Code entrypoint: `src/nanohorizon/craftax_core/rollout.py`
- Harness repair: `src/nanohorizon/craftax_core/http_shim.py`
- Proxy record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-12_recent_trajectory_proxy/`
- Reproduction command for the proxy slice: inspect `command.txt` and `comparison.json` in the record bundle
- Open risk: the improvement is only validated as a prompt-shape proxy until the real Craftax package and live policy endpoint are available in this workspace
