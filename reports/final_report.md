# Craftax Native Reward Fallback Candidate

## Context & objective

The task was to make one small NanoHorizon change that could plausibly improve Craftax scoring or logging without broad harness churn. I chose a compatibility/scoring edge case in the shared reward extraction path: some rollout payloads only carry `reward_info.details.native_env_reward_total`, while the current helpers only score achievement-based fields or `reward_info.outcome_reward`.

Success for this run meant:

- keep the Craftax HTTP/render surfaces stable
- implement the smallest honest code change
- verify it with a repeated-rollout comparison through the existing eval path
- leave behind durable artifacts and a reviewable PR

## Experiments cited

1. `experiments/craftax_reward_fallback/experiment_log.txt`
   - Question: what is the exact candidate, and how was it verified?
   - Outcome: supporting.
   - Evidence: recorded the change, the synthetic repeated-rollout comparison, and the verification commands.

2. `experiments/craftax_reward_fallback/results/repeated_rollout_comparison.json`
   - Question: does the new fallback change the eval summary on a repeated rollout set with missing achievement fields?
   - Outcome: supporting.
   - Evidence: baseline mean outcome reward `0.0`, candidate mean outcome reward `2.0`, delta `+2.0`.

3. `src/nanohorizon/shared/craftax_data.py`
   - Question: can the shared evaluator recover a meaningful score from legacy or partial rollout payloads?
   - Outcome: supporting.
   - Evidence: `rollout_outcome_reward()` now falls back to `details.native_env_reward_total`.

4. `src/nanohorizon/baselines/offline_sft.py`, `src/nanohorizon/baselines/prompt_opt.py`, `src/nanohorizon/baselines/rlvr.py`
   - Question: can the duplicated baseline-side scoring helpers stay consistent with the shared evaluator?
   - Outcome: supporting.
   - Evidence: the same fallback was propagated into each mirrored helper.

5. `tests/test_craftax_reward_fallback.py`
   - Question: does the existing eval summary path report the fallback correctly, and is the propagation visible in the repo source?
   - Outcome: supporting.
   - Evidence: the test checks `evaluate_model()` on three repeated synthetic rollouts and verifies the mirrored helper source contains the fallback string.

## Insights

1. The candidate is genuinely narrow: it changes only reward extraction behavior for partial/legacy rollout payloads and leaves the Craftax rollout HTTP contract unchanged.
2. The fallback is useful because the existing eval summary path can now score repeated rollouts even when achievement-based fields are missing, instead of silently collapsing them to `0.0`.
3. The measured delta on the synthetic repeated-rollout comparison is large and unambiguous for this edge case: `0.0 -> 2.0` mean outcome reward.
4. The remaining risk is scope, not correctness: this candidate does not prove live Craftax model improvement, only that the scoring/logging path is more robust on partially populated rollout payloads.

## Research artifacts produced

- Environment notes: `experiments/craftax_reward_fallback/experiment_log.txt`
- Comparison output: `experiments/craftax_reward_fallback/results/repeated_rollout_comparison.json`
- Code changes:
  - `src/nanohorizon/shared/craftax_data.py`
  - `src/nanohorizon/baselines/offline_sft.py`
  - `src/nanohorizon/baselines/prompt_opt.py`
  - `src/nanohorizon/baselines/rlvr.py`
- Regression coverage: `tests/test_craftax_reward_fallback.py`

## Quality & validation

- Passed: `PYTHONPATH=src python3 -m pytest tests/test_craftax_reward_fallback.py tests/test_craftax_interface.py`
- Passed: repeated synthetic comparison through `nanohorizon.shared.eval_model.evaluate_model`
- Baseline result: mean outcome reward `0.0`
- Candidate result: mean outcome reward `2.0`
- Not validated:
  - live Craftax model rollouts
  - full repo test suite
  - `uv run pytest ...`, because this workspace points at an unavailable local `synth-ai` path dependency
  - `tests/test_craftax_core_contract.py`, because it currently fails to import `create_app` from `src/nanohorizon/craftax_core/http_shim.py` in this workspace

## Reproduction & handoff

- Candidate verification:
  - `PYTHONPATH=src python3 -m pytest tests/test_craftax_reward_fallback.py tests/test_craftax_interface.py`
- Comparison receipt:
  - `experiments/craftax_reward_fallback/artifacts/compare_eval_command.txt`
- The comparison used `nanohorizon.shared.eval_model.evaluate_model` with three repeated synthetic rollouts and only `reward_info.details.native_env_reward_total` populated.
- Suggested follow-up: if a live Craftax run produces more partial rollout payloads, this fallback should preserve score accounting instead of discarding those episodes.
