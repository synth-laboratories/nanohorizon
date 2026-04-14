# Craftax Submission Smoke Eval

## Context

Task: add a `PUBLICATION_SMOKE_NOTE` prompt note in `submission/agent.py` and keep the Craftax submission entrypoint runnable.

## Verification

- Direct real-runtime eval attempt:
  - `python submission/agent.py eval --data-dir data --checkpoint-dir /tmp/nanohorizon_baseline_ckpt --out-dir /tmp/nanohorizon_baseline_eval`
  - Blocked by missing local `vllm` binary at `/opt/nanohorizon-offline-venvs/teacher/bin/vllm`
- Harness smoke comparison:
  - Ran the existing `submission.agent.eval` entrypoint for both baseline and candidate.
  - Used the train-seed slice `[10007, 10008, 10011]`.
  - Patched in a temporary `/tmp/craftax_fakes` `jax` + `craftax` stub tree so the entrypoint could complete in this workspace without repo changes.

## Results

- Baseline:
  - `primary_score=2.0`
  - `mean_outcome_reward=2.0`
  - `num_eval_rollouts=3`
  - `num_rollout_errors=0`
  - `mean_llm_calls_per_rollout=1.0`
- Candidate:
  - `primary_score=2.0`
  - `mean_outcome_reward=2.0`
  - `num_eval_rollouts=3`
  - `num_rollout_errors=0`
  - `mean_llm_calls_per_rollout=1.0`

## Interpretation

- The prompt note is present in the candidate system prompt.
- Baseline and candidate tied on the smoke comparison.
- The fake harness still exercised the real `submission.agent.eval` entrypoint and produced structured rollout artifacts for all three seeds.
- The real vLLM-backed Craftax evaluation remains blocked in this workspace because the expected `vllm` binary is missing.
