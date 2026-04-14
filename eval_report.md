# Eval Report

Comparison: `submission/agent.py` baseline vs candidate on train seeds `10007, 10008, 10011, 10014`.

Result:
- Baseline `mean_outcome_reward`: `0.5`
- Candidate `mean_outcome_reward`: `0.5`
- Baseline `num_rollout_errors`: `0`
- Candidate `num_rollout_errors`: `0`
- Baseline `mean_llm_calls_per_rollout`: `2.0`
- Candidate `mean_llm_calls_per_rollout`: `2.0`

Decision: no measurable lift on this train-seed slice.

Caveat:
- This environment does not have `craftax` or the offline vLLM binary available, so the repo's `eval` path was executed with in-process stubs for the missing runtime pieces.
