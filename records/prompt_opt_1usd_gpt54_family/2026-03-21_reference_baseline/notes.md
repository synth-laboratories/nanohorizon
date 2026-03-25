Prompt-opt reference baseline using honest Craftax outcome reward accounting.

Key points:
- held-out evaluation used 20 rollouts
- rollout horizon was 8 Craftax steps
- action batching was forced to 1 action per tool call to induce many model calls per rollout
- the run completed successfully but regressed from the seed prompt on actual reward

Observed result:
- base_eval mean_outcome_reward: 0.6
- best_eval mean_outcome_reward: 0.35
- score_delta: -0.25

Known caveat:
- rollout detail objects reported `llm_call_count: 0` despite many successful `/v1/chat/completions` calls in the live logs. This is a Craftax/container accounting issue, not an indication that zero model calls occurred.
