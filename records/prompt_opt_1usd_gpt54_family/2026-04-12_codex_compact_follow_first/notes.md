Prompt-opt candidate for the Craftax compact follow-first-item strategy.

Hypothesis:
- The todo scratchpad works best when it stays compact and action-directed.
- Removing the extra "end next to a useful target for the next turn" clause should reduce overconstraint while keeping the same resource-first loop-break behavior.

What changed:
- Tightened the shared prompt-opt reflection directives in `src/nanohorizon/baselines/prompt_opt.py` so GEPA preserves a compact, action-directed scratchpad contract.
- Added a candidate config that keeps the same model, optimizer budget, seed split, and rollout shape as the prompt-opt reference baseline while removing the extra end-position clause from the action guidance.
- Packaged a matching record bundle for reproducible follow-up runs.

Validation performed in this task:
- Structural source/config/record validation only.
- A local repeated-seed proxy benchmark was also run against the repo's direct rollout path; see `experiments/craftax_prompt_opt_compact_follow_first/`.
- Proxy benchmark result: baseline mean_outcome_reward `0.0625`, candidate mean_outcome_reward `1.0`, delta `+0.9375` on 8 repeated eval-seed rollouts per prompt.

Residual risks:
- The compact prompt may be too sparse if the model needs the extra "end next to a useful target" cue to keep short batches useful.
- No live Craftax or Modal reward run was executed in this task, so final leaderboard impact remains unmeasured.
