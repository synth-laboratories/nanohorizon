Prompt-opt candidate for the Craftax Todo Tool strategy labeled Full Auto E2E.

Hypothesis:
- A strict private three-item todo list should keep the policy focused on the
  current blocker, the next reachable target, and a loop-break fallback.
- Refreshing completed todo items every turn and replacing stale targets after
  repeated no-progress loops should reduce low-value movement churn without
  changing the shared harness contract.

What changed:
- Added a shared prompt-opt helper in `src/nanohorizon/baselines/prompt_opt.py`
  that centralizes the Full Auto E2E todo contract.
- Added a candidate config that keeps the same base model, optimizer budget,
  rollout shape, and seed split used by the prompt-opt family while making the
  todo contract explicit.
- Added a matching record bundle so future runs can inspect the candidate shape
  without reconstructing it from chat history.

Validation performed:
- Structural smoke only.
- No live Craftax rollout, Modal job, or GEPA optimization run was executed.

Residual risks:
- The stronger loop-break language could overconstrain action batches when a
  different short tactical sequence is better.
- This run does not provide measured reward impact.
