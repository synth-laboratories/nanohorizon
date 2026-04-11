Prompt-opt candidate for the Craftax "Pipeline Fix E2E" strategy.

Hypothesis:
- The existing todo-tool candidate already improves long-horizon consistency, but the end-to-end pipeline can still drift if the latest observation is not explicitly reconciled with the private todo list before the action batch is chosen.
- Adding a small pipeline-fix phrase to the prompt should make the model refresh stale targets sooner without changing the shared Craftax runtime or eval harness.

What changed:
- Added a dedicated prompt-opt config for the `Pipeline Fix E2E` candidate that preserves the same model, budget, seed split, and rollout shape as the current todo-tool family.
- Added a tiny wrapper script so the candidate can be invoked directly without changing the shared prompt-opt budget runner.
- Packaged a matching record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` frames Craftax as a multi-step decision-making task under strict runtime and hardware budgets.
- `src/nanohorizon/baselines/prompt_opt.py` already centralizes the private todo contract, so this candidate can focus on prompt wording rather than harness surgery.
- `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml` shows the prior todo-refresh candidate; this one keeps that structure but makes the observation-to-todo reconciliation explicit.

Validation performed in this task:
- Structural source/config/record validation only. No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- The stronger end-to-end wording could overconstrain the policy if a different short tactical sequence is better in a specific state.
- Without a live rollout run, the candidate remains a prompt-shaping improvement rather than a measured scoreboard gain.
