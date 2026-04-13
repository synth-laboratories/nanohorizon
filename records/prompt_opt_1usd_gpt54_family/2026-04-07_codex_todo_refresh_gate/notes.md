Prompt-opt candidate for the Craftax "Todo Tool" strategy with a stricter refresh gate.

Hypothesis:
- The current todo candidate names subgoals, but GEPA reflection can still drift toward vaguer planning language.
- Preserving an explicit three-item private todo contract in the prompt-opt source and asking the final action batch to follow that first todo item should reduce stale movement loops without touching the shared Craftax runtime or eval harness.

What changed:
- Centralized the todo-tool contract in `src/nanohorizon/baselines/prompt_opt.py` so reflection and feedback keep the same private three-item scratchpad requirements.
- Added a prompt-opt candidate config that keeps the same model, optimizer budget, seed split, and rollout shape as the prompt-opt baseline while tightening the action-batch guidance around the active todo item.
- Packaged a matching candidate record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` defines Craftax as a multi-step planning task under strict runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` already tried to preserve generic todo guidance during reflection, but the richer refresh-and-replace behavior was duplicated informally rather than encoded once as a preserved contract.
- `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml` still uses a minimal seed prompt with no explicit todo or loop-break structure.

Validation performed in this task:
- Structural source/config/record validation passed.
- `uv run pytest tests/test_codex_todo_refresh_gate_candidate.py` passed.
- A live baseline-vs-candidate prompt-opt comparison was attempted with the repo's existing Craftax eval path, but the public rollout/reflection endpoint returned `404 modal-http: invalid function call` during GEPA reflection, so the reward comparison could not complete in this workspace.

Residual risks:
- The stronger exact-4-action wording could overconstrain action batches in situations where a different short tactical sequence is better.
- The prompt-opt eval path remains blocked here until the rollout/reflection endpoint is reachable again or Modal bootstrap/auth is available.
- Because no live reward comparison completed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.
