Prompt-opt candidate for the Craftax "Todo Tool" strategy with a stricter refresh gate and a compact working-memory buffer.

Hypothesis:
- The current todo candidate names subgoals, but GEPA reflection can still drift toward vaguer planning language.
- Preserving an explicit three-item private todo contract in the prompt-opt source, exposing it as a working-memory buffer, and asking the final action batch to follow that first todo item should reduce stale movement loops without touching the shared Craftax runtime or eval harness.

What changed:
- Centralized the todo-tool contract in `src/nanohorizon/baselines/prompt_opt.py` so reflection and feedback keep the same private three-item scratchpad requirements.
- Added a prompt-opt candidate config that keeps the same model, optimizer budget, seed split, and rollout shape as the prompt-opt baseline while tightening the action-batch guidance around the active todo item and making the buffer capacity explicit.
- Packaged a matching candidate record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` defines Craftax as a multi-step planning task under strict runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` already tried to preserve generic todo guidance during reflection, but the richer refresh-and-replace behavior was duplicated informally rather than encoded once as a preserved contract.
- `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml` still uses a minimal seed prompt with no explicit todo or loop-break structure.

Validation performed in this task:
- Structural source/config/record validation only. No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured. The local `uv run` verifier was also blocked by a stale `synth-ai` path in the project dependency metadata.

Residual risks:
- The stronger "follow the first todo item" wording could overconstrain action batches in situations where a different short tactical sequence is better.
- Preserving the todo contract during reflection may still not survive if reward signal is too noisy on the small seed split.
- Because no live rollout run was executed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.
