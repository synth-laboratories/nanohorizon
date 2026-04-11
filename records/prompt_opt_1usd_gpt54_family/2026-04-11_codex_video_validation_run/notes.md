Prompt-opt candidate for the Craftax "Video Validation Run" scratchpad variant.

Hypothesis:
- A slightly more explicit private todo contract should help the model keep a short, auditable plan during rollout and avoid stale movement loops.
- The prompt keeps the same model, optimizer budget, seed split, and rollout shape as the prompt-opt baseline while sharpening the scratchpad language for quick visual inspection during a video validation run.

What changed:
- Added a prompt-opt candidate config that asks the model to maintain a private three-item todo list covering blocker/danger, next target, and fallback progress action.
- Tightened the scratchpad wording so completed items are refreshed every turn, stale targets are replaced after no-progress loops, and the final batch stays aligned with the first todo item.
- Packaged a matching candidate record bundle for reproducible follow-up runs.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` defines Craftax as a multi-step planning task under strict runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` already centralizes the todo-tool contract, so a narrower seed-prompt variant is the lowest-risk place to test this wording.
- `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml` and `configs/craftax_prompt_opt_qwen35_4b_codex_durable_intent_fix.yaml` show the existing prompt-opt candidate shape and record-bundle convention.

Validation performed in this task:
- Structural source/config/record validation only. No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- The sharper fallback wording could overconstrain some tactical action batches.
- The extra video-validation phrasing might be neutral rather than helpful if it does not improve the model's actual behavior.
- Because no live rollout run was executed, this remains a prompt-shaping candidate rather than a measured improvement.

