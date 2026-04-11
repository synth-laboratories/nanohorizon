Prompt-opt candidate for the Craftax "Todo Tool" strategy.

Hypothesis:
- The existing todo refresh gate already centralizes the private three-item scratchpad.
- A small end-to-end handoff guard can keep that scratchpad current without changing the shared Craftax runtime or eval harness.

What changed:
- Added a new candidate config that preserves the same model, optimizer budget,
  seed split, and rollout shape while tightening the end-to-end handoff wording.
- Added a structural test and record bundle for the Auto Push E2E variant.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` frames Craftax as a multi-step planning task under strict runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` centralizes the todo-tool contract in `TODO_SCRATCHPAD_REQUIREMENTS`.
- The new config stays within the existing prompt-opt budget and rollout envelope.

Validation performed in this task:
- Structural source/config/record validation only. No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- The extra end-to-end handoff wording could overconstrain otherwise good short tactical action batches.
- Because no live rollout run was executed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.

