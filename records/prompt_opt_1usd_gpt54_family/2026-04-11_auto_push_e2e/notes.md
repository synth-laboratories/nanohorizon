Prompt-opt candidate for the Craftax "Todo Tool" strategy.

Hypothesis:
- The shared todo directive can be the single source of truth for both seed prompting and reflection.
- A small end-to-end handoff guard can keep that scratchpad current without changing the shared Craftax runtime or eval harness.

What changed:
- Centralized the seed prompt around `todo_scratchpad_directive()` so the private three-item scratchpad contract is reused instead of duplicated.
- Kept the candidate config aligned with the shared prompt builder and updated the structural test to compare the config against the builder output.
- Preserved the record bundle for the Auto Push E2E variant.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` frames Craftax as a multi-step planning task under strict runtime budgets.
- `src/nanohorizon/baselines/prompt_opt.py` now sources both seed and reflection prompt text from the same todo-contract helper.
- The new config stays within the existing prompt-opt budget and rollout envelope.

Validation performed in this task:
- Structural source/config/record validation only. `uv run --with pytest pytest -q tests/test_auto_push_e2e_candidate.py` passed with `4 passed`.
- No live Craftax, Modal, or GEPA reward run was executed, so reward impact remains unmeasured.

Residual risks:
- Because no live rollout run was executed, this remains a packaging and prompt-shaping candidate rather than a measured improvement.
- GitHub publication is blocked by repository configuration: `create_github_pr` rejected every tried repo slug as not present in the configured GitHub repos list.
