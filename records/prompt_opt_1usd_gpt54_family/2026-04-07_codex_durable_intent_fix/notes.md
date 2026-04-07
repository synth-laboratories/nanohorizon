Prompt-opt candidate for the Craftax "Todo Tool" strategy.

Hypothesis:
- The current Qwen prompt asks for valid actions and strict tool use but gives no compact structure for tracking subgoals across its private reasoning.
- Adding a tiny private three-item todo list should improve long-horizon consistency without changing the shared Craftax runtime or eval harness.

What changed:
- Added a new prompt-opt config with a seed prompt that asks the model to maintain a private todo list covering immediate resource/survival need, next position, and next unlock.
- Preserved the existing tool contract, model target, budget, seed split, and rollout shape from the prompt-opt reference baseline.

Evidence gathered before choosing this change:
- `docs/task-craftax.md` defines Craftax as a multi-step planning task under strict runtime budgets.
- `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml` currently uses a minimal seed prompt with no explicit durable-intent scaffold.
- `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/notes.md` shows the prior prompt-opt baseline regressed on held-out reward, so a small manual seed-prompt correction is lower risk than broad harness changes.

Validation performed in this task:
- Structural record validation only. No live Craftax/Modal run was executed, so reward impact remains unmeasured.

Residual risks:
- The hidden todo instruction could increase reasoning verbosity or distract from local tactical cues.
- GEPA may rewrite away the todo structure unless the seed prompt produces a measurable bootstrap lift.
