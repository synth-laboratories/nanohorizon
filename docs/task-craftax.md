# Craftax Task

Candidate label: `Server Push E2E`

Objective: make the smallest honest improvement to the Craftax approach by
centralizing a compact Todo Tool / scratchpad contract that the agent can keep
fresh while it works.

Scope and constraints:

- Keep the change narrow and reviewable.
- Do not add SFT or RL.
- Preserve the Craftax harness surfaces if they exist in the full repo:
  - `docs/task-craftax.md`
  - `src/nanohorizon/craftax_core/http_shim.py`
  - `src/nanohorizon/craftax_core/runner.py`
  - `src/nanohorizon/craftax_core/metadata.py`
  - `scripts/run_craftax_model_eval.sh`
- Prefer a single source of truth for the todo contract instead of repeating it
  in multiple prompts or notes.

Implementation intent:

- Keep a private three-item scratchpad contract explicit from seed prompt to
  reflection feedback.
- Treat completed items as server-pushed state: remove them immediately and
  replace them with the next most useful action.
- If progress stalls, refresh the scratchpad instead of letting stale todos
  accumulate.
- Package the candidate with a reproducible config, a small structural test,
  and durable notes so later runs can inspect the decision.
