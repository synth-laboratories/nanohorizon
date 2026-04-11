# Craftax Task

Candidate label: `Daytona E2E Run 3`

Objective: make the smallest honest improvement to the Craftax prompt-opt path by
keeping a compact three-item todo scratchpad contract explicit from seed prompt
to reflection feedback.

Primary strategy: Todo Tool.

Compatibility surfaces to preserve:

- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

Implementation intent:

- keep the todo contract centralized in `src/nanohorizon/baselines/prompt_opt.py`
- keep the prompt-shaping change narrow and reviewable
- avoid SFT or RL changes
- document the candidate in a matching config, record bundle, and structural test
