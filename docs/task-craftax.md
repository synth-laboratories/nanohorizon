# Craftax Task

Candidate label: `Daytona E2E Run 3`

Objective: implement the smallest honest improvement to the NanoHorizon Craftax harness that helps the agent keep a compact todo scratchpad while acting.

Primary strategy: Todo Tool.

Implementation intent:
- provide a compact scratchpad for subgoals
- keep the prompt and tool contract small
- avoid broad harness changes unless required

Compatibility surfaces to preserve:
- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`
