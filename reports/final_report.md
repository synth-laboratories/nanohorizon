# Shared History E2E

## Context & Objective

The run goal was to implement a compact NanoHorizon candidate around shared history and a todo scratchpad, while preserving the shared Craftax harness surfaces unless a verified necessity required a change.

The git checkout was effectively empty at start, so the honest implementation target became a self-contained NanoHorizon todo board that can be reused by future harness wiring.

## Change

- Added `src/nanohorizon/craftax_core/todo_tool.py` with a small `TodoBoard` and `TodoItem` model.
- Added `tests/test_todo_tool.py` coverage for round-trip serialization, ordered todo creation, missing-item handling, and invalid-status coercion.
- Added `pyproject.toml` plus the generated `uv.lock` so the `src/` layout is importable under `uv run` without manually exporting `PYTHONPATH`.
- Aligned the board serialization with the richer `/app` optimizer-state shape by emitting both `items` and `project_todo` and accepting either key on load.
- Recorded the workspace caveat and the decision trail in `findings.txt` and `experiments/nanohorizon_shared_history_e2e/experiment_log.txt`.

## Validation

Verification passed under `uv`.
The command was:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Result: `4` tests passed.

## Caveats

- The expected Craftax files named in the task brief were not present in the git seed, so there was no existing harness code to patch.
- The `/app` tree contains the richer NanoHorizon implementation context, including persisted `project_todo` state, but it is not the tracked git checkout here.
- The workspace still does not include the named Craftax harness files, so this candidate stays intentionally narrow and does not modify those surfaces.
