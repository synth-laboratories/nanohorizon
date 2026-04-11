# Shared History E2E

## Context & Objective

The run goal was to implement a compact NanoHorizon candidate around shared history and a todo scratchpad, while preserving the shared Craftax harness surfaces unless a verified necessity required a change.

The git checkout was effectively empty at start, so the honest implementation target became a self-contained NanoHorizon todo board that can be reused by future harness wiring.

## Change

- Added `src/nanohorizon/craftax_core/todo_tool.py` with a small `TodoBoard` and `TodoItem` model.
- Added `tests/test_todo_tool.py` coverage for round-trip serialization, ordered todo creation, `project_todo` alias compatibility, missing-item handling, and invalid-status coercion.
- Aligned the board serialization with the richer `/app` optimizer-state shape by emitting both `items` and `project_todo` and accepting either key on load.
- Kept the tests on the editable `uv` install path so the suite imports `nanohorizon` directly without a local `sys.path` shim.
- Recorded the workspace caveat and the decision trail in `findings.txt` and `experiments/nanohorizon_shared_history_e2e/experiment_log.txt`.

## Validation

Verification passed under `uv`.
The command was:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Result: `5` tests passed.

## Caveats

- The expected Craftax files named in the task brief were not present in the git seed, so there was no existing harness code to patch.
- The `/app` tree contains the richer NanoHorizon implementation context, including persisted `project_todo` state, but it is not the tracked git checkout here.
- The workspace still does not include the named Craftax harness files, so this candidate stays intentionally narrow and does not modify those surfaces.
- GitHub PR creation is blocked by the configured allowlist for this workspace. `github_push` rejected every repo slug tried for this project, including `synth/nanohorizon`, `smr/nanohorizon`, and the UUID-shaped binding derived from the git remote, so the branch could not be published as a real GitHub PR from here.
