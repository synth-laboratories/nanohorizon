# Server Push E2E Notes

- Candidate label: `Server Push E2E`
- Strategy: `Todo Tool`
- The candidate centralizes a compact scratchpad contract in `src/nanohorizon/craftax_core/metadata.py`.
- The runtime payload exposes `todo_item_count: 3` and
  `scratchpad_mode: compact-three-item`.
- The scratchpad is treated as server-pushed state: completed items come off the list immediately.
- The contract is intentionally small:
  - keep at most three live todo items
  - treat completed items as server-pushed state and remove them immediately
  - replace stalled items with the next best action
- Verification used the recorded `uv run --no-project --with pyyaml --with pytest --python 3.11 pytest -q tests/test_server_push_e2e_candidate.py`
  command, `./scripts/run_craftax_model_eval.sh`, and a direct
  `PYTHONPATH=src uv run --python 3.11 python` smoke check of
  `candidate_record()` and `validate_candidate_record()`.
- No live Craftax rollout was executed in this workspace.
