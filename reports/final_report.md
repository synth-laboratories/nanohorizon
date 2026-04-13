# Craftax Local State Handling Candidate

## Context & objective

The objective was to make the smallest honest Craftax change that improves reliability by clarifying local runtime state handling, while keeping the shared harness surfaces stable. The specific risk under review was whether `DeterministicCraftaxRunner` checkpoint/restore could alias mutable environment state and leak later mutations into repeated rollouts.

## Experiments cited

1. `PYTHONPATH=src uv run --no-sync python - <<'PY' ...` repeated-seed mutable-state rewind check on seeds `0..7` using `checkpoint(copy_state=False)` as the aliasing baseline
   - Question: does the pre-change aliasing path preserve rewind correctness when the environment mutates state in place?
   - Outcome: negative.
   - Evidence: `experiments/craftax_candidate_local_state_handling/results/state_rewind_reliability.json` records `0/8` matches for the baseline.

2. `tests/test_craftax_core_runner.py::test_checkpoint_snapshots_mutable_runtime_state_by_default`
   - Question: does the candidate keep checkpoint/restore isolated from in-place mutation across repeated seeds?
   - Outcome: supporting.
   - Evidence: the regression test passes under the candidate code path.

3. `PYTHONPATH=src uv run --no-sync python -m pytest -q tests/test_craftax_core_runner.py::test_checkpoint_snapshots_mutable_runtime_state_by_default tests/test_craftax_core_runner.py::test_runner_checkpoint_restore_and_rewind`
   - Question: do the candidate change and the pre-existing Craftax runner tests still pass together?
   - Outcome: supporting.
   - Evidence: 2 tests passed.

4. `PYTHONPATH=src uv run --no-sync python - <<'PY' ...` repeated-seed mutable-state rewind check on seeds `0..7` using `checkpoint()` as the candidate path
   - Question: does the candidate actually repair the aliasing issue across repeated seeds?
   - Outcome: supporting.
   - Evidence: `experiments/craftax_candidate_local_state_handling/results/state_rewind_reliability.json` records `8/8` matches for the candidate.

## Insights

1. The failure mode was real: the aliasing checkpoint path let later in-place mutations leak into restore.
2. The smallest reliable fix is local to `src/nanohorizon/craftax_core/runner.py`: make `checkpoint()` snapshot state by default and restore the saved `last_info` alongside the state payload.
3. The change is behaviorally meaningful rather than cosmetic: on the same repeated-seed mutable-state harness, rewind moved from `0/8` correct matches to `8/8`.
4. The existing Craftax interface surfaces stayed stable. The fix did not require changes to `docs/task-craftax.md`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/metadata.py`, or `scripts/run_craftax_model_eval.sh`.

## Research artifacts produced

- Source change: `src/nanohorizon/craftax_core/runner.py`
- Regression test: `tests/test_craftax_core_runner.py`
- Experiment log: `experiments/craftax_candidate_local_state_handling/experiment_log.txt`
- Result artifact: `experiments/craftax_candidate_local_state_handling/results/state_rewind_reliability.json`
- Repo handoff notes: `findings.txt`

## Quality & validation

- Validated the candidate against a repeated-seed mutable-state harness using 8 seeds.
- Validated the code with `PYTHONPATH=src uv run --no-sync python -m pytest -q tests/test_craftax_core_runner.py::test_checkpoint_snapshots_mutable_runtime_state_by_default tests/test_craftax_core_runner.py::test_runner_checkpoint_restore_and_rewind`.
- Known caveat: the repository’s normal `uv` project sync path is blocked by a machine-specific `cloud` source in `pyproject.toml` that points at `/Users/joshpurtell/Documents/GitHub/synth-ai`. Verification therefore used `--no-sync` plus explicitly installed runtime packages.
- Not validated: live Craftax leaderboard score, Modal execution, or any broader harness changes beyond the runner checkpoint/restore path.

## Reproduction & handoff

- Candidate change: `src/nanohorizon/craftax_core/runner.py` now snapshots `checkpoint()` by default and copies checkpoint payloads plus `last_info` during restore.
- Regression command:
  - `PYTHONPATH=src uv run --no-sync python -m pytest -q tests/test_craftax_core_runner.py::test_checkpoint_snapshots_mutable_runtime_state_by_default tests/test_craftax_core_runner.py::test_runner_checkpoint_restore_and_rewind`
- Rewind comparison command:
  - `PYTHONPATH=src uv run --no-sync python - <<'PY' ...` repeated-seed mutable-state rewind check over seeds `0..7`, comparing `checkpoint(copy_state=False)` to `checkpoint()`
- Open risk: if external callers intentionally depend on aliasing checkpoints for performance, they now need to opt out explicitly with `copy_state=False`.
- Commit: `5709fe5700d3191f6152f9b777debbe4135aa7e0`
- GitHub PR: [#44](https://github.com/synth-laboratories/nanohorizon/pull/44)
