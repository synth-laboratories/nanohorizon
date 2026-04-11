# NanoHorizon Craftax Candidate Report

## Context & Objective

This workspace began as a stub checkout with only `README.md`. The task was to make the smallest honest improvement to the Craftax approach while keeping the preserved harness surfaces stable. Because the original benchmark tree was absent, the candidate is a minimal, reviewable video-validation scaffold that makes the todo-based validation state durable.

## Experiments Cited

1. `scripts/verify_video_validation_run.py`
   - Question: does the scaffold expose the expected metadata and scratchpad contract?
   - Outcome: supporting once the files were added.
   - Evidence: the verifier checks the candidate label, preserved surfaces, scratchpad presence, and runner summary shape.
2. `scripts/run_craftax_model_eval.sh`
   - Question: can the candidate be exercised through a repo-owned `uv` entrypoint?
   - Outcome: supporting.
   - Evidence: the script invokes `python -m nanohorizon.craftax_core.runner` under `uv`.
3. `experiments/craftax_full_e2e_with_video/results/verification.json`
   - Question: does the runner emit a durable machine-readable summary?
   - Outcome: supporting.
   - Evidence: the file captures the candidate metadata and scratchpad snapshot produced by `build_runner_summary()`.
4. `experiments/craftax_full_e2e_with_video/experiment_log.txt`
   - Question: is the candidate state recorded durably for later review?
   - Outcome: supporting.
   - Evidence: the log captures the stub-checkout constraint and the chosen scaffold shape.

## Insights

1. The smallest honest change in this workspace is not a policy claim. It is a compact todo scratchpad plus a stable runner/verifier loop that future work can extend.
2. The candidate remains reviewable because the harness surfaces are narrow and the evaluation entrypoint is a single `uv`-driven script.
3. No benchmark score was measured here. That is a deliberate limitation, not a hidden success.

## Research Artifacts Produced

### Environments

- `pyproject.toml`
- `scripts/run_craftax_model_eval.sh`
- `scripts/verify_video_validation_run.py`

### Data

- `experiments/craftax_full_e2e_with_video/todo.json`
- `experiments/craftax_full_e2e_with_video/results/verification.json`
- `experiments/craftax_full_e2e_with_video/experiment_log.txt`

### Models / Checkpoints

- None. This run did not train or promote a model checkpoint.

## Quality & Validation

- Verified the scaffold with `PYTHONPATH=src uv run python scripts/verify_video_validation_run.py`.
- Verified the repo-owned runner entrypoint with `bash scripts/run_craftax_model_eval.sh metadata`.
- Explicitly not validated: any real Craftax leaderboard movement, any live environment rollout, or any SFT/RL training loop.

## Reproduction & Handoff

- Run `PYTHONPATH=src uv run python scripts/verify_video_validation_run.py`
- Run `bash scripts/run_craftax_model_eval.sh metadata`
- Inspect the durable todo scratchpad at `experiments/craftax_full_e2e_with_video/todo.json`
- Inspect the machine-readable verifier result at `experiments/craftax_full_e2e_with_video/results/verification.json`
- Git branch push succeeded on `worker/run-91956353-be83-4635-a1f6-ae42700e2e74`
- GitHub PR creation was attempted with repo slug `00000000-0000-0000-0000-000000000001/77224b95-36c5-45e7-9be1-35890e9b29b7` and rejected because that repo is not in the configured GitHub repos list
- Open risk: this is a scaffold, not a measured benchmark improvement.
