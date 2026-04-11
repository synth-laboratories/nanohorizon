# NanoHorizon Craftax Candidate

## Context & objective

This run targeted the NanoHorizon leaderboard candidate `Daytona E2E Run 3`.
The objective was the smallest honest improvement to the Craftax approach:
add a compact todo scratchpad so the agent can track subgoals while keeping the
named Craftax harness surfaces stable. No SFT or RL changes were introduced.

## Experiments cited

1. `tests/test_craftax_core.py`
   - Question: does the Todo Tool scaffold expose the expected stable surfaces?
   - Outcome: supporting.
   - Evidence: `tests/test_craftax_core.py`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`.
2. `uv run --python 3.11 python -m unittest discover -s tests -v`
   - Question: does the scaffold import and behave cleanly under local verification?
   - Outcome: supporting.
   - Evidence: command output from this run.
3. `scripts/run_craftax_model_eval.sh`
   - Question: does the eval entrypoint render the candidate summary and write the smoke payload artifact?
   - Outcome: supporting.
   - Evidence: `scripts/run_craftax_model_eval.sh`, `.out/craftax_eval/smoke_payload.json`.

## Insights

1. A compact scratchpad is enough to make the candidate more inspectable without changing the broader Craftax contract. Supported by `src/nanohorizon/craftax_core/runner.py` and `tests/test_craftax_core.py`.
2. Keeping the harness surfaces explicit in metadata makes the change easy to review downstream. Supported by `src/nanohorizon/craftax_core/metadata.py` and `docs/task-craftax.md`.
3. The rollout compatibility alias can remain stable while still exposing the new todo-driven candidate context. Supported by `src/nanohorizon/craftax_core/http_shim.py`.

## Research artifacts produced

### Environments

- Local project scaffold in `/workspace`.
- Python entrypoint via `uv run --python 3.11 python -m nanohorizon.craftax_core.runner`.

### Data

- No external dataset was introduced.
- The candidate uses a fixed todo board defined in `src/nanohorizon/craftax_core/metadata.py`.

### Models / checkpoints

- No model was trained or checkpointed.

## Quality & validation

- Added unittest coverage for the todo board rendering, the `/rollout` alias,
  the runner summary shape, and the smoke payload write path.
- The package initializer exports `CraftaxRunner` lazily so the entrypoint stays
  warning-free under `python -m`.
- The eval script now emits `.out/craftax_eval/smoke_payload.json` through the
  runner compatibility path that accepts both `--format json` and the legacy
  `--smoke --json --output` flags.
- Not validated: actual benchmark scoring, remote container execution, or any
  training-time behavior.

## Reproduction & handoff

- Candidate label: `Daytona E2E Run 3`
- Strategy: `Todo Tool`
- Reproduce locally with:

```bash
uv run --python 3.11 python -m unittest discover -s tests -v
uv run --python 3.11 python -m nanohorizon.craftax_core.runner --format json
./scripts/run_craftax_model_eval.sh
```

- Open risks: this is a minimal scaffold in an otherwise empty repository, so
  the impact is limited to the harness/doc surface added in this run.
- PR creation is blocked in this workspace because `create_github_pr` rejected
  every repo string exposed by the runtime metadata, so no configured GitHub
  repo slug was available to open the review.
