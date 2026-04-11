# Server Push E2E

## Context & objective

This run implemented a compact Todo Tool / scratchpad candidate for the
Craftax approach under the label `Server Push E2E`.
The objective was to make the smallest honest change that improves the agent's
ability to keep a short, current todo list while staying reviewable and
reproducible. The implemented shape keeps one shared scratchpad contract across
the task-info payload, HTTP shim, and runtime runner.

## Experiments cited

1. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does the candidate centralize a single compact scratchpad
     contract?
   - Outcome: supporting.
   - Evidence: the module defines the three-item todo contract, the prompt
     contract, the refresh helper, and the record validator.
2. `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`
   - Question: does the live Craftax surface expose the same three-item contract
     at runtime?
   - Outcome: supporting.
   - Evidence: the runtime task info includes `todo_item_count: 3`,
     `scratchpad_mode: compact-three-item`, `scratchpad_requirements`, and the
     rendered todo scratchpad.
3. `tests/test_server_push_e2e_candidate.py`
   - Question: is the candidate structurally consistent and does the runner
     payload match the contract?
   - Outcome: supporting.
   - Evidence: the test checks the label, the three-item scratchpad contract,
     the shared runner payload, and the verifier bundle.
4. `scripts/run_craftax_model_eval.sh`
   - Question: does the wrapper script still execute the candidate end-to-end?
   - Outcome: supporting.
   - Evidence: the script passed with `4 passed in 0.02s` via the recorded smoke
     path.
5. `experiments/server_push_e2e/results/verifier.json`
   - Question: is the candidate packaged durably for later inspection?
   - Outcome: supporting.
   - Evidence: the verifier bundle records the scratchpad-rendering smoke checks
     and a pass status.

## Insights

1. Centralizing the contract in the shared Craftax helper stack keeps the prompt
   wording and the runtime scratchpad in sync without broad harness changes.
2. The three-item limit is now enforced in the runtime payload itself, not just
   in the prose, which makes the contract easier to verify.
3. The candidate remains intentionally narrow: it changes prompt shaping,
   scratchpad refresh behavior, and packaging, but not model training or rollout
   infrastructure.
4. The plain `uv run --python 3.11 pytest ...` invocation is not sufficient in
   this repo because `pytest` is not a project dependency; the reproducible path
   is the explicit `uv run --no-project --with pyyaml --with pytest --python 3.11
   pytest -q tests/test_server_push_e2e_candidate.py` verifier command.

## Research artifacts produced

- Candidate contract module: `src/nanohorizon/baselines/prompt_opt.py`
- Craftax runtime mirror: `src/nanohorizon/craftax_core/metadata.py`
- Craftax runtime surface: `src/nanohorizon/craftax_core/http_shim.py`
- Runner: `src/nanohorizon/craftax_core/runner.py`
- Task doc: `docs/task-craftax.md`
- Structural test: `tests/test_server_push_e2e_candidate.py`
- Experiment verifier: `experiments/server_push_e2e/results/verifier.json`
- Candidate bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-11_server_push_e2e/`

## Quality & validation

- Executed: `uv run --no-project --with pyyaml --with pytest --python 3.11 pytest -q tests/test_server_push_e2e_candidate.py`
- Result: 4 tests passed.
- Executed: `./scripts/run_craftax_model_eval.sh`
- Result: 4 tests passed.
- Additional smoke check: direct `PYTHONPATH=src python` import and
  `validate_candidate_record(candidate_record())` both passed.
- Not validated: live Craftax rollout, benchmark reward, or GEPA search.

## Reproduction & handoff

- Reproduce with:

```bash
uv run --no-project --with pyyaml --with pytest --python 3.11 pytest -q tests/test_server_push_e2e_candidate.py
./scripts/run_craftax_model_eval.sh
```

- Open risk: no live Craftax benchmark rollout was run in this workspace, so
  the result is a structural candidate rather than a scored leaderboard
  submission.
