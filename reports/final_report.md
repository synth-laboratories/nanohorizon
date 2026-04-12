# Craftax Repeat-Action Parser Candidate

## Context & objective

The objective for this run was to make a small but meaningful Craftax improvement in the NanoHorizon repo, keep the harness stable, and back the candidate with a reproducible baseline-vs-candidate comparison.

The concrete bug fixed here is in Craftax action extraction: the parser was collapsing repeated valid actions inside a tool call, which destroys macro-actions like `move_right, move_right, move_right, move_right`.

## Experiments cited

1. `experiments/craftax_repeat_actions_parser/results/baseline_vs_candidate.json`
   - Question: does preserving repeated valid actions improve a deterministic fixed-seed rollout slice?
   - Outcome: supporting.
   - Evidence: baseline mean outcome reward `0.0`, candidate mean outcome reward `1.0`, baseline mean action count `1`, candidate mean action count `4`.

2. `tests/test_craftax_core_runtime_guarantees.py`
   - Question: does the parser preserve repeated valid actions and still normalize invalid tokens?
   - Outcome: supporting.
   - Evidence: focused runtime test now asserts repeated `move_right` entries survive extraction.

3. `tests/test_craftax_core_contract.py`
   - Question: does the HTTP shim still expose the rollout route and proxy the rollout helper expected by the FastAPI contract?
   - Outcome: supporting.
   - Evidence: contract test now passes with the shim wrapper and route wiring intact.

4. `tests/test_craftax_interface.py`
   - Question: does the prompt-history shaping code remain internally consistent?
   - Outcome: supporting.
   - Evidence: unit test covers the reward-history summary/advice fields that are already present in the dirty tree.

## Insights

1. Repeated macro-actions were being silently compressed before execution. That is a real Craftax behavior bug, not just a formatting nit.
2. Preserving duplicates increases executed action count on the deterministic slice from `1` to `4` and lifts the synthetic fixed-seed outcome reward from `0.0` to `1.0`.
3. The improvement is localized to the action sanitizers in `src/nanohorizon/craftax_core/rollout.py` and `src/nanohorizon/shared/openai_compat.py`; the shared evaluation path still behaves normally after the change.
4. The worktree also contained a prompt-history augmentation in `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/metadata.py`, and `tests/test_craftax_interface.py`. Those edits passed unit tests but were not isolated in the reward slice.

## Research artifacts produced

### Environments

- Validation commands ran under `uv run --no-project` with ad hoc package injection for `httpx`, `numpy`, `pyyaml`, and `pytest`.
- The deterministic comparison used `run_rollout_request` with a fake runner and fixed seeds `[11, 23, 37]`.

### Data

- Deterministic evaluation seeds: `11`, `23`, `37`.
- Fixed tool-call payload: a repeated `move_right` batch of length `4`.

### Models / checkpoints

- No checkpoints were trained or selected in this run.

## Quality & validation

- `uv run --no-project --with httpx --with numpy --with pyyaml --with pytest --with fastapi python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_core_runtime_guarantees.py tests/test_craftax_interface.py -q`
- Result: `18 passed`
- Deterministic fixed-seed slice:
  - baseline mean outcome reward: `0.0`
  - candidate mean outcome reward: `1.0`
  - baseline mean action count: `1.0`
  - candidate mean action count: `4.0`

Known limits:

- This comparison isolates parser behavior with a fake runner. It is not a live Craftax model endpoint or a full environment benchmark.
- I did not validate a real upstream Qwen endpoint in this run.

## Reproduction & handoff

- Source change: `src/nanohorizon/craftax_core/rollout.py`
- Shared sanitizer: `src/nanohorizon/shared/openai_compat.py`
- Supporting interface tests: `tests/test_craftax_core_runtime_guarantees.py`, `tests/test_craftax_interface.py`
- Experiment artifacts: `experiments/craftax_repeat_actions_parser/`
- Exact comparison command: `experiments/craftax_repeat_actions_parser/command.txt`
- Main open risk: this is a parser-level improvement; the reward gain is synthetic and should be rechecked on a live Craftax endpoint when one is available.
