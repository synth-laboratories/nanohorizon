# NanoHorizon Daytona E2E Run 3

## Context & objective

This run implemented a small prompt-opt candidate for the Craftax todo-tool
strategy. The goal was to keep the existing Craftax surfaces stable while
adding a reviewable `Daytona E2E Run 3` variant that preserves the compact
three-item scratchpad contract and adds a small end-to-end handoff guard.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_codex_daytona_e2e_run_3.yaml`
   - Question: does the new candidate variant keep the same budget and rollout
     envelope while tightening the todo handoff wording?
   - Outcome: supporting.
   - Evidence: the config keeps the same model, optimizer budget, rollout
     shape, and seed split as the existing todo-refresh-gate baseline.
2. `tests/test_daytona_e2e_run_3_candidate.py`
   - Question: is the new config packaged reproducibly and grounded in the
     central todo contract?
   - Outcome: supporting.
   - Evidence: the test checks the config wording, the centralized
     `TODO_SCRATCHPAD_REQUIREMENTS`, and the record bundle metadata.
3. `tests/test_codex_todo_refresh_gate_candidate.py`
   - Question: does the shared todo-contract source remain stable while the
     new candidate is added?
   - Outcome: supporting.
   - Evidence: the existing todo-refresh-gate checks still pass.
4. `uv run --no-project --with pyyaml --with pytest --python 3.11 pytest -q tests/test_daytona_e2e_run_3_candidate.py tests/test_codex_todo_refresh_gate_candidate.py`
   - Question: does the candidate surface stay structurally valid under local
     verification?
   - Outcome: supporting.
   - Evidence: `6 passed in 0.06s`.

## Insights

1. The smallest honest improvement is still a prompt/config refinement, not a
   harness rewrite. The existing todo contract in
   `src/nanohorizon/baselines/prompt_opt.py` already centralizes the scratchpad
   behavior; the new config just tightens the handoff wording.
2. Keeping the seed prompt close to the existing todo-refresh-gate baseline
   preserves the original search envelope while making the new variant easy to
   review.
3. A separate candidate bundle is useful even when the underlying contract is
   shared, because it gives the new variant a stable artifact path and a
   reproducible command.

## Research artifacts produced

### Environments

- Prompt-opt config at
  `configs/craftax_prompt_opt_qwen35_4b_codex_daytona_e2e_run_3.yaml`
- Record bundle at
  `records/prompt_opt_1usd_gpt54_family/2026-04-11_daytona_e2e_run_3/`

### Data

- No new training data was introduced.
- The config reuses the existing `craftax_prompt_opt_starter_seeds.json`
  split.

### Models / checkpoints

- No model was trained or checkpointed.

## Quality & validation

- Structural pytest coverage passed for the new candidate and the existing
  todo-refresh-gate candidate.
- The config stayed within the established 1 USD budget and 3/4 action rollout
  envelope.
- Not validated: live Craftax reward, Modal runtime behavior, or GEPA search
  output.

## Reproduction & handoff

- Candidate label: `Daytona E2E Run 3`
- Strategy: `Todo Tool`
- Reproduce locally with:

```bash
uv run --no-project --with pyyaml --with pytest --python 3.11 pytest -q tests/test_daytona_e2e_run_3_candidate.py tests/test_codex_todo_refresh_gate_candidate.py
NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_daytona_e2e_run_3.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh
```

- Open risk: the reward impact of the new handoff wording remains unmeasured
  until a live optimization run is executed.

