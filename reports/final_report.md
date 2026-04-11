# NanoHorizon Craftax Candidate Report

## Context & Objective

The workspace started as a near-empty checkout, but the meaningful candidate scaffold was present in untracked files under `src/nanohorizon/baselines/prompt_opt.py`, `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`, and the Craftax smoke shim. The task was to make the smallest honest improvement to the Craftax approach, keep the shared harness surfaces stable, and validate the result with a reproducible smoke pass.

The candidate is `Full Auto E2E` using the `Todo Tool` strategy. The improvement is packaging the prompt contract into a nested candidate config, a compact prompt helper, and a smoke runner that round-trips the config and emits candidate artifacts.

## Experiments Cited

1. `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
   - Question: what candidate config captures the requested Todo Tool strategy and rollout shape?
   - Outcome: supporting.
   - Evidence: the file encodes the nested candidate, policy, optimizer, prompt, rollout, data, and output settings.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: can the candidate prompt contract be represented as a reproducible prompt-opt scaffold?
   - Outcome: supporting.
   - Evidence: the module defines the seed prompt, todo contract, nested candidate config, and config loader.

3. `src/nanohorizon/craftax_core/metadata.py` and `src/nanohorizon/craftax_core/runner.py`
   - Question: can the preserved Craftax-facing surfaces round-trip the candidate config and emit artifacts?
   - Outcome: supporting verifier.
   - Evidence: `metadata.py` renders the candidate manifest and prompt text, while `runner.py` validates the config and writes `candidate_summary.json` and `candidate_prompt.txt`.

4. `experiments/nanohorizon_leaderboard_candidate/results/candidate_summary.json`
   - Question: did the smoke path run end-to-end and preserve the expected summary payload?
   - Outcome: supporting verifier.
   - Evidence: the summary records the candidate config, loaded config, and output directory for the smoke run.

5. `scripts/run_craftax_model_eval.sh`
   - Question: can the candidate be exercised reproducibly through `uv` with artifacts written to disk?
   - Outcome: supporting.
   - Evidence: the script runs the Craftax runner through an editable install and writes the summary and prompt artifacts into the configured results directory.

## Insights

1. The smallest honest change is the Todo Tool prompt contract itself, not a broader harness rewrite. The candidate keeps the change explicit and reviewable.
2. The candidate is reproducible because the nested config and `candidate_config()` now round-trip cleanly, and the smoke runner emits durable artifacts for future inspection.
3. The verifier surfaced one real integration issue: the smoke lane must run with an editable install so `nanohorizon` imports resolve correctly. That is now baked into `scripts/run_craftax_model_eval.sh`.

## Research Artifacts Produced

### Environments

- Project metadata: `pyproject.toml`
- Candidate module: `src/nanohorizon/baselines/prompt_opt.py`
- Craftax surfaces: `src/nanohorizon/craftax_core/metadata.py`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`
- Smoke entrypoint: `scripts/run_craftax_model_eval.sh`

### Data

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
- Candidate summary: `experiments/nanohorizon_leaderboard_candidate/results/candidate_summary.json`
- Candidate prompt: `experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt`

### Models / Checkpoints

- No model weights were trained or promoted in this run.
- The submission is a prompt/context-shaping candidate only.

## Quality & Validation

- Validation command: `bash scripts/run_craftax_model_eval.sh`
- Additional smoke: `uv run --with-editable . --python 3.12 python -m unittest discover -s tests -v`
- The smoke path now writes the candidate summary and prompt artifacts and the unittest smoke suite passes against the same tree.
- Explicitly not validated: real Craftax benchmark score movement, remote rollout behavior, or any SFT/RL training loop.

## Reproduction & Handoff

- Reproduce with: `bash scripts/run_craftax_model_eval.sh`
- Review the candidate summary in `experiments/nanohorizon_leaderboard_candidate/results/candidate_summary.json`
- Review the candidate prompt in `experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt`
- Open risk: the workspace still does not contain a real Craftax environment rollout, so this run establishes a reproducible candidate scaffold rather than a benchmark-measured leaderboard delta.

