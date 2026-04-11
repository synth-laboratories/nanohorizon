# NanoHorizon Craftax Candidate Report

## Context & objective

The workspace started as a near-empty checkout and the task was to produce a concrete NanoHorizon leaderboard candidate for Craftax with the smallest honest improvement. The chosen candidate label is `Full Auto E2E`, and the strategy is to keep a compact Todo Tool scratchpad explicit while preserving the shared harness surfaces unless a direct justification exists.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
   - Question: what is the smallest candidate config that captures the Todo Tool strategy and the requested rollout shape?
   - Outcome: supporting.
   - Evidence: the YAML config encodes the candidate label, strategy, seed prompt, todo contract, rollout settings, and output root.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: can the candidate strategy be represented as a reproducible prompt-opt scaffold with a compact private todo list?
   - Outcome: supporting.
   - Evidence: the module builds the seed prompt, todo scratchpad directive, and candidate config used by the smoke path.

3. `tests/test_candidate.py`
   - Question: does the candidate metadata/prompt/runner stay aligned with the chosen strategy?
   - Outcome: supporting verifier.
   - Evidence: the smoke test checks manifest content, prompt content, and manifest writing behavior.

4. `scripts/run_craftax_model_eval.sh`
   - Question: can the candidate be exercised reproducibly through `uv` with artifacts written to disk?
   - Outcome: supporting.
   - Evidence: the script runs the prompt-opt smoke path and the unittest suite.

## Insights

1. The smallest honest change is the Todo Tool scratchpad itself, not a broader harness rewrite. The config and prompt helper keep that change explicit and reviewable.
2. The candidate is reproducible from the workspace because the manifest and prompt are emitted to `experiments/nanohorizon_leaderboard_candidate/results/` and validated by `tests/test_candidate.py`.
3. The verifier surface exposed one real caveat: `uv run --python 3.11 ...` fails because `pyproject.toml` requires Python >=3.12. The final harness path is now explicit about Python 3.12 in `scripts/run_craftax_model_eval.sh`.

## Research artifacts produced

### Environments

- Project metadata: `pyproject.toml`
- Entry modules: `src/nanohorizon/baselines/prompt_opt.py`, `src/nanohorizon/craftax_core/metadata.py`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`
- Smoke entrypoint: `scripts/run_craftax_model_eval.sh`

### Data

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
- Candidate command note: `records/prompt_opt_1usd_gpt54_family/2026-04-11_full_auto_e2e/command.txt`
- Candidate run metadata and metrics: `records/prompt_opt_1usd_gpt54_family/2026-04-11_full_auto_e2e/metadata.json`, `metrics.json`, `run_config.yaml`, `system_info.json`

### Models / checkpoints

- No model weights were trained or promoted in this run.
- The submission is a prompt/context-shaping candidate only.

## Quality & validation

- Validation command: `bash scripts/run_craftax_model_eval.sh`
- Direct smoke validation also passed with `uv run python -m unittest discover -s tests -v`
- The smoke path restored `runner.py --write`, so the manifest emission path is covered by the verifier.
- The smoke test covers the candidate metadata, prompt text, and manifest writing behavior.
- Explicitly not validated: real Craftax benchmark score movement, remote rollout behavior, or any SFT/RL training loop.

## Reproduction & handoff

- Reproduce with: `bash scripts/run_craftax_model_eval.sh`
- Review the candidate manifest in `experiments/nanohorizon_leaderboard_candidate/results/candidate_manifest.json`
- Review the candidate prompt in `experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt`
- Review the prompt-opt record bundle in `records/prompt_opt_1usd_gpt54_family/2026-04-11_full_auto_e2e/`
- Open risk: the workspace still does not contain a real Craftax environment rollout, so this run establishes a reproducible candidate scaffold rather than a benchmark-measured leaderboard delta.
