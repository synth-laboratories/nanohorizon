# NanoHorizon Craftax Candidate Report

## Context & objective

The workspace was used to produce the smallest honest NanoHorizon Craftax candidate that still improves the prompt-side approach. The chosen candidate label is `Full Auto E2E`, and the strategy is to keep the Todo Tool contract explicit while preserving the shared Craftax harness surfaces.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
   - Question: what candidate config captures the requested Todo Tool strategy and rollout shape?
   - Outcome: supporting.
   - Evidence: the config encodes the nested candidate, policy, optimizer, prompt, rollout, and output settings.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: can the candidate prompt contract be represented as a reproducible prompt-opt scaffold?
   - Outcome: supporting.
   - Evidence: the module defines the seed prompt, todo contract, candidate config, and smoke artifact writer.

3. `src/nanohorizon/craftax_core/metadata.py` and `src/nanohorizon/craftax_core/runner.py`
   - Question: can the preserved Craftax-facing surfaces round-trip the candidate config and emit artifacts?
   - Outcome: supporting verifier.
   - Evidence: `metadata.py` renders the candidate manifest and prompt text, while `runner.py` validates the config and writes `smoke_summary.json`, `candidate_manifest.json`, and `candidate_prompt.txt`.

4. `experiments/nanohorizon_leaderboard_candidate/results/smoke_summary.json`
   - Question: did the smoke path run end-to-end and preserve the expected summary payload?
   - Outcome: supporting verifier.
   - Evidence: the summary records the candidate manifest, loaded config, prompt turn, and verification modes.

5. `scripts/run_craftax_model_eval.sh`
   - Question: can the candidate be exercised reproducibly through `uv` with artifacts written to disk?
   - Outcome: supporting.
   - Evidence: the script runs the Craftax runner through an editable install and writes the summary and prompt artifacts into the results directory.

## Insights

1. The smallest honest change is the Todo Tool prompt contract itself, not a broader harness rewrite. The candidate keeps the change explicit and reviewable.
2. The candidate is reproducible from the workspace because the manifest, prompt, and smoke summary are emitted to `experiments/nanohorizon_leaderboard_candidate/results/` by the verifier path.
3. The verifier surfaced one real integration issue: the smoke lane must run with an editable install so `nanohorizon` imports resolve correctly. That is now baked into `scripts/run_craftax_model_eval.sh`.

## Research artifacts produced

### Environments

- Project metadata: `pyproject.toml`
- Candidate module: `src/nanohorizon/baselines/prompt_opt.py`
- Craftax surfaces: `src/nanohorizon/craftax_core/metadata.py`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`
- Smoke entrypoint: `scripts/run_craftax_model_eval.sh`

### Data

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml`
- Smoke summary: `experiments/nanohorizon_leaderboard_candidate/results/smoke_summary.json`
- Candidate manifest: `experiments/nanohorizon_leaderboard_candidate/results/candidate_manifest.json`
- Candidate prompt: `experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt`

### Models / checkpoints

- No model weights were trained or promoted in this run.
- The submission is a prompt/context-shaping candidate only.

## Quality & validation

- Validation commands: `bash scripts/run_craftax_model_eval.sh` and `uv run --with-editable . --python 3.12 python -m unittest discover -s tests -v`
- The smoke path now writes the manifest, prompt, and summary artifacts and the unittest smoke suite passes against the same tree.
- Explicitly not validated: real Craftax benchmark score movement, remote rollout behavior, or any SFT/RL training loop.

## Reproduction & handoff

- Reproduce with: `bash scripts/run_craftax_model_eval.sh`
- Review the candidate manifest in `experiments/nanohorizon_leaderboard_candidate/results/candidate_manifest.json`
- Review the smoke summary in `experiments/nanohorizon_leaderboard_candidate/results/smoke_summary.json`
- Review the candidate prompt in `experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt`
- Open risk: the workspace still does not contain a real Craftax environment rollout, so this run establishes a reproducible candidate scaffold rather than a benchmark-measured leaderboard delta.
