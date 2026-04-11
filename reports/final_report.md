# Final Report

## Context & objective
This run targeted NanoHorizon candidate `Pipeline Fix E2E` with the smallest honest change that improves the Craftax approach. The GitHub repository already contained the full Craftax stack, so the branch was narrowed to a prompt-opt candidate package rather than a harness rewrite.

## Experiments cited
- `experiments/pipeline_fix_e2e/experiment_log.txt`
  - Question: is the GitHub `main` tree a full Craftax repo, and can the candidate be scoped without changing shared harness surfaces?
  - Outcome: supportive. The clean worktree confirmed the real repo already contains the Craftax stack; the candidate was then narrowed to a new prompt-opt package and smoke-validated.
- `tests/test_codex_pipeline_fix_e2e_candidate.py`
  - Question: does the candidate config, wrapper, and record bundle match the intended `Pipeline Fix E2E` wording and packaging?
  - Outcome: supportive. The regression test passed in an ephemeral `uv` environment with `pytest` and `pyyaml`.
- `artifacts/verifier_feedback.txt`
  - Question: what smoke validation was used before declaring the candidate ready?
  - Outcome: supportive. It records the exact pytest and record-validation commands and the pass result.

## Insights
1. The seed checkout was not a faithful copy of the real NanoHorizon repo, so the correct candidate base is GitHub `main`, not the original seed branch.
2. A dedicated prompt-opt package is the smallest honest change that fits the objective without altering the shared Craftax harness surfaces.
3. The pipeline fix is implemented as prompt wording plus a narrow wrapper, which keeps the change reviewable and avoids destabilizing the existing prompt-opt runner.

## Research artifacts produced
- Environments:
  - GitHub-main worktree at `/workspace-gh`
  - Candidate wrapper in `scripts/run_craftax_prompt_opt_pipeline_fix_e2e.sh`
- Data:
  - No new external data was introduced
  - Record bundle under `records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_pipeline_fix_e2e/`
- Models / checkpoints:
  - None

## Quality & validation
- Intended smoke checks:
  - `uv run pytest tests/test_codex_pipeline_fix_e2e_candidate.py`
  - `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_pipeline_fix_e2e`
- Passed smoke checks:
  - `uv run --no-project --with pytest --with pyyaml --python 3.11 pytest tests/test_codex_pipeline_fix_e2e_candidate.py`
  - `PYTHONPATH=src python3 -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_pipeline_fix_e2e`
- Known failure mode:
  - No live Craftax rollout was executed in this task, so scoreboard impact remains unmeasured.
- Explicitly not validated:
  - Any benchmark score change
  - Any Modal or live runtime execution

## Reproduction & handoff
- Candidate branch: `worker/pipeline-fix-e2e-98e8ee64`
- Relevant files:
  - `configs/craftax_prompt_opt_qwen35_4b_codex_pipeline_fix_e2e.yaml`
  - `scripts/run_craftax_prompt_opt_pipeline_fix_e2e.sh`
  - `records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_pipeline_fix_e2e/`
- Open risk:
  - The candidate is still a prompt-shaping improvement; its actual effect on Craftax reward has not been measured.
