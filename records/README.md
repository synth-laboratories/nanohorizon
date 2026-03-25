# Records

Each submission lives in:

- `records/<track>/<date>_<name>/`

Minimum required files:

- `metadata.json`
- `metrics.json`
- `system_info.json`
- `command.txt`
- `run_config.yaml`

Optional files:

- `notes.md`
- `artifacts/manifest.json`
- `eval_rollouts.jsonl`
- `train.log`

The validator (`uv run python -m nanohorizon.shared.validate_record`, or `PYTHONPATH=src python3 -m …` without uv) checks only the minimum bundle shape for now.

New submissions should also expose held-out achievement coverage in `metrics.json`:

- `submission_mean_outcome_reward` for the held-out eval set
- `submission_achievement_frequencies` for the canonical full-Craftax achievements over the held-out 20 seeds
- each achievement entry should include both `count` and `frequency`

## Offline / SFT Records

This mirrors the same pattern as a lightweight benchmark board: each row links to a self-contained `records/.../` bundle with the command, config, metrics, and system info needed to reproduce the run.

| Run | Score | Student | Teacher | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- | --- |
| `reference_baseline` | `0.5` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` | `2026-03-20` | [info](offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) | `./scripts/run_offline_training.sh` |
| `modal_4b_nochange_baseline` | `0.7` | `Qwen/Qwen3.5-4B` | `-` | `2026-03-22` | [info](offline_20min_1xa100_40gb/2026-03-22_modal_4b_nochange_baseline/) | `./scripts/run_craftax_model_eval.sh` |

## RLVR Records

The RLVR reference path now mirrors the same structure: one user-editable Python file and one stable shell runner.

Interpret RLVR rows as follows:

- `Score` is the completed run's `metrics.json -> final_mean_outcome_reward`
- bootstrap and post-update movement live in `periodic_eval/step_XXX/summary.json`
- a partial exploratory run can be useful for debugging, but should not replace the checked-in reference row unless it finishes with a clean final bundle

| Run | Score | Model | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- |
| `reference_baseline` | `0.0` | `Qwen/Qwen3.5-4B` | `2026-03-21` | [info](rlvr_20min_2xa100_40gb/2026-03-21_reference_baseline/) | `./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh` |

## Prompt-opt Records

Interpret prompt-opt rows as follows:

- `Score` is the completed run's `metrics.json -> primary_score`, which is actual Craftax `mean_outcome_reward`
- `bootstrap_score` in `metrics.json` is the seed prompt baseline on the same held-out eval set
- `best_gepa_val_score` is GEPA's internal search objective and does not replace the real reward score

| Run | Score | Model | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- |
| `reference_baseline` | `0.35` | `Qwen/Qwen3.5-4B` | `2026-03-21` | [info](prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/) | `./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh` |

## Classic Records

Interpret classic rows as follows:

- the environment must come from the upstream JAX `craftax` package
- the run must target the `1M` Craftax-Classic regime
- the tracked policy must stay under `100M` parameters and start from random init

For classic, interpret `Score` as `eval_summary.json -> mean_episode_return`.

| Run | Score | Model | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- |
| `modal_reference_baseline` | `0.543` | `PPO-RNN` | `2026-03-24` | [info](classic/2026-03-24_modal_reference_baseline/) | `./scripts/run_classic_craftax_1m_modal.sh` |
