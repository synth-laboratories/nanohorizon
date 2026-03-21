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

The validator (`uv run python -m nanohorizon.validate_record`, or `PYTHONPATH=src python3 -m …` without uv) checks only the minimum bundle shape for now.

## Offline / SFT Records

This mirrors the same pattern as a lightweight benchmark board: each row links to a self-contained `records/.../` bundle with the command, config, metrics, and system info needed to reproduce the run.

| Run | Score | Student | Teacher | Summary | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `reference_baseline` | `0.5` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` | Crafter FBC with tool-calling traces, 2k thinking budget, and held-out compare (`+0.2` delta over base) | `2026-03-20` | [info](offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) | `./scripts/run_offline_training.sh` |

## RLVR Records

The RLVR reference path now mirrors the same structure: one user-editable Python file and one stable shell runner.

Interpret RLVR rows as follows:

- `Score` is the completed run's `metrics.json -> final_mean_outcome_reward`
- bootstrap and post-update movement live in `periodic_eval/step_XXX/summary.json`
- a partial exploratory run can be useful for debugging, but should not replace the checked-in reference row unless it finishes with a clean final bundle

| Run | Score | Model | Summary | Date | Info | Reproduce |
| --- | ---: | --- | --- | --- | --- | --- |
| `reference_baseline` | `0.0` | `Qwen/Qwen3.5-4B` | Clustered Modal Crafter GRPO smoke run with one public Crafter service, one clustered learner-plus-inference runtime, and single-script learner logic in `src/nanohorizon/rlvr_training.py` | `2026-03-21` | [info](rlvr_20min_2xa100_40gb/2026-03-21_reference_baseline/) | `./scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh` |
