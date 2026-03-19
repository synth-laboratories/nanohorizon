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

The validator in `tools/validate_record.py` checks only the minimum bundle shape for now.
