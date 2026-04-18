# NanoHorizon Craftax Candidate Eval Report

## Candidate

- Task: `nanohorizon_leaderboard_candidate`
- Candidate label: `codex-20260418T103225Z`
- Edited surface: `submission/agent.py`

## What Changed

- Replaced the generic Craftax system prompt with an explicit achievement ladder.
- Added a concrete fallback exploration policy:
  - `move_right, move_right, move_down, move_down, move_left, move_left, move_up, move_up`
- Enabled thinking and shortened the rollout configuration in the submission config.
- Kept the submission contract in a single file and preserved the `train(data_dir, checkpoint_dir)` / `eval(checkpoint_dir, data_dir, out_dir)` entrypoints.

## Validation

- Syntax check: `python -m py_compile submission/agent.py`
- Contract smoke: stubbed `evaluate_model()` returned a 3.0 primary score and exercised the submission wrapper end to end.
- Honest live smoke: one train seed, one rollout step, OpenAI-compatible inference, `primary_score = 0.0`, `llm_call_count = 2`, no achievements.

## Caveats

- Wider honest live eval attempts in this container were killed before completion.
- Because of that, this workspace did not produce evidence that the candidate cleared the `primaryScore > 2.5` target.
- The prompt and rollout settings are still aligned with the repo harness contract, so the code is reviewable and runnable even though the measured score here is not yet competitive.
