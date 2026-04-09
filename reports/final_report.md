# Craftax Exploration-Bonus Rollout Candidate

## Context & objective
- Task objective: implement a NanoHorizon leaderboard candidate change for Craftax with count-based exploration shaping in rollout decisions, while avoiding SFT/RL approaches.
- Constraint followed: keep shared harness surfaces unchanged unless required; only touched rollout logic in the core file.

## Experiments cited
1. `py_compile`
   - Path: `uv run --no-project --python 3.11 --with pyyaml python -m py_compile src/nanohorizon/craftax_core/rollout.py`
   - Question: is the modified rollout module syntactically valid after adding exploration accounting logic and compatibility helper?
   - Outcome: passing (no syntax errors).
2. `verifier_smoke_fake_runner`
   - Path: `artifacts/craftax_exploration_bonus_verifier_feedback.json`
   - Command: `cd /home/daytona/workspace && uv run --no-project --python 3.11 --with pyyaml --with numpy --with httpx --with fastapi python - <<'PY' ...`
   - Question: does per-step bonus follow `1/sqrt(visit_count[cell])` under repeated-cell visits?
   - Outcome: supporting. Fake rollout produced outcome reward `2.7071067811865475` with `exploration_bonus=2.7071067811865475` and term sequence matching positions `[(0,1), (1,1), (0,1)]` with `[1.0, 1.0, 0.7071067811865475]`.
3. `rollout_contract_delta`
   - Path: `git diff` in working tree
   - Question: are preserved harness surfaces untouched by this change?
   - Outcome: supporting. Only `src/nanohorizon/craftax_core/rollout.py` was changed, matching the task constraint list.

## Insights
1. Per-cell exploration shaping is implemented cleanly by tracking cell visit counts and adding `1/sqrt(count)` into native reward at decision time.
2. Moving from batch stepping to per-action stepping is required for correct bonus accounting; introducing `_step_runner` keeps compatibility with both `step` and legacy `step_many` runner APIs.
3. The verifier smoke demonstrates numeric behavior end-to-end for rollout scoring and trace fields, including explicit `exploration_bonus` and `native_env_reward_plus_exploration_bonus` artifacts.
4. No live environment or GPU-based execution was run; impact on benchmark leaderboard score remains unverified in this task.

## Research artifacts produced
- Modified source: `src/nanohorizon/craftax_core/rollout.py`
- Verifier artifact: `artifacts/craftax_exploration_bonus_verifier_feedback.json`
- Durable run notes: `findings.txt`
- Finalized report: `reports/final_report.md`

## Quality & validation
- Executed command:
  - `uv run --no-project --python 3.11 --with pyyaml python -m py_compile src/nanohorizon/craftax_core/rollout.py` (pass)
- Executed verifier smoke:
  - Inline `uv` Python script with fake `make_runner` / `_chat_completion` and action sequence `move_up, move_right, move_left`
  - Recorded in `artifacts/craftax_exploration_bonus_verifier_feedback.json`
- Explicit caveats:
  - Not validated on live Craftax container, no Modals/benchmarks, and no `run_craftax_model_eval.sh` call.
  - Not a full leaderboard score comparison; this task validates strategy mechanics and contract compatibility only.

## Reproduction & handoff
- Reproduce the checker locally:
  1) `cd /home/daytona/workspace`
  2) Use `uv` with the exact dependency set used above for any rollout smoke script
  3) Run the inline script that imports `nanohorizon.craftax_core.rollout` from `/home/daytona/workspace/src`
  4) Confirm `artifacts/craftax_exploration_bonus_verifier_feedback.json`.
- Recommended next step before scaling:
  - run an official submission-style live Craftax eval loop and compare benchmark metric deltas against baseline candidate.
- Risk notes:
  - this change alters scoring composition (`outcome_reward`) and could affect downstream consumers that only expect native rewards; additional downstream calibration may be needed.
