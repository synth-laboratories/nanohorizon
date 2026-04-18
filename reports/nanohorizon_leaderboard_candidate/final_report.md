# NanoHorizon Leaderboard Candidate

## Context & Objective

The goal for this run was to produce a real NanoHorizon Craftax submission candidate by editing only [submission/agent.py](/workspace/submission/agent.py), then run a lightweight honest eval on the train seeds and publish the branch as a reviewable GitHub PR.

The candidate hypothesis was narrow:

- suppress sapling and plant detours unless survival forces them
- prioritize trees for wood
- move immediately to table placement, wood pickaxe, then stone pickaxe
- use a simple fallback search rule when no useful target is adjacent

## Experiments Cited

1. [submission/agent.py](/workspace/submission/agent.py)
   - Question: can a single-file prompt/config change express the intended tree-first Craftax candidate?
   - Outcome: supporting for packaging, inconclusive for quality.
   - Evidence: the file now sets a short reactive policy prompt, one-action batch sizing, and a reduced max-step budget.

2. [reports/nanohorizon_leaderboard_candidate/eval/result.json](/workspace/reports/nanohorizon_leaderboard_candidate/eval/result.json)
   - Question: does the final candidate show any train-seed lift under honest rollout evaluation?
   - Outcome: negative.
   - Evidence: six train seeds were evaluated successfully, but `primary_score` and `mean_outcome_reward` were both `0.0`, with no achievements unlocked.

3. [reports/nanohorizon_leaderboard_candidate/eval/eval_summary.json](/workspace/reports/nanohorizon_leaderboard_candidate/eval/eval_summary.json)
   - Question: what did the rollout harness observe at the aggregate level?
   - Outcome: negative.
   - Evidence: `num_eval_rollouts = 6`, `num_rollout_errors = 0`, `mean_llm_calls_per_rollout = 8.0`, and `achievement_names = []`.

## Insights

1. The prompt change was operationally valid but not enough to solve the search problem on this observation surface. The model repeatedly chose movement actions and never reached a reward-bearing interaction on the six train seeds.
2. Suppressing saplings and plants removed the known distraction class, but the remaining failure mode is exploration. The agent still needs a stronger directional/search policy to find trees and resources before the short horizon expires.
3. The honest eval path is now verified end to end. The run can reproduce the train-seed result without hidden mocks, and the rollout artifacts are present in `reports/nanohorizon_leaderboard_candidate/eval/`.

## Research Artifacts Produced

### Environments

- Local Craftax runtime via the repo rollout harness.
- Remote inference endpoint:
  - `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- Installed runtime dependency for honest eval:
  - `craftax` from PyPI, plus its JAX stack in the workspace Python environment.

### Data

- Train seeds: `10007`, `10008`, `10011`, `10014`, `10018`, `10003`
- Source: [data/craftax/craftax_prompt_opt_starter_seeds.json](/workspace/data/craftax/craftax_prompt_opt_starter_seeds.json)

### Models / Checkpoints

- Submission checkpoint:
  - [reports/nanohorizon_leaderboard_candidate/train/checkpoint.json](/workspace/reports/nanohorizon_leaderboard_candidate/train/checkpoint.json)
- Eval outputs:
  - [reports/nanohorizon_leaderboard_candidate/eval/result.json](/workspace/reports/nanohorizon_leaderboard_candidate/eval/result.json)
  - [reports/nanohorizon_leaderboard_candidate/eval/eval_summary.json](/workspace/reports/nanohorizon_leaderboard_candidate/eval/eval_summary.json)

## Quality & Validation

- Executed honest rollout eval on the six train seeds listed above.
- Eval command used the remote Gemini OpenAI-compatible endpoint with `gemini-2.5-flash-lite`.
- Observed aggregate result:
  - `primary_score = 0.0`
  - `mean_outcome_reward = 0.0`
  - `num_rollout_errors = 0`
  - `mean_llm_calls_per_rollout = 8.0`
- Explicitly not validated:
  - held-out seed lift
  - leaderboard submission score
  - any broader search strategy beyond the prompt-level candidate

## Reproduction & Handoff

- Candidate entrypoint: `python -m submission.agent`
- Honest eval command:
  - `NANOHORIZON_EVAL_INFERENCE_URL='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions' NANOHORIZON_EVAL_REQUEST_MODEL='gemini-2.5-flash-lite' NANOHORIZON_EVAL_API_KEY=... python -m submission.agent eval --checkpoint-dir /workspace/reports/nanohorizon_leaderboard_candidate/train --data-dir /workspace/data --out-dir /workspace/reports/nanohorizon_leaderboard_candidate/eval`
- Main risk remaining:
  - the candidate still defaults to a one-direction drift when the model fails to infer a stronger search heuristic from the prompt alone
- Open follow-up:
  - replace the fallback search instruction with something stronger than a memoryless parity rule if the next run can budget another controlled comparison
