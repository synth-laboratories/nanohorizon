Candidate package for the single-step progress variant.

This bundle is intentionally marked not run. The live Craftax runtime packages
needed for a real rollout were unavailable in this workspace, so the tracked
evidence for this turn lives in `experiments/2026-04-13_codex_local_rerun2_2/`
instead.

Proxy comparison summary:
- repeated seeds: 10001, 10010, 10017, 10019
- repeats: 2
- baseline mean_outcome_reward: 2.000
- candidate mean_outcome_reward: 2.000
- candidate mean_llm_call_count: 5.750 vs baseline 12.000
- caveat: this is a local proxy using the repo rollout path with a fake runner,
  not the live Craftax leaderboard runtime.
