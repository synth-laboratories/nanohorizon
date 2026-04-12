# Achievement Roadmap v2

- Candidate scope: `src/nanohorizon/shared/eval_model.py`, `src/nanohorizon/craftax_core/rollout.py`, and the minimal HTTP shim restoration needed to keep the existing Craftax contract testable.
- Prompt change: add an explicit achievement progression roadmap and a zero-reward-for-repeat framing to the default eval system prompt.
- Rollout change: carry `achievements_unlocked` and `next_targets` into each user prompt and track them in rollout metadata/turn traces.
- Harness repair: restore a small `create_app()` surface in `http_shim.py` so the repo's own Craftax contract tests can collect again.
- Validation: targeted Craftax tests passed after the shim repair.
- Benchmark eval: blocked. The environment lacks `jax`, `craftax`, `craftaxlm`, and `vllm`, so the requested seed-10000-to-10004 baseline-vs-candidate evaluation could not be run here.
- Baseline reference used for context: `records/offline_20min_1xa100_40gb/2026-03-22_modal_4b_nochange_baseline/eval_summary.json` reports `mean_outcome_reward = 1.0` on the seed subset `10000..10004`.
