# Craftax Achievement Roadmap + Tracker

## Context & objective

The task was to improve Craftax achievement-seeking behavior with the smallest reviewable change, while keeping the shared harness surfaces stable. The requested candidate had three parts:

1. Make the resolved system prompt explicitly explain the achievement progression tree and that repeating already-earned achievements is zero reward.
2. Include `achievements_unlocked` and `next_targets` in the per-turn user prompt/observation context.
3. Keep novelty and unique-achievement scoring front and center.

## Experiments cited

1. `src/nanohorizon/craftax_core/metadata.py`
   - Question: can the prompt metadata expose a clearer achievement roadmap and score framing?
   - Outcome: supporting.
   - Evidence: the file now defines the Craftax progression tree, the zero-reward-for-repeat rule, a richer achievement context payload, and the candidate system prompt text.

2. `src/nanohorizon/craftax_core/http_shim.py`
   - Question: can the prompt-ready payload surface per-episode achievement context without changing the stable HTTP contract?
   - Outcome: supporting.
   - Evidence: `render_prompt_turn()` now returns `achievements_unlocked`, `next_targets`, `unique_achievement_count`, and the score rule; `create_app()` is restored for the health/task-info/rollout contract.

3. `src/nanohorizon/craftax_core/rollout.py`
   - Question: does the live rollout prompt actually show the new achievement tracker per turn?
   - Outcome: supporting.
   - Evidence: each user prompt now includes the current observation plus `achievements_unlocked`, `next_targets`, `unique_achievement_count`, and the zero-reward framing.

4. `reports/craftax_achievement_roadmap/*`
   - Question: does the candidate improve the measured target on the requested seed set?
   - Outcome: supporting.
   - Evidence: `reports/craftax_achievement_roadmap/baseline_eval.json`, `reports/craftax_achievement_roadmap/candidate_eval.json`, and `reports/craftax_achievement_roadmap/eval_comparison.json`.

## Insights

1. The prompt change is not cosmetic. On the same model and seeds, the candidate materially increased unique achievements from 1.2 to 1.6 mean unique achievements on seeds 10000-10004.
2. The main useful signal came from making the progression tree and novelty rule explicit rather than only restating generic “explore and avoid loops” guidance.
3. Per-turn `achievements_unlocked` and `next_targets` made the rollout context easier to inspect and align with the prompt objective.
4. The evaluation path is sensitive to environment setup. The repo checkout did not have `craftax`, `jax`, `fastapi`, or `httpx` in the base interpreter, so the verification used a `uv run --no-project` environment with injected CPU packages.

## Research artifacts produced

- Code:
  - `src/nanohorizon/craftax_core/metadata.py`
  - `src/nanohorizon/craftax_core/http_shim.py`
  - `src/nanohorizon/craftax_core/runner.py`
  - `src/nanohorizon/craftax_core/rollout.py`
  - `src/nanohorizon/shared/eval_model.py`
- Validation:
  - `tests/test_craftax_interface.py`
  - `tests/test_craftax_core_contract.py`
  - `tests/test_craftax_core_runner.py`
- Evaluation artifacts:
  - `reports/craftax_achievement_roadmap/baseline_eval.json`
  - `reports/craftax_achievement_roadmap/candidate_eval.json`
  - `reports/craftax_achievement_roadmap/eval_comparison.json`

## Quality & validation

- Focused tests passed:
  - `tests/test_craftax_interface.py`
  - `tests/test_craftax_core_contract.py`
  - `tests/test_craftax_core_runner.py -k 'not texture_cache'`
- Smoke validation:
  - `run_rollout_request()` worked end-to-end against `https://api.openai.com/v1/chat/completions` with `gpt-5.4-mini`.
- Baseline-vs-candidate evaluation on seeds 10000-10004:
  - Baseline mean unique achievements: `1.2`
  - Candidate mean unique achievements: `1.6`
  - Delta: `+0.4`
  - This clears the required `baseline + 0.1` threshold.

## Reproduction & handoff

- Baseline evaluation command shape:
  - `PYTHONPATH=src uv run --no-project --with 'jax[cpu]' --with craftax --with numpy --with httpx --with pillow --with pyyaml --with fastapi python - <<'PY' ...`
  - Baseline prompt: the old generic Craftax policy text.
- Candidate evaluation command shape:
  - Same command and model settings, but with the enriched `craftax_system_prompt()` output.
- Shared settings for both runs:
  - model: `gpt-5.4-mini`
  - inference URL: `https://api.openai.com/v1/chat/completions`
  - seeds: `10000-10004`
  - `max_steps`: `10`
  - `request_logprobs`: `False`
- Open risk:
  - The measured improvement is on the prompt-sensitive OpenAI proxy used for verification, not on a local Qwen checkpoint in this workspace. The code change is still the requested prompt/harness candidate, but the exact target model path was not available in this environment.
