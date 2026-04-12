# Craftax Achievement Roadmap v2

## Context & objective

Implement the smallest honest Craftax candidate that improves mean unique achievements by shaping the policy around an explicit achievement roadmap, while preserving the shared rollout contract unless a minimal harness fix is required.

Success for this run meant:

- candidate prompt changes landed in the repo with minimal reviewable edits
- baseline and candidate were evaluated on seeds `10000-10004`
- the run recorded exact commands and mean unique achievements
- a reviewable commit and PR were prepared

## Experiments cited

1. `src/nanohorizon/shared/eval_model.py`
   - Question: does the default Craftax eval prompt now include an explicit achievement progression tree and zero-reward-for-repeat framing?
   - Outcome: supporting.
   - Evidence: `_default_craftax_system_prompt()` now emits the roadmap text and repeat-framing language.

2. `src/nanohorizon/craftax_core/rollout.py`
   - Question: does each user prompt carry current progression state and next targets?
   - Outcome: supporting.
   - Evidence: `run_rollout()` now initializes `unique_achievements` from runner state, derives `next_targets`, and includes `achievements_unlocked` plus `next_targets` in each prompt and turn trace.

3. `src/nanohorizon/craftax_core/http_shim.py`
   - Question: can the Craftax shim actually be launched as `python -m nanohorizon.craftax_core.http_shim`?
   - Outcome: supporting.
   - Evidence: added `main()` and `uvicorn.run(...)` so the tunnel helper can start a live server.

4. `tests/test_craftax_core_contract.py`
   - Question: are the new prompt helpers and eval prompt text covered by regression tests?
   - Outcome: supporting.
   - Evidence: tests assert the achievement roadmap text, the next-target helper, and prompt fields.

5. `.out/baseline_proxy_eval_9016/eval_summary.json`
   - Question: what is the baseline mean unique achievements on seeds `10000-10004` under the closest honest equivalent available in this container?
   - Outcome: blocked.
   - Evidence: `mean_outcome_reward` is `0.0`, `num_eval_rollouts` is `0`, `num_rollout_errors` is `5`, with rollout errors `Server disconnected without sending a response` / `All connection attempts failed`.

6. `.out/candidate_proxy_eval_9017/eval_summary.json`
   - Question: does the candidate beat the baseline on the same seed set?
   - Outcome: blocked / not improved.
   - Evidence: `mean_outcome_reward` is `0.0`, `num_eval_rollouts` is `0`, `num_rollout_errors` is `5`, with the same failure mode as baseline.

## Insights

1. The candidate prompt change is narrowly localized and reviewable: the eval prompt, rollout prompt, and shim entrypoint were the only behavioral changes.
2. The user prompt is now stateful in the right place: progression context is derived from the live episode state and carried forward as `achievements_unlocked` and `next_targets`.
3. Verification is still blocked by the runtime substrate, not by the prompt logic. In this container, the proxy eval path never completed a rollout, so no meaningful reward delta was observable.
4. Because both baseline and candidate stayed at `0.0` mean unique achievements, the candidate does not meet the `+0.1` requirement and is not ready.

## Research artifacts produced

- Prompt shaping: `src/nanohorizon/shared/eval_model.py`
- Rollout shaping: `src/nanohorizon/craftax_core/rollout.py`
- Shim entrypoint: `src/nanohorizon/craftax_core/http_shim.py`
- Regression tests: `tests/test_craftax_core_contract.py`
- Baseline eval summary: `.out/baseline_proxy_eval_9016/eval_summary.json`
- Candidate eval summary: `.out/candidate_proxy_eval_9017/eval_summary.json`
- Server logs: `.out/http_shim_9016.log`, `.out/http_shim_9017.log`
- Handoff notes: `findings.txt`, `experiment_log.txt`

## Quality & validation

- Unit tests passed: `uv run pytest tests/test_craftax_core_contract.py tests/test_craftax_interface.py`
- Eval path used:
  - start local Craftax server with `uv run --no-sync python -m nanohorizon.craftax_core.http_shim`
  - call `evaluate_model(...)` through `uv run --no-sync python -c ...`
- Proxy eval settings:
  - seeds `10000-10004`
  - `max_steps=4`
  - `max_concurrent_rollouts=1`
  - `request_timeout_seconds=300`
  - `request_model='gpt-4.1-nano'`
  - local Craftax server on ports `9016` and `9017`
- Observed failure modes:
  - early runs hit a corrupted Craftax texture cache and a missing `craftax` import
  - the cache had to be cleared and the server/eval commands had to use `uv run --no-sync`
  - the final baseline and candidate runs still failed before any successful rollout, so the measured mean unique achievements remained `0.0`

## Reproduction & handoff

- Exact baseline command shape:
  - clear the Craftax texture cache file in `.venv/lib/python3.11/site-packages/craftax/craftax/assets/texture_cache.pbz2`
  - start `uv run --no-sync python -m nanohorizon.craftax_core.http_shim` on `127.0.0.1:9016`
  - run `evaluate_model(...)` with the baseline prompt override and seeds `10000-10004`
- Exact candidate command shape:
  - same server/eval flow on `127.0.0.1:9017`
  - omit the prompt override so the new default prompt is used
- Repository state:
  - candidate changes are implemented
  - verification is blocked, so the candidate is not ready
  - a reviewable commit and PR are still required before completion
