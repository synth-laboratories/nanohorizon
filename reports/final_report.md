# Craftax Recent-Turn History Candidate

## Context & objective

The objective for this run was a narrow NanoHorizon Craftax harness improvement that could plausibly help leaderboard performance without changing the protected interface surfaces. I chose a prompt-shaping change in the rollout path: include a compact recent-turn history block in each policy prompt so later decisions can react to the last few actions and rewards.

Success for this run meant:

1. a concrete code change in the NanoHorizon repo
2. a baseline-vs-candidate eval slice with repeated seeds or rollouts
3. verifier-driven review through targeted tests
4. a reviewable commit and PR-ready state

## Experiments cited

1. `src/nanohorizon/craftax_core/rollout.py`
   - Question: does adding a recent-turn history block to the prompt change the rollout contract in the intended way?
   - Outcome: supporting.
   - Evidence: the rollout now appends a bounded recent-turn summary before each model call.

2. `tests/test_craftax_core_runner.py::test_rollout_includes_recent_turn_history_in_later_prompts`
   - Question: does the second rollout prompt actually contain the new history block?
   - Outcome: supporting.
   - Evidence: the test passed and asserts the first prompt lacks the block while the second prompt includes it.

3. `experiments/craftax_recent_turn_history_prompt/scripts/compare_recent_turn_history_eval.py`
   - Question: does the prompt change improve a repeated-seed rollout slice through the existing rollout path?
   - Outcome: supporting as a proxy eval.
   - Evidence: deterministic fake policy/env comparison run over four repeated seeds.

4. `experiments/craftax_recent_turn_history_prompt/results/comparison.json`
   - Question: what is the measured baseline-vs-candidate delta?
   - Outcome: supporting for the proxy eval.
   - Evidence: baseline mean outcome reward `1.0`; candidate mean outcome reward `2.0`; delta `+1.0`. Baseline mean native env reward total `4.0`; candidate `5.0`; delta `+1.0`.

## Insights

1. Carrying recent action/reward context into later prompts is a real behavioral change, not a cosmetic rewrite. The rollout path now gives the policy a compact memory of the last few turns.
2. The prompt change is verifiable locally with a focused regression test that inspects the second call directly.
3. The proxy eval shows the intended effect: when the policy can see recent-turn history, it switches to the higher-value second-turn action and scores higher on repeated seeds.
4. I did not validate a real leaderboard model or a live Craftax submission here. The eval slice is a deterministic proxy that exercises the repo's existing rollout path.

## Research artifacts produced

- Code change: `src/nanohorizon/craftax_core/rollout.py`
- Regression test: `tests/test_craftax_core_runner.py`
- Eval script: `experiments/craftax_recent_turn_history_prompt/scripts/compare_recent_turn_history_eval.py`
- Eval output: `experiments/craftax_recent_turn_history_prompt/results/comparison.json`
- Experiment log: `experiments/craftax_recent_turn_history_prompt/experiment_log.txt`
- Durable repo notes: `findings.txt`

## Quality & validation

- Passed: `PYTHONPATH=src uv run --no-project --with pytest --with httpx --with fastapi --with numpy --with pillow --with pyyaml python -m pytest tests/test_craftax_interface.py tests/test_craftax_core_runner.py::test_rollout_includes_recent_turn_history_in_later_prompts`
- Passed: `PYTHONPATH=src uv run --no-project --with httpx --with numpy --with pyyaml python experiments/craftax_recent_turn_history_prompt/scripts/compare_recent_turn_history_eval.py --output-dir experiments/craftax_recent_turn_history_prompt/results --seeds 10000 10001 10002 10003`
- Known gap: the broader `tests/test_craftax_core_contract.py` still imports a missing `create_app` symbol from `nanohorizon.craftax_core.http_shim` in this checkout, so I avoided using that file as the verifier surface.
- Known environment caveat: `uv run` sync mode hit a stale direct file dependency on `/Users/joshpurtell/Documents/GitHub/synth-ai`; verification used `--no-project` with explicit package layers instead.

## Reproduction & handoff

- Candidate behavior entrypoint: `src/nanohorizon/craftax_core/rollout.py`
- Reproduction command for the proxy slice:

```bash
PYTHONPATH=src uv run --no-project --with httpx --with numpy --with pyyaml \
  python experiments/craftax_recent_turn_history_prompt/scripts/compare_recent_turn_history_eval.py \
  --output-dir experiments/craftax_recent_turn_history_prompt/results \
  --seeds 10000 10001 10002 10003
```

- Rough result: candidate proxy mean outcome reward improved from `1.0` to `2.0` across four repeated seeds.
- Open risk: this is not a real leaderboard measurement. A live model-backed eval slice would still be needed before claiming true leaderboard uplift.
