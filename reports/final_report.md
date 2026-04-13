# Craftax Prompt-Opt Candidate Report

## Context & Objective
This run targeted the NanoHorizon Craftax prompt-opt track. The goal was to make the smallest honest candidate change that plausibly improves the `Qwen/Qwen3.5-4B` Craftax policy while preserving the shared Craftax harness surfaces.

The candidate is represented by the refined prompt contract in [configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml), the matching source-side contract update in [src/nanohorizon/baselines/prompt_opt.py](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/src/nanohorizon/baselines/prompt_opt.py), and the blocked record bundle at [records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_blocked/](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_blocked/).

## Experiments Cited
1. Proxy baseline-vs-candidate comparison, saved at [experiments/craftax_loop_break/results/baseline_vs_candidate_proxy_max2.json](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/experiments/craftax_loop_break/results/baseline_vs_candidate_proxy_max2.json).
   - Question: does the todo-refresh candidate improve the repo-shaped proxy score on repeated held-out seeds?
   - Outcome: negative.
   - Evidence: baseline mean outcome reward `2.0`, candidate mean outcome reward `2.0`; baseline mean search score `2.078125`, candidate mean search score `1.925`.
2. Wider proxy comparison across three prompt variants, saved at [experiments/craftax_loop_break/results/proxy_compare_three_prompts.json](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/experiments/craftax_loop_break/results/proxy_compare_three_prompts.json).
   - Question: among the baseline, durable-intent, and todo-refresh variants, which prompt looked best on the same proxy?
   - Outcome: negative for the candidate family.
   - Evidence: the baseline ranked ahead of both candidate variants on mean proxy search score.
3. Real Craftax rollout attempt.
   - Question: can the candidate be verified on the actual Craftax rollout path?
   - Outcome: blocked.
   - Evidence: the direct rollout process was killed with exit code `137` after Craftax texture processing began, so no full-environment reward delta could be measured in this workspace.

## Insights
1. The candidate remains isolated to prompt-opt packaging and the shared prompt-contract source, not the protected Craftax harness surfaces.
2. The focused regression coverage is passing and confirms the candidate text still encodes the intended private three-item todo contract.
3. Proxy verification did not show a reward lift for the candidate family, so there is no honest evidence to claim improvement from the in-run comparisons.
4. The true Craftax verifier is currently blocked in this workspace, so any reward claim for this candidate must wait for a fresh environment that can complete the rollout.

## Research Artifacts Produced
### Environments
- Proxy eval used the repo rollout path with `direct://local`, `gpt-4.1-nano`, and the fake Craftax runner from [tests/_craftax_fakes.py](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/tests/_craftax_fakes.py).
- The real rollout attempt used the same repo code path but was killed during Craftax texture warmup.

### Data
- Proxy seeds: `[10001, 10002, 10004, 10005]` from [data/craftax/craftax_prompt_opt_eval20_seeds.json](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/data/craftax/craftax_prompt_opt_eval20_seeds.json).
- No new training data or checkpoint artifacts were produced.

### Models / checkpoints
- Proxy verifier model: `gpt-4.1-nano`.
- Prompt-opt policy family: `Qwen/Qwen3.5-4B`.
- No weights or checkpoints were updated in this run.

## Quality & Validation
- Passed focused regression tests:
  - `uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py`
- Completed proxy comparisons are stored under [experiments/craftax_loop_break/results/](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/experiments/craftax_loop_break/results/).
- Not validated: a full Craftax reward improvement on the live rollout path.

## Reproduction & Handoff
- Candidate config: [configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml)
- Prompt-contract source: [src/nanohorizon/baselines/prompt_opt.py](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/src/nanohorizon/baselines/prompt_opt.py)
- Regression tests: [tests/test_codex_todo_refresh_gate_candidate.py](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/tests/test_codex_todo_refresh_gate_candidate.py) and [tests/test_codex_durable_intent_candidate.py](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/tests/test_codex_durable_intent_candidate.py)
- Blocked-eval bundle: [records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_blocked/](/synth/state/.out/smr/projects/80295569-70a2-43e2-b584-685d086fee69/runs/de99a044-a73c-443f-9bda-cbe7cc4ca005/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_blocked/)
- Residual risk: the candidate is still only supported by structural validation and proxy comparisons; the true Craftax rollout path was not available to prove an actual reward lift.
