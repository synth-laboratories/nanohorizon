# Craftax Decision Brief Candidate

## Context & objective

This run implemented a meaningfully different Craftax prompt policy for the NanoHorizon prompt-opt track by adding a deterministic decision brief and recent-history context to the rollout prompt. The objective was to keep the shared Craftax harness surfaces stable unless truly required, then validate the candidate against a baseline on repeated seeds and record the result honestly.

## Experiments cited

1. [Proxy eval bundle](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/experiments/craftax_decision_brief_proxy/results/proxy_eval.json)
   - Question: does the decision-brief candidate outperform the baseline prompt on a repeated-seed eval slice?
   - Outcome: supporting on the deterministic proxy slice.
   - Evidence: baseline mean outcome reward `0.0`, candidate mean outcome reward `1.0`, uplift `+1.0` on seeds `[10001, 10010, 10017, 10019] x2`.

2. [Targeted Craftax tests](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/tests/test_craftax_interface.py)
   - Question: does the decision brief produce the intended structured prompt fields?
   - Outcome: supporting.
   - Evidence: tests now assert `decision_brief.mode`, `priority_targets`, `loop_risk`, and the updated structured prompt payload.

3. [Rollout contract tests](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/tests/test_craftax_core_contract.py)
   - Question: does the real `run_rollout` path include the new decision-context JSON and preserve the existing HTTP shim contract?
   - Outcome: supporting.
   - Evidence: `create_app` works again, `/health` and `/task_info` respond, and the rollout prompt now carries a compact decision context block.

4. [Candidate config](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml)
   - Question: is the new prompt contract actually packaged as a reviewable candidate?
   - Outcome: supporting.
   - Evidence: the seed prompt explicitly requires reading the compact decision brief and recent reward-history window before choosing actions.

## Insights

1. The new policy is behaviorally different, not just a wording tweak. It adds a decision brief with `mode`, `primary_focus`, `fallback_action`, `loop_risk`, and `priority_targets`, then threads that into the rollout prompt.
2. The rollout path now carries more than the raw observation string. It includes a bounded recent-history window plus the decision brief JSON, which makes the candidate easier to steer away from repeated loops.
3. The repo had a pre-existing `create_app` gap in `http_shim`; restoring that compatibility surface was necessary to keep the existing Craftax contract tests runnable.
4. The only reward uplift observed in this run is the deterministic proxy slice. I did not reach a live Qwen rollout path in this workspace, so no production score claim is supported here.

## Research artifacts produced

### Environments

- Proxy evaluation script: [experiments/craftax_decision_brief_proxy/scripts/run_proxy_eval.py](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/experiments/craftax_decision_brief_proxy/scripts/run_proxy_eval.py)
- Proxy eval result: [experiments/craftax_decision_brief_proxy/results/proxy_eval.json](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/experiments/craftax_decision_brief_proxy/results/proxy_eval.json)
- The proxy ran with `PYTHONPATH=src uv run --no-project --with fastapi --with httpx --with pyyaml --with numpy ...` because the project resolver could not reach the workspace’s pinned training dependency path during the live sync path.

### Data

- Seeds used for the proxy slice: `[10001, 10010, 10017, 10019] x2`
- Baseline config: [configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml)
- Candidate config: [configs/craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml)

### Models / checkpoints

- No weights or checkpoints were trained in this run.
- The candidate is a prompt-policy configuration for `Qwen/Qwen3.5-4B`, not a model update.

## Quality & validation

- Passed: `PYTHONPATH=src uv run --no-project --with fastapi --with httpx --with pyyaml --with numpy python -m pytest tests/test_craftax_interface.py tests/test_craftax_core_contract.py tests/test_codex_decision_brief_candidate.py`
- Passed: the proxy eval bundle reports baseline `0.0` vs candidate `1.0` mean outcome reward on the repeated-seed slice.
- Not validated: live Craftax rollout against a real Qwen endpoint, Modal-backed policy serving, or any leaderboard score.

## Reproduction & handoff

- Exact proxy command: [experiments/craftax_decision_brief_proxy/command.txt](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/experiments/craftax_decision_brief_proxy/command.txt)
- Candidate record bundle: [records/prompt_opt_1usd_gpt54_family/2026-04-12_codex_decision_brief/](/synth/state/.out/smr/projects/a080a317-7b4e-4a88-b099-9215a30c9957/runs/9f36356d-015a-403f-818c-0e6985dedeed/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-12_codex_decision_brief/)
- Main caveat: the uplift is proxy-verified only. The live model route was not reachable in this workspace, so the result should be treated as a structured code-and-policy improvement with a deterministic evaluation slice, not as a production score claim.
