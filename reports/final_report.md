# Craftax Compact Follow-First Candidate

## Context & objective

The task was to make the smallest honest Craftax improvement in the NanoHorizon repo without touching the protected shared harness surfaces unless absolutely necessary. I chose a prompt-opt change: keep the private todo contract compact and action-directed, then benchmark the baseline prompt against the new compact follow-first candidate using the repo's direct rollout/eval path.

## Experiments cited

1. [`src/nanohorizon/baselines/prompt_opt.py`](/workspace/src/nanohorizon/baselines/prompt_opt.py)
   - Question: can GEPA reflection preserve a stable Craftax todo contract without inflating it into a generic planning rubric?
   - Outcome: supporting.
   - Evidence: `TODO_SCRATCHPAD_REQUIREMENTS`, `REFLECTION_PROMPT_TEMPLATE`, `build_reflection_system_directive`, and `_feedback_for_rollout` now all carry the compact, action-directed wording.

2. [`configs/craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml`](/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml)
   - Question: does a compact follow-first-item prompt keep the same model and rollout shape while removing the extra end-position clause?
   - Outcome: supporting.
   - Evidence: the candidate uses the same model, budget, and seed split as the reference prompt-opt setup, but the action guidance now ends at "follows the first todo item".

3. [`records/prompt_opt_1usd_gpt54_family/2026-04-12_codex_compact_follow_first/`](/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-12_codex_compact_follow_first/)
   - Question: is the candidate packaged reproducibly even before a live run?
   - Outcome: supporting for packaging, inconclusive for live reward.
   - Evidence: `command.txt`, `metadata.json`, `metrics.json`, `notes.md`, `run_config.yaml`, and `system_info.json` are present and validator-clean.

4. [`experiments/craftax_prompt_opt_compact_follow_first/`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/)
   - Question: does the compact candidate beat the baseline on a small repeated-seed benchmark slice?
   - Outcome: supporting on the proxy benchmark, not a live Craftax score.
   - Evidence: [`results/proxy_compare.json`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/results/proxy_compare.json) records 8 repeated eval-seed rollouts for baseline and candidate.

## Insights

1. The useful part of the todo strategy is the stable three-item private contract, not extra planning prose. The compact wording in [`prompt_opt.py`](/workspace/src/nanohorizon/baselines/prompt_opt.py) keeps that contract intact while reducing overconstraint.
2. The new candidate is narrower than the prior todo-refresh variant because it drops the "end next to a useful target for the next turn" requirement and keeps only the follow-first-item instruction.
3. On the proxy benchmark, the compact candidate is materially better than the baseline: baseline mean outcome reward was `0.0625`, candidate mean outcome reward was `1.0`, for a delta of `+0.9375`.
4. This run does not establish live leaderboard improvement. The benchmark is a local repeated-seed proxy built on the repo's direct rollout/eval path, so it is useful for directional comparison but not a substitute for the real Craftax harness.

## Research artifacts produced

- Environments:
  - Direct rollout/eval proxy executed from [`experiments/craftax_prompt_opt_compact_follow_first/benchmark_proxy.py`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/benchmark_proxy.py) via `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with numpy --with pillow --with modal --with gepa ...`
  - The benchmark used 8 repeated eval-seed rollouts per prompt.
- Data:
  - Baseline prompt config: [`configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`](/workspace/configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml)
  - Candidate prompt config: [`configs/craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml`](/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml)
  - Result artifact: [`experiments/craftax_prompt_opt_compact_follow_first/results/proxy_compare.json`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/results/proxy_compare.json)
- Models / checkpoints:
  - No weights were trained or promoted in this run.
  - The candidate stays on `Qwen/Qwen3.5-4B` with the existing prompt-opt budget.

## Quality & validation

- Executed:
  - `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with numpy --with pillow --with modal --with gepa pytest tests/test_codex_compact_follow_first_candidate.py tests/test_codex_todo_refresh_gate_candidate.py tests/test_codex_durable_intent_candidate.py -q`
  - `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with numpy --with pillow --with modal --with gepa python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-12_codex_compact_follow_first`
  - `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with numpy --with pillow --with modal --with gepa python experiments/craftax_prompt_opt_compact_follow_first/benchmark_proxy.py`
- Results:
  - 9 prompt-opt tests passed.
  - Record validation returned `ok: true`.
  - Proxy benchmark summary: baseline mean outcome reward `0.0625`, candidate mean outcome reward `1.0`, delta `+0.9375`.
- Explicitly not validated:
  - live Craftax reward
  - Modal runtime execution
  - GEPA search output on the real model

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Benchmark entrypoint: `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with numpy --with pillow --with modal --with gepa python experiments/craftax_prompt_opt_compact_follow_first/benchmark_proxy.py`
- Durable evidence:
  - [`findings.txt`](/workspace/findings.txt)
  - [`experiments/craftax_prompt_opt_compact_follow_first/experiment_log.txt`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/experiment_log.txt)
  - [`experiments/craftax_prompt_opt_compact_follow_first/results/proxy_compare.json`](/workspace/experiments/craftax_prompt_opt_compact_follow_first/results/proxy_compare.json)
- Open risk:
  - The compact prompt may still be too sparse for the live Craftax harness, and the benchmark here is only a proxy.
