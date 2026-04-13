# Craftax Single-Step Progress Candidate

## Context & objective

The task was to build the smallest honest Craftax prompt-opt candidate that could plausibly improve NanoHorizon leaderboard behavior without touching the protected shared harness surfaces. The starting point was the checked-in 4-action todo-refresh variant, which had already shown weaker reward than the older 1-action seed prompt in the repo notes.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: what does the repository already consider a stronger prompt-opt baseline?
   - Outcome: supporting.
   - Evidence: the tracked notes record a seed-prompt score of `0.6` and a regressed GEPA result of `0.35`.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_single_step_progress.yaml`
   - Question: does a 1-action prompt with direct resource guidance preserve the policy contract while removing the overconstrained todo scaffold?
   - Outcome: supporting at the packaging level.
   - Evidence: the seed prompt now requests a single full-Craftax action, prioritizes nearby resources, and avoids the private todo-list scaffold.

3. `tests/test_codex_single_step_progress_candidate.py`
   - Question: is the new candidate packaged consistently and does it point at the new record bundle?
   - Outcome: supporting.
   - Evidence: the test checks the candidate prompt text and the not-run record bundle metadata.

4. `experiments/2026-04-13_codex_local_rerun2_2/proxy_eval.py`
   - Question: can the repo rollout path be exercised repeatedly in this workspace even though the live Craftax runtime packages are missing?
   - Outcome: supporting for proxy evaluation only.
   - Evidence: the script runs repeated seeds through `nanohorizon.craftax_core.rollout.run_rollout_request` with a fake runner and local HTTP proxy.

5. `experiments/2026-04-13_codex_local_rerun2_2/results/proxy_baseline_vs_candidate.json`
   - Question: does the candidate outperform the 4-action todo baseline on the available repeated-seed proxy?
   - Outcome: inconclusive for reward, supporting for efficiency.
   - Evidence: baseline and candidate both averaged `2.0` proxy outcome reward, but candidate reduced mean `llm_call_count` from `12.0` to `5.75`.

6. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_single_step_progress`
   - Question: is the candidate recorded as a reproducible package?
   - Outcome: supporting for packaging, not run for live scoring.
   - Evidence: `command.txt`, `metadata.json`, `metrics.json`, `notes.md`, `run_config.yaml`, and `system_info.json`.

## Insights

1. The strongest prompt-opt signal already in the repo is the 1-action seed prompt, not the 4-action todo variant.
2. The new candidate keeps the policy contract simple: one tool call, one action, direct resource progress.
3. In the local proxy comparison, the candidate did not raise outcome reward, but it did reduce model-call count materially.
4. The live Craftax runtime could not be executed in this workspace because `craftax`, `jax`, and `modal` were unavailable, so any leaderboard claim remains unvalidated here.

## Research artifacts produced

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_single_step_progress.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_single_step_progress/`
- Proxy evaluation script: `experiments/2026-04-13_codex_local_rerun2_2/proxy_eval.py`
- Proxy evaluation outputs: `experiments/2026-04-13_codex_local_rerun2_2/results/proxy_baseline_vs_candidate.json`
- Proxy evaluation summary: `experiments/2026-04-13_codex_local_rerun2_2/results/proxy_baseline_vs_candidate.md`
- Experiment log: `experiments/2026-04-13_codex_local_rerun2_2/experiment_log.txt`
- Repo handoff note: `findings.txt`

## Quality & validation

- Executed: `uv run --no-project --no-sync --with pytest --with pyyaml --with httpx --with numpy python -m pytest tests/test_codex_single_step_progress_candidate.py tests/test_codex_todo_refresh_gate_candidate.py`
- Result: 5 tests passed.
- Executed: `uv run --no-project --no-sync python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_single_step_progress`
- Result: `ok: true`
- Executed: `PYTHONPATH=src:. uv run --no-project --no-sync --with pyyaml --with httpx --with numpy python experiments/2026-04-13_codex_local_rerun2_2/proxy_eval.py`
- Result: repeated-seed proxy comparison completed; baseline and candidate tied on proxy outcome reward, candidate lowered mean model-call count.
- Not validated: live Craftax leaderboard reward, Modal execution, or real model rollout behavior.

## Reproduction & handoff

- Candidate reproduce command:

```bash
NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_single_step_progress.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh
```

- Proxy comparison reproduce command:

```bash
PYTHONPATH=src:. uv run --no-project --no-sync --with pyyaml --with httpx --with numpy python experiments/2026-04-13_codex_local_rerun2_2/proxy_eval.py
```

- Main caveat: this run does not include a live Craftax rollout because the runtime packages were absent in the workspace, so the candidate is still unscored against the real leaderboard path.
- Recommended next step: execute the candidate config in an environment with the Craftax runtime stack available, then compare reward against the same held-out seed split.
