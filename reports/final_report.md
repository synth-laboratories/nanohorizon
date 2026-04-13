# Craftax Todo Refresh Gate Candidate

## Context & objective

Implement the smallest honest Craftax candidate for the todo-tool idea without changing the protected shared harness surfaces. The goal was to keep the prompt-opt reflection path aligned with the same private three-item scratchpad contract used by the candidate prompt, and to gather at least one repeat-seed baseline-vs-candidate comparison with durable evidence.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: is a narrow prompt-only intervention safer than a harness change?
   - Outcome: supporting.
   - Evidence: the checked-in prompt-opt baseline documents the prior reward regression and establishes the reference score for this track.

2. `src/nanohorizon/baselines/prompt_opt.py`
   - Question: does prompt optimization preserve a stable todo-tool contract during GEPA reflection?
   - Outcome: supporting.
   - Evidence: the source centralizes the private three-item scratchpad requirements in `TODO_SCRATCHPAD_REQUIREMENTS` and reuses them in reflection instructions and rollout feedback.

3. `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
   - Question: does the candidate add a compact but stricter loop-break / action-gating variant?
   - Outcome: supporting.
   - Evidence: the prompt refreshes todo items every turn, replaces stale targets after no-progress loops, and asks the short action batch to follow the current first todo item.

4. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_local_compare`
   - Question: on a local verifier surrogate with repeated eval seeds, does the candidate outperform the baseline?
   - Outcome: supporting for prompt-shape uplift, inconclusive for real Craftax reward.
   - Evidence: `comparison_summary.json`, `baseline_summary.json`, `candidate_summary.json`, `baseline_rollouts.jsonl`, `candidate_rollouts.jsonl`, and `notes.md`.

## Insights

1. The narrowest honest improvement here remains prompt and reflection shaping, not a harness edit.
2. Preserving one exact private three-item contract across seed prompt, GEPA reflection, and rollout feedback is the useful part of the todo strategy.
3. The candidate prompt's extra loop-break and first-todo guidance is strong enough to win on the local verifier surrogate across repeated seeds.
4. Live Craftax reward remains unmeasured in this workspace because the Modal path was not authenticated, so the surrogate result should not be mistaken for leaderboard evidence.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/`
- Local comparison bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_local_compare/`
- Local comparison script: `scripts/compare_craftax_prompt_opt_local.py`
- Structural regression test: `tests/test_codex_todo_refresh_gate_candidate.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with numpy --with httpx --python 3.11 python -m pytest tests/test_codex_todo_refresh_gate_candidate.py tests/test_craftax_interface.py`
- Result: `6 passed`
- Executed: `PYTHONPATH=src uv run --no-project --with modal --with gepa --with httpx --with pyyaml --with numpy --python 3.11 python scripts/compare_craftax_prompt_opt_local.py --output-dir records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_local_compare --repeats 2`
- Result: baseline mean outcome reward `0.0`, candidate mean outcome reward `1.5`, baseline mean search score `0.0234375`, candidate mean search score `1.7625`
- Verifier feedback: live Modal Craftax feedback was unavailable in this workspace, so the comparison was recorded with a local surrogate verifier and explicitly labeled as such.

## Reproduction & handoff

- Candidate entrypoint: `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Local comparison entrypoint: `PYTHONPATH=src uv run --no-project --with modal --with gepa --with httpx --with pyyaml --with numpy --python 3.11 python scripts/compare_craftax_prompt_opt_local.py --output-dir records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_local_compare --repeats 2`
- Main risk: the stronger "follow the first todo item" wording could overconstrain otherwise good short tactical action batches.
- Residual risk: the surrogate comparison is not a substitute for a live Craftax rollout, so the true leaderboard impact remains unknown until the Modal path is available.
- Recommended verifier focus:
  - confirm the centralized todo contract remains present in reflection instructions
  - inspect whether the follow-the-first-item wording is compact enough to avoid overlong reasoning
  - run the candidate config against the reference baseline in the live Craftax path when the runtime is available
