# Craftax Local Runtime Final Smoke 2

## Context & objective

The objective was to keep the Craftax prompt-opt surface small and reviewable while pushing the policy toward earlier resource collection behavior. The candidate keeps the existing tool contract and adds a more explicit wood-first bootstrap ladder inside the system prompt.

## Experiments cited

1. `records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline`
   - Question: what is the checked-in prompt-opt baseline?
   - Outcome: supporting baseline.
   - Evidence: `metrics.json` reports `primary_score: 0.35` and `bootstrap_score: 0.6`.

2. `configs/craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml`
   - Question: does the candidate make early-game resource collection more explicit without changing harness surfaces?
   - Outcome: supporting.
   - Evidence: the prompt now says `collect_wood -> place_table -> make_wood_pickaxe -> collect_stone` and keeps the three-item private todo contract.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/scripts/proxy_smoke.py`
   - Question: can the repo’s existing concurrent rollout path be used for a narrow baseline-vs-candidate smoke?
   - Outcome: supporting as a proxy.
   - Evidence: the script reuses `nanohorizon.shared.craftax_data.collect_rollouts_concurrently_with_summary` with a deterministic responder over the held-out starter eval seeds.

4. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2`
   - Question: is the candidate packaged reproducibly?
   - Outcome: supporting.
   - Evidence: `metadata.json`, `metrics.json`, `prompt_bundle.json`, `run_config.yaml`, `system_info.json`, `command.txt`, and `notes.md` are present and validated.

## Insights

1. The smallest useful change here is still prompt shaping, not harness surgery.
2. Making the resource ladder concrete appears to be worth more than another generic planning reminder.
3. The proxy smoke is directional, but it does distinguish the candidate from the baseline on the same held-out starter seeds.
4. Because the live Qwen/Craftax lane was unavailable in this run, the reward evidence should be treated as a proxy rather than leaderboard-grade validation.

## Research artifacts produced

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml`
- Candidate test: `tests/test_codex_local_runtime_final_smoke_2_candidate.py`
- Candidate record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/`
- Proxy smoke script: `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/scripts/proxy_smoke.py`
- Repo handoff: `findings.txt`

## Quality & validation

- Executed: direct Python assertions for `tests/test_codex_local_runtime_final_smoke_2_candidate.py`
- Result: passed.
- Executed: `PYTHONPATH=src python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2`
- Result: `{ "ok": true, "warnings": [] }`
- Proxy smoke result: baseline mean outcome reward `65.5`, candidate mean outcome reward `89.5`, delta `+24.0` on the held-out starter eval seeds `[10001, 10010, 10017, 10019]`.
- Not validated: live Craftax reward, Modal runtime behavior, or an actual Qwen rollout.
- Noted issue: `uv run` hit a workspace dependency-resolution error, so the smoke was executed with `PYTHONPATH=src python` instead.

## Reproduction & handoff

- Candidate entrypoint: `PYTHONPATH=src python records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/scripts/proxy_smoke.py`
- Candidate command log: `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/command.txt`
- Candidate record bundle contains the prompt, metrics, and proxy smoke notes needed to replay the comparison.
- Remaining risk: the candidate is promising on the proxy smoke, but it still needs a real Qwen/Craftax rollout before anyone should treat it as leaderboard-grade.
