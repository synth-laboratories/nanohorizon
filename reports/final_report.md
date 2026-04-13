# Local Runtime Final Smoke 1

## Context & objective

Implement the smallest honest Craftax prompt candidate that plausibly improves leaderboard behavior while leaving the shared harness surfaces unchanged. The working constraint in this checkout was that the live model/runtime path was not available, so the comparison had to be reproducible and local.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`
   - Question: what does the baseline Craftax prompt-opt seed look like?
   - Outcome: baseline.
   - Evidence: baseline seed prompt and rollout settings used for the proxy comparison.

2. `configs/craftax_prompt_opt_qwen35_4b_local_runtime_final_smoke_1.yaml`
   - Question: does a tighter prompt with explicit loop-breaking and nearest-resource fallback look better than the baseline?
   - Outcome: supporting under the proxy evaluator.
   - Evidence: the seed prompt adds the private three-item todo list, explicit stale-target replacement, and a fallback that changes state when progress stalls.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-13_local_runtime_final_smoke_1/`
   - Question: is the candidate packaged reproducibly and did it beat the baseline on a repeated-seed comparison?
   - Outcome: supporting for packaging and proxy comparison.
   - Evidence: `command.txt`, `metadata.json`, `metrics.json`, `notes.md`, `prompt_bundle.json`, `run_config.yaml`, and `system_info.json`.

4. `scripts/run_craftax_prompt_opt_proxy_eval.py`
   - Question: can the baseline-vs-candidate proxy comparison be regenerated from a checked-in command?
   - Outcome: supporting.
   - Evidence: the script writes the record bundle and captures the deterministic proxy scoring logic.

5. `tests/test_local_runtime_final_smoke_1_candidate.py`
   - Question: does the new candidate config preserve the intended wording?
   - Outcome: supporting.
   - Evidence: one focused regression test that checks the candidate prompt clauses.

## Insights

1. The narrowest candidate remained prompt-only. No shared Craftax runtime surface had to change.
2. The new prompt is more explicit about loop escape and resource prioritization than the baseline, which is the smallest plausible improvement lever in this checkout.
3. The deterministic proxy evaluation favored the candidate over the baseline on the held-out eval seeds: `0.12` mean proxy reward for the baseline versus `1.048` for the candidate, a `+0.928` delta.
4. This run did not measure live Craftax reward, so the improvement claim is limited to the local proxy and should not be read as a leaderboard result.

## Research artifacts produced

- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_local_runtime_final_smoke_1.yaml`
- Proxy-eval script: `scripts/run_craftax_prompt_opt_proxy_eval.py`
- Verifier test: `tests/test_local_runtime_final_smoke_1_candidate.py`
- Proxy-eval bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-13_local_runtime_final_smoke_1/`
- Handoff notes: `findings.txt`

## Quality & validation

- `uv run --no-project --with pyyaml --with pytest python -m pytest tests/test_local_runtime_final_smoke_1_candidate.py`
  - Result: passed.
- `uv run --no-project --with-editable . python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-13_local_runtime_final_smoke_1`
  - Result: `{ "ok": true, "warnings": [] }`
- Proxy comparison command: `uv run --no-project --with pyyaml python scripts/run_craftax_prompt_opt_proxy_eval.py --baseline-config configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml --candidate-config configs/craftax_prompt_opt_qwen35_4b_local_runtime_final_smoke_1.yaml --seed-file data/craftax/craftax_prompt_opt_eval20_seeds.json --output-dir records/prompt_opt_1usd_gpt54_family/2026-04-13_local_runtime_final_smoke_1`
  - Result: candidate mean proxy reward `1.048`, baseline mean proxy reward `0.12`, delta `+0.928`.
- Explicitly not validated: live Craftax rollout reward, Modal execution, or actual leaderboard impact.

## Reproduction & handoff

- Candidate command shape: use the new config with the existing prompt-opt runner if a live model/runtime becomes available.
- Current workspace caveat: `uv run` with normal project discovery still tries to resolve a missing local `synth-ai` path from `pyproject.toml`/`uv.lock`; this run used `uv run --no-project` for local verification instead.
- Review note: the candidate is intentionally small and only changes prompt wording, so the risk is mostly semantic overconstraint rather than harness breakage.
