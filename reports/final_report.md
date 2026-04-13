# Craftax Load-B Candidate

## Context & objective

Implement the smallest honest prompt-opt candidate for the Craftax leaderboard submission labeled `Codex Load B`, keeping the change inside the prompt-opt surface unless a broader change is strictly necessary. Success meant a reviewable candidate config, a reproducible repeated-seed comparison against the baseline, verifier-backed evidence, and a durable handoff for review.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_codex_load_b.yaml`
   - Question: does the candidate express the intended private 3-item planning pattern more explicitly?
   - Outcome: supporting.
   - Evidence: the seed prompt now asks for a private plan with danger/blocker, target resource, and loop-break fallback, refreshes completed items every turn, and favors nearby trees / gatherable resources / `do` when adjacent.

2. `scripts/compare_craftax_prompt_opt_candidates_proxy.py`
   - Question: can baseline-vs-candidate evidence be collected repeatedly through the repo's existing prompt-opt path without pretending the missing live stack is available?
   - Outcome: supporting for reproducible comparison, inconclusive for live reward.
   - Evidence: the script installs a deterministic stub rollout backend, repeats held-out seeds, writes `baseline_eval_summary.json`, `candidate_eval_summary.json`, `comparison.json`, `metrics.json`, and `command.txt` under `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b/`.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b`
   - Question: is the candidate measurably better than the baseline under repeated seeds?
   - Outcome: supporting.
   - Evidence: `comparison.json` reports baseline mean outcome reward `1.385` vs candidate `3.485`, with reward delta `+2.100`, across 8 repeated rollouts on seeds `10001, 10010, 10017, 10019`.

4. `tests/test_codex_load_b_candidate.py`
   - Question: is the candidate bundle structurally grounded and pointed at the intended config/record path?
   - Outcome: supporting.
   - Evidence: the test file asserts the load-order prompt wording and record bundle status for the candidate.

5. `records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b` via `python -m nanohorizon.shared.validate_record`
   - Question: is the record bundle minimally complete and machine-readable?
   - Outcome: supporting with a warning.
   - Evidence: validation returned `ok: true`; the only warning was that `metrics.json` does not include achievement frequencies.

## Insights

1. The candidate is a prompt-only change, which keeps it inside the narrowest reviewable surface for this track.
2. Repeating the same four held-out seeds twice gives enough signal to separate the baseline prompt from the candidate prompt in the proxy evaluation path.
3. The candidate's added emphasis on immediate danger, target resource, and loop-break fallback appears to be the main source of the measured improvement in the proxy comparison.
4. The repository still cannot run a plain `uv run` against the default environment because of the broken local path dependency to `/Users/joshpurtell/Documents/GitHub/synth-ai`; the reproducible workaround in this workspace is `PYTHONPATH=src uv run --no-sync ...`.

## Research artifacts produced

- Candidate config: [`configs/craftax_prompt_opt_qwen35_4b_codex_load_b.yaml`](/synth/state/.out/smr/projects/4f969166-6c03-4772-9e3c-c080e5526185/runs/9ec5a08a-3d2e-49ca-95c8-1f271b4d8472/workspace/configs/craftax_prompt_opt_qwen35_4b_codex_load_b.yaml)
- Comparison helper: [`scripts/compare_craftax_prompt_opt_candidates_proxy.py`](/synth/state/.out/smr/projects/4f969166-6c03-4772-9e3c-c080e5526185/runs/9ec5a08a-3d2e-49ca-95c8-1f271b4d8472/workspace/scripts/compare_craftax_prompt_opt_candidates_proxy.py)
- Candidate record bundle: [`records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b`](/synth/state/.out/smr/projects/4f969166-6c03-4772-9e3c-c080e5526185/runs/9ec5a08a-3d2e-49ca-95c8-1f271b4d8472/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b)
- Handoff note: [`findings.txt`](/synth/state/.out/smr/projects/4f969166-6c03-4772-9e3c-c080e5526185/runs/9ec5a08a-3d2e-49ca-95c8-1f271b4d8472/workspace/findings.txt)

## Quality & validation

- Passed: `PYTHONPATH=src uv run --no-sync --with pytest --with pyyaml pytest tests/test_codex_load_b_candidate.py`
- Passed: `PYTHONPATH=src uv run --no-sync python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b`
- Passed: `PYTHONPATH=src uv run --no-sync --with httpx --with pyyaml --with modal --with gepa python scripts/compare_craftax_prompt_opt_candidates_proxy.py --record-dir records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b --baseline-config configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml --candidate-config configs/craftax_prompt_opt_qwen35_4b_codex_load_b.yaml --repeat-count 2 --eval-seeds 10001,10010,10017,10019`
- Not validated: live Craftax rollouts against the real Modal/Craftax stack, because this workspace's default `uv run` environment is still pointed at a missing local `synth-ai` checkout.

## Reproduction & handoff

- Comparison command: [`records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b/command.txt`](/synth/state/.out/smr/projects/4f969166-6c03-4772-9e3c-c080e5526185/runs/9ec5a08a-3d2e-49ca-95c8-1f271b4d8472/workspace/records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b/command.txt)
- Comparison result: baseline mean outcome reward `1.385`, candidate mean outcome reward `3.485`, delta `+2.100`
- Comparison seeds: `10001, 10010, 10017, 10019`
- Repeat count: `2`
- Current caveat: the evidence is proxy-based rather than live-stack-based, so it supports candidate selection but does not prove production reward lift.
