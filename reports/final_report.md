# Craftax Offline Trace Filter Candidate

## Context & objective

Implement the smallest honest NanoHorizon Craftax change that plausibly improves offline SFT quality by filtering training traces to examples with `3+` unique achievements and biasing selection toward rare achievements such as `defeat_zombie` and `eat_cow`, while leaving the shared Craftax harness surfaces stable.

## Experiments cited

1. `src/nanohorizon/baselines/offline_sft.py`
   - Question: can the offline SFT row builder enforce the `3+` unique-achievement gate and rank rare-achievement traces first without touching the Craftax harness?
   - Outcome: supporting.
   - Evidence: the row builder now records `unique_achievement_count` and `priority_achievement_hits`, filters out rows below `min_unique_achievements`, and ranks rows by outcome reward plus rarity-aware metadata.

2. `configs/craftax_offline_reference.yaml`
   - Question: can the offline reference config express the new filter without changing the harness contract?
   - Outcome: supporting.
   - Evidence: the config now sets `teacher_generation.min_unique_achievements: 3` and `teacher_generation.priority_achievements: [defeat_zombie, eat_cow]`.

3. `tests/test_craftax_core_runtime_guarantees.py`
   - Question: does the candidate still preserve the tool-only Craftax row behavior while excluding low-diversity traces and preferring rare-achievement traces?
   - Outcome: supporting.
   - Evidence: the regression covers the original tool-only reasoning path and a synthetic comparison that rejects `2`-achievement rollouts while ranking the rare-achievement rollout ahead of the generic one.

4. `records/offline_20min_1xa100_40gb/2026-04-13_launch_validation/`
   - Question: does a repeated-seed comparison show the candidate changing row selection in the intended direction?
   - Outcome: supporting for the filter objective, inconclusive for live leaderboard impact.
   - Evidence: `comparison_summary.json` shows baseline kept `6` rollouts with mean unique-achievement count `2.6667`, while the candidate kept `4` rollouts with mean unique-achievement count `3.25` and increased mean priority-achievement hits from `0.0` to `1.25`; `verification.json` reports `passed: true`.

5. `data/craftax/cpt_rollouts_text.jsonl`
   - Question: does the current archived rollout corpus already satisfy the new filter?
   - Outcome: negative.
   - Evidence: a census over the file found `0` rollouts with `3+` unique achievements, so the stricter gate is currently useful as a quality filter but would reject this archived corpus entirely.

## Insights

1. The candidate is intentionally small: it changes only offline SFT row selection and config, not the Craftax runtime contract.
2. The `3+` unique-achievement gate is real and measurable. On the synthetic repeated-seed comparison, it reduced kept rows from `6` to `4` while increasing the mean unique-achievement count from `2.6667` to `3.25`.
3. The rarity bias is working in the right direction. The verifier-backed comparison reports higher priority-achievement hits and orders the `defeat_zombie`/`eat_cow` rollout ahead of the generic `3`-achievement rollout.
4. The current archived CPT corpus is too weak for this filter to help by itself. It contains no `3+` unique-achievement traces, so this idea only becomes useful if teacher generation produces richer rollouts.

## Research artifacts produced

- Source change: `src/nanohorizon/baselines/offline_sft.py`
- Offline config: `configs/craftax_offline_reference.yaml`
- Regression test: `tests/test_craftax_core_runtime_guarantees.py`
- Comparison bundle: `records/offline_20min_1xa100_40gb/2026-04-13_launch_validation/`
- Durable notes: `findings.txt`

## Quality & validation

- Executed direct assertion checks with `PYTHONPATH=src python3` because `pytest` was unavailable in system Python and `uv run` hit a missing local-path dependency in this checkout.
- Verified that the new filter keeps only rows with `3+` unique achievements and ranks rare-achievement rows first.
- Verified that the original tool-only Craftax SFT row path still emits the expected reasoning-content/tool-call structure.
- Verified the repeated-seed comparison bundle: baseline `6` kept rollouts versus candidate `4`, with the candidate increasing mean unique-achievement count and priority-achievement hits.
- Explicitly not validated: live Modal training, a full offline training run, or leaderboard score impact.

## Reproduction & handoff

- To inspect the code path: `src/nanohorizon/baselines/offline_sft.py`
- To inspect the reference settings: `configs/craftax_offline_reference.yaml`
- To inspect the verifier bundle: `records/offline_20min_1xa100_40gb/2026-04-13_launch_validation/`
- The current branch still needs the required git push and PR flow before the task can be considered complete.
- Main risk: the archived rollout corpus does not yet contain any `3+` unique-achievement traces, so this filter is only beneficial if future teacher generation produces richer data.
