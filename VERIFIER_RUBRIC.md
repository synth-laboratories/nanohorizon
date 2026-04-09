# NanoHorizon Go-Explore Prompt Optimization Verifier Rubric

Review the final submission bundle and decide whether the run should count as a
successful open-ended algorithm-research attempt on Go-Explore for Craftax.

The high-level judgment axes are:
- `lift`: did the final algorithm/prompt produce real held-out improvement?
- `scientific_quality`: did the run behave like serious algorithm research?
- `throughput_engineering`: did the implementation search efficiently enough to
  be useful for repeated hill-climbing?

You must inspect:
- `eval_report.md`
- `artifacts/go_explore_result.json`
- `artifacts/experiment_summary.json`
- `artifacts/reportbench_output.json`
- `workspace/run_go_explore.py`

Do not assume success from prose alone. Ground every judgment in the submitted
artifacts.

## What To Verify

1. Lift on held-out Go-Explore results.
- Compare the final held-out baseline and best results in
  `artifacts/experiment_summary.json` and `artifacts/go_explore_result.json`.
- Check that held-out reward uplift is positive.
- Check that held-out achievement evidence is present and preferably improved or
  at least non-regressive.
- Use the per-seed held-out tables in the report to judge whether uplift looks
  broad enough to be credible rather than a single-seed fluke.
- Treat held-out lift as the primary outcome criterion.

2. Real Go-Explore execution happened.
- Confirm the bundle shows an actual search loop, not just baseline-only eval.
- Require evidence of at least 2 search iterations beyond baseline.
- Require evidence that at least 50 training rollouts were used.
- Require that elapsed runtime is at least 5 minutes, unless the report clearly
  explains an earlier stop caused by a configured termination condition.

3. The final report is grounded and complete.
- The report should describe the baseline, the best prompt, and what changed in
  the final Go-Explore behavior or code.
- The report should summarize the main experiment variants the worker tried, not
  just the winning prompt.
- The report must include per-seed held-out reward comparison.
- The report must include per-seed held-out achievement comparison.
- The report must state termination reason, elapsed time, total cost, estimated
  inference cost, and uplift per minute.

4. The experiment trail is real and informative.
- Inspect `iteration_log` and `archive_top_5` in `artifacts/go_explore_result.json`.
- Confirm the worker tried multiple distinct prompt variants, not the same prompt
  repeated with cosmetic edits only.
- Confirm the bundle preserves both successful and unsuccessful attempts.
- Prefer runs where the report explains what classes of changes were tried
  (for example: resource-priority changes, exploration-order changes, crafting
  strategy changes, risk/combat changes).
- If the final prompt choice is not clearly supported by the experiment trail,
  mark the run down.

5. Scientific quality: fidelity to Go-Explore at an algorithm/pseudocode level.
- Inspect `workspace/run_go_explore.py` and judge whether the final code reflects
  a meaningful Go-Explore-style search attempt rather than a trivial no-op.
- Look for Go-Explore-like ingredients at a high level:
  - an archive or memory of previously discovered promising states/prompts
  - explicit reuse of promising prior discoveries when generating new candidates
  - some notion of stepping stones, waypoints, or intermediate discoveries
    rather than only one-shot global prompt rewrites
  - a search/update loop that hill-climbs from prior good results
- It does not need literal game-state waypoint restoration, but it should show
  fidelity to the spirit of Go-Explore rather than generic random prompt search.
- If the implementation is really just shallow mutation search with Go-Explore
  branding, call that out.
- If the code appears inconsistent with the report or the artifacts, mark the
  run down.

6. Throughput and engineering quality.
- Judge how long the implementation takes to perform the completed search.
- Prefer submissions that make it easy to estimate how long 500 rollouts take in
  practice from `training_rollouts_used` and `elapsed_seconds`.
- Check `uplift_per_minute` and the implied rollouts-per-minute.
- Ask whether this implementation is fast enough to support repeated hill-climb
  cycles on the NanoHorizon/Craftax leaderboard objective.
- Penalize runs that achieve tiny lift with poor throughput unless the report
  clearly explains why the slower run produced unusually valuable scientific insight.

## Scoring Guidance

Return a JSON review with:
- `score`: float in `[0, 1]`
- `summary`: short overall judgment
- `strengths`: short list
- `weaknesses`: short list
- `axes`: object with float scores in `[0, 1]` for:
  - `lift`
  - `scientific_quality`
  - `throughput_engineering`
- `checks`: object with boolean-or-short-string outcomes for:
  - `heldout_lift_positive`
  - `heldout_lift_credible`
  - `search_executed`
  - `minimum_search_effort_met`
  - `experiment_variants_reviewed`
  - `failed_and_successful_attempts_recorded`
  - `final_choice_supported_by_experiments`
  - `go_explore_fidelity_present`
  - `waypoint_or_stepping_stone_logic_present`
  - `report_complete`
  - `achievement_evidence_present`
  - `achievement_uplift_non_regressive`
  - `uplift_per_minute_credible`
  - `500_rollout_runtime_estimated`
  - `throughput_viable_for_hillclimbing`
  - `final_code_review_passed`

Suggested interpretation:
- `0.9-1.0`: strong success
- `0.7-0.89`: qualified success with some caveats
- `0.4-0.69`: mixed or weak evidence
- `<0.4`: failure or insufficient evidence

When in doubt, prefer this weighting:
- `lift` is primary
- `scientific_quality` is second
- `throughput_engineering` is third

Favor artifact-grounded honesty over optimism.
