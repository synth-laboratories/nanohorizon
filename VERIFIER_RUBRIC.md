# NanoHorizon Unique-Achievement GEPA Verifier Rubric

Return strict JSON with:

- `score`
- `summary`
- `criteria`
- `notes`

Assign `0.0` if any of these are true:

- `artifacts/unique_achievement_baseline.json` is missing
- `artifacts/unique_achievement_gepa_result.json` is missing
- `artifacts/unique_achievement_summary.json` is missing
- `artifacts/reportbench_output.json` is missing
- `eval_report.md` is missing
- `artifacts/unique_achievement_baseline.json` does not ground the baseline in
  the required local Gemini prompt-opt path
- the bundle uses an `offline_*`, `rlvr_*`, checked-in reference, or other
  non-local baseline
- the bundle relies on copied prior-run surrogate, manual-probe, or reference
  artifacts instead of a fresh run for this lane

Score the lane on these criteria:

1. `baseline_grounding` weight `0.20`
The bundle shows a real fresh local prompt-opt baseline run using
`run_craftax_prompt_opt_gemini25_flash_lite_local.sh` and
`craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml`. The baseline must
identify `baseline_track_id = "prompt_opt_local_gemini25_flash_lite"`.

2. `lift` weight `0.30`
The optimized run improves held-out unique-achievement performance over baseline
on the same measurement set.

3. `scientific_quality` weight `0.20`
The report explains what GEPA tried, preserves the experiment trail, and does
not confuse GEPA search-score movement with the real held-out objective.

4. `throughput_reporting` weight `0.15`
The bundle reports wall-clock time or throughput for the 500-rollout run.
If training rollout usage is materially below budget, it must also report a
concrete termination reason.

5. `artifact_quality` weight `0.15`
The summary and report are coherent, complete, and grounded in concrete
artifacts.
