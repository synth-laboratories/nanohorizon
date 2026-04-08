Spark v4 E2E prompt candidate for compact todo scratchpad tracking.

Hypothesis:
- A hard-coded three-item private scratchpad in the rollout prompt loop plus a
  deterministic loop-refresh rule will reduce repeated no-progress movement.
- The new scratchpad entries are compact and carried per turn, so the harness can
  track danger, target progression, and fallback actions across a rollout.

What changed:
- Added compact todo scratchpad initialization and refresh logic inside the core
  rollout loop (`src/nanohorizon/craftax_core/rollout.py`).
- Added regression coverage to assert the scratchpad appears in user prompts and
  refreshes when the action pattern repeats without progress.
- Added a candidate config and not-run record bundle under
  `records/prompt_opt_1usd_gpt54_family/2026-04-08_spark_v4_e2e`.

Verification status:
- Structured candidate packaging is in place.
- Live GEPA run not executed in this task; leaderboard score remains unmeasured.
