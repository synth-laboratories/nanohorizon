# Prompt Opt $1 GPT-5.4 Family

This is the prompt optimization benchmark track for NanoHorizon.

**Track ID:** `prompt_opt_1usd_gpt54_family` (used in `records/<track_id>/…`)

## Contract

- policy model: `Qwen/Qwen3.5-4B`
- environment: Craftax
- optimizer budget: $1 total compute spend
- optimizer models: any mix of `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`

## Allowed During The Optimization Budget

- prompt search
- system prompt rewriting
- action-format prompt iteration
- exemplar selection
- tool-free prompt tuning with `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano` as the optimizer family

## Not Allowed During The Optimization Budget

- weight updates to the policy model
- swapping the final deployed policy away from `Qwen/Qwen3.5-4B`
- exceeding $1 total optimizer spend

## Expected Starter Script

```bash
./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh
```

## Reference Baseline

- checked-in record:
  [../../records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/](../../records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/)
- checked-in score: `0.35`
- bootstrap seed-prompt score on the same held-out eval: `0.6`
- score delta: `-0.25`

Interpretation:

- the leaderboard score for this track is actual Craftax `mean_outcome_reward`
- `GEPA` search score is tracked in `metrics.json -> best_gepa_val_score`, but it is not the track score
- this checked-in baseline is a completed honest-accounting reference run, even though it regressed from the seed prompt

## Reference Stack

- Modal for execution
- `gpt-5.4-mini` proposer by default
- shared Craftax eval harness
- default example GPU: `L4` via `NANOHORIZON_MODAL_GPU_PROMPT_OPT`

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- `prompt_bundle.json`
