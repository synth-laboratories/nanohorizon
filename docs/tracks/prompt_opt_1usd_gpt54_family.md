# Prompt Opt $1 GPT-5.4 Family

This is the prompt optimization benchmark track for NanoHorizon.

**Track ID:** `prompt_opt_1usd_gpt54_family` (used in `records/<track_id>/…`)

## Contract

- policy model: `Qwen/Qwen3.5-0.8B`
- environment: Crafter
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
- swapping the final deployed policy away from `Qwen/Qwen3.5-0.8B`
- exceeding $1 total optimizer spend

## Expected Starter Script

```bash
./scripts/run_crafter_prompt_opt_qwen35_08b_gpt54_budget.sh
```

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- `prompt_bundle.json`
