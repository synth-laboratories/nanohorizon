# Prompt Opt $1 GPT-5.4 Family

This is the prompt optimization benchmark track for NanoHorizon.

**Track ID:** `prompt_opt_1usd_gpt54_family` (used in `records/<track_id>/…`)

## Contract

- policy model: `Qwen/Qwen3.5-4B`
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
- swapping the final deployed policy away from `Qwen/Qwen3.5-4B`
- exceeding $1 total optimizer spend

## Expected Starter Script

```bash
./scripts/run_crafter_prompt_opt_qwen35_4b_gpt54_budget.sh
```

## Reference Stack

- Modal for execution
- `gpt-5.4-mini` proposer by default
- shared Crafter eval harness
- default example GPU: `L4` via `NANOHORIZON_MODAL_GPU_PROMPT_OPT`

## Expected Record Bundle

- `metadata.json`
- `run_config.yaml`
- `metrics.json`
- `system_info.json`
- `command.txt`
- `prompt_bundle.json`
