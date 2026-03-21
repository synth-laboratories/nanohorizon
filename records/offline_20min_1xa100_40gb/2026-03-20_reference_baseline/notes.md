# Reference baseline

This record packages the current NanoHorizon offline / SFT reference baseline for Crafter.

## Reproduce

From the repository root:

```bash
./scripts/run_offline_training.sh
```

That script already defaults to the proved settings:

- student: `Qwen/Qwen3.5-4B`
- teacher: `Qwen/Qwen3.5-9B`
- tool-calling-only Crafter traces
- `thinking_budget_tokens = 2000`
- held-out compare: `20` eval rollouts at concurrency `10`

## Measured result

- base mean reward: `0.3`
- finetuned mean reward: `0.5`
- reward delta: `+0.2`

## Source artifacts used for this record

- local training artifact: `artifacts/fbc_ngrok_proof_20260320T205744Z/modal_offline_result.json`
- local comparison artifact: `artifacts/fbc_compare_only_20260320T215130Z/comparison_summary.json`

## Caveat

The Crafter summary payload still reports `llm_call_count: 0` even on successful rewarded rollouts. That is a container-side accounting bug, not an eval failure.
