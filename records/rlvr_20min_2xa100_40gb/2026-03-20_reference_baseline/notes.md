# Reference baseline

This bundle documents the checked-in RLVR reference implementation.

## Reproduce

From the repository root:

```bash
./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh
```

Edit only:

```bash
src/nanohorizon/rlvr_training.py
```

## Architecture

- Modal-hosted Craftax service built from `Craftax core runtime`
- Modal-hosted vLLM served through a stable OpenAI-compatible proxy
- learner loop in `src/nanohorizon/rlvr_training.py`
- grouped on-policy Craftax rollouts with group-relative rewards
- sequence-level clipped GRPO-style LoRA updates

## Status

The implementation is in repo. This placeholder record is intentionally unscored until the first full reference run lands.
