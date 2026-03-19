# Track overview

NanoHorizon has **three** official tracks on the Crafter task. Full rules and contracts live in per-track docs:

| Track ID | Doc |
| --- | --- |
| `rlvr_20min_2xa100_40gb` | [RLVR track](tracks/rlvr_20min_2xa100_40gb.md) |
| `offline_20min_1xa100_40gb` | [Offline track](tracks/offline_20min_1xa100_40gb.md) |
| `prompt_opt_1usd_gpt54_family` | [Prompt optimization track](tracks/prompt_opt_1usd_gpt54_family.md) |

Task definition: [docs/task-crafter.md](task-crafter.md)

## RLVR track — summary

- Intent: end-to-end engineering for a short real training run; model-in-the-loop data and RLVR-style updates.
- Budget: 20 minutes wall clock, 2× A100 40GB.

## Offline track — summary

- Intent: gains from fixed Crafter data under a strict single-GPU budget; data selection and offline post-training.
- Budget: 20 minutes wall clock, 1× A100 40GB.

## Prompt optimization track — summary

- Intent: improve Crafter performance via prompts and prompt search, not weight updates; $1 optimizer budget on GPT-5.4 family models; deployed policy stays `Qwen/Qwen3.5-0.8B`.
- Budget: $1 total optimizer spend across `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`.

## Shared principles

- Same task family: Crafter ([task doc](task-crafter.md)).
- Same base model family: `Qwen/Qwen3.5-0.8B` (except optimizer APIs on the prompt track).
- Same public record bundle shape under `records/<track_id>/…`.
- Evaluation should be reproducible from pinned code and config.
