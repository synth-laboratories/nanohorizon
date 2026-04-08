# Track overview

NanoHorizon now has **four** official tracks across two task families. The first family is long-horizon RL on Craftax; the second is classic RL on Craftax-Classic. Full rules and contracts live in per-track docs:

| Track ID | Doc |
| --- | --- |
| `rlvr_20min_2xa100_40gb` | [RLVR track](tracks/rlvr_20min_2xa100_40gb.md) |
| `offline_20min_1xa100_40gb` | [Offline track](tracks/offline_20min_1xa100_40gb.md) |
| `prompt_opt_1usd_gpt54_family` | [Prompt optimization track](tracks/prompt_opt_1usd_gpt54_family.md) |
| `classic` | [Classic track](tracks/classic.md) |

Task definitions: [docs/task-craftax.md](task-craftax.md) and [docs/task-craftax-classic.md](task-craftax-classic.md)

## Planned Track Families

These are planning docs for proposed lanes, not finalized public benchmark
contracts yet.

| Track ID | Doc |
| --- | --- |
| `cpt_rlvr_qwen35_0p8b` | [CPT -> RLVR planned track](tracks/cpt_rlvr_qwen35_0p8b.md) |
| `pivot_verifier_qwen35_4b` | [Pivot verifier planned track](tracks/pivot_verifier_qwen35_4b.md) |
| `ttc_reflexion_todo_nano` | [Test-time context planned track](tracks/ttc_reflexion_todo_nano.md) |

## RLVR track — summary

- Intent: end-to-end engineering for a short real training run; model-in-the-loop data and RLVR-style updates.
- Budget: 20 minutes wall clock, 2× A100 40GB.

## Offline track — summary

- Intent: gains from fixed Craftax data under a strict single-GPU budget; data selection and offline post-training.
- Budget: 20 minutes wall clock, 1× A100 40GB.

## Prompt optimization track — summary

- Intent: improve Craftax performance via prompts and prompt search, not weight updates; $1 optimizer budget on GPT-5.4 family models; deployed policy stays `Qwen/Qwen3.5-4B`.
- Budget: $1 total optimizer spend across `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`.

## Classic track — summary

- Intent: small-model RL from random initialization on Craftax-Classic in JAX.
- Regime: Craftax `1M`, policy cap `<100M`, separate methods and hardware from the Craftax tracks.
- Runtime: no container abstractions, no Modal substrate, no OpenAI-compatible inference layer.

## Shared principles

- Craftax tracks share the Craftax task family ([task doc](task-craftax.md)).
- Classic uses a separate Craftax-Classic task family ([task doc](task-craftax-classic.md)).
- Base model target is `Qwen/Qwen3.5-4B` only for the Craftax tracks unless a track doc states otherwise.
- Shared Qwen family baseline; the offline reference runner currently uses `Qwen/Qwen3.5-4B` with a `Qwen/Qwen3.5-9B` teacher.
- Same public record bundle shape under `records/<track_id>/…`.
- Evaluation should be reproducible from pinned code and config.
- Craftax example runners are Modal-first and share the same Modal substrate under `src/nanohorizon/modal_*.py`.
- Classic is intentionally outside that substrate and should remain a direct JAX/Craftax training path.
