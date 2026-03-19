# Track Overview

NanoHorizon starts with two official tracks.

## RLVR Track

Path: `tracks/rlvr_20min_2xa100_40gb`

Intent:
- measure end-to-end engineering skill for a short-horizon real training run
- allow model-in-the-loop data collection and RLVR-style updating

Budget:
- 20 minutes wall clock
- 2x A100 40GB

## Offline Track

Path: `tracks/offline_20min_1xa100_40gb`

Intent:
- measure what can be gained from fixed Crafter data under a strict single-GPU budget
- encourage strong data selection, filtering, and offline post-training

Budget:
- 20 minutes wall clock
- 1x A100 40GB

## Shared Principles

- same task family: Crafter
- same base model family: `Qwen/Qwen3.5-0.8B`
- same public record bundle shape
- evaluation should be reproducible from pinned code and config

## Prompt Optimization Track

Path: `tracks/prompt_opt_1usd_gpt54_family`

Intent:
- measure how much Crafter performance can be improved through prompt engineering and prompt search rather than weight updates
- let entrants spend a small fixed optimizer budget on GPT-5.4 family models while keeping the deployed policy fixed to `Qwen/Qwen3.5-0.8B`

Budget:
- $1 total optimizer spend
- spend may be split across `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`
- evaluation policy remains `Qwen/Qwen3.5-0.8B`
