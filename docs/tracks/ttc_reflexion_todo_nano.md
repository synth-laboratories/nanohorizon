# Test-Time Context (Reflexion + TODO) Nano

This is the planned test-time-context track family for NanoHorizon.

**Track ID:** `ttc_reflexion_todo_nano`

## Contract

- policy model: `gpt-4.1-nano`
- environment: Craftax
- optimization surface: test-time context only (prompt/context/memory control)
- no policy weight updates
- no model-family swap for the deployed policy

## Core Idea

This lane isolates a simple question:

Can we improve long-horizon Craftax behavior by changing how context is
constructed and updated at inference time, while keeping model weights fixed?

The methods focus on two concrete patterns:

- Reflexion-style self-feedback and episodic memory
- explicit hidden TODO planning and refresh logic

Reflexion reference:

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

Inference from that paper:

- verbal feedback can improve decision quality without gradient updates
- memory and self-critique quality are first-class bottlenecks
- gains depend on disciplined memory filtering, not just adding more text

## Method Family

### Reflexion-style memory

- append short post-step critiques to a compact memory buffer
- keep only high-signal memories (errors, dead-end patterns, successful pivots)
- inject selected memories into the next step prompt

### TODO-style planner

- maintain a private three-item TODO list each turn
- force ordered prioritization: blocker first, progress action second, safety
  check third
- refresh TODO items when state changes or repeated action loops are detected

### Context hygiene

- preserve strict token budgets for state, memory, and TODO sections
- summarize stale context instead of unbounded accumulation
- prefer deterministic template slots over free-form prompt drift

## Allowed Changes

- system prompt and context template design
- memory write rules and memory selection heuristics
- TODO generation and refresh gates
- context compaction and section-budget policies
- short-horizon self-critique formatting and scoring heuristics

## Not Allowed

- gradient-based model updates
- finetuning adapters or LoRA updates
- replacing `gpt-4.1-nano` as the deployed policy model

## Initial Baseline Surface

First baseline files:

- `src/nanohorizon/baselines/prompt_opt.py`
- `configs/craftax_reflexion_nano_baseline.yaml`
- `scripts/run_craftax_reflexion_nano_baseline.sh`
- `configs/craftax_reflexion_nano.yaml`
- `configs/craftax_reflexion_nano_10x10.yaml`

This lane should begin with TODO-refresh and context-discipline baselines, then
add Reflexion-style memory updates as an ablation-controlled step.

## Evaluation Stance

- primary score: held-out Craftax `mean_outcome_reward`
- run paired ablations against the same fixed policy model:
  - base prompt only
  - TODO only
  - Reflexion memory only
  - TODO + Reflexion combined
- report token overhead, context length distribution, and loop-rate metrics

## Next Steps

1. Run the first public baseline record under
   `records/ttc_reflexion_todo_nano/...`.
2. Add an ablation table that separates TODO gains from Reflexion-memory gains.
