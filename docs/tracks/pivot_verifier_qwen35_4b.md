# Pivot Verifier Qwen3.5 4B

This is the active pivot-verifier training track family for NanoHorizon.

**Track ID:** `pivot_verifier_qwen35_4b`

## Contract

- downstream policy family: `Qwen/Qwen3.5-4B`
- verifier family: defaulting to `Qwen/Qwen3.5-4B`
- environment: Craftax
- training unit: pivot-point decision groups extracted from rollouts
- canonical verifier input:
  `(s_t, thinking_t, a_t, s_{t+1})`
- canonical verifier outputs:
  - process reward for `thinking_t`
  - progress reward for `a_t`
  - total pivot score used for ranking or weighting
- downstream policy optimization:
  - state-to-action DPO on high-vs-low scoring pivot candidates
  - or RLVR on pivot data with verifier-derived rewards

## Current Status

The baseline now exists and runs end to end on Modal `L4`.

Implemented surfaces:

- native Qwen 3.5 tool-calling verifier training
- bounded thinking-budget evaluation with vLLM
- rubric-backed golden Craftax eval set
- verifier SFT plus sampled reward-weighted refinement
- downstream preference export from verifier-labeled pivot groups
- a cheap-GPU downstream preference-policy baseline using a reference-free
  pairwise objective

Latest successful smoke run:

- Modal run: [ap-aVMgLDFK10S6ytaSyPDbcK](https://modal.com/apps/synth-laboratories/main/ap-aVMgLDFK10S6ytaSyPDbcK)
- verifier train rows: `2`
- golden eval rows: `16`
- eval parse rate: `1.0`
- eval `mae_progress_reward`: `0.045`
- eval `mae_total_reward`: `0.2791`
- eval pairwise accuracy: `0.678`
- downstream candidate groups: `6`
- downstream preference pairs: `5`

Current baseline interpretation:

- verifier stage: SPCT-shaped generative verifier with native tool calls
- refinement stage: reward-weighted sampled refinement, not PPO/GRPO
- downstream policy stage: verifier-labeled preference tuning on pivot groups

## Core Idea

The track treats verifier quality as the main bottleneck.

At each pivot point, we collect or sample multiple candidate actions from the
same state, execute them far enough to observe `s_{t+1}`, and train a verifier
to score:

- how useful the reasoning was for this pivot
- how much the chosen action advanced the trajectory

That verifier then becomes the labeler for downstream policy improvement.

## Separation Rule

This track should borrow successful data-construction and pivot-sampling
patterns from the current `pivotrl` work, but it should remain a separate track
family.

That means:

- reuse ideas and helper patterns where they help
- do not treat this as a rename of the current `pivotrl` pipeline
- keep separate configs, scripts, records, and success criteria
- evaluate verifier quality as its own first-class objective

## Intended Pipeline

1. bootstrap Craftax trajectories with a teacher or current policy
2. slice trajectories into pivot tuples
3. sample candidate think-and-act continuations at each pivot
4. execute candidates to obtain `s_{t+1}` and short-horizon evidence
5. train the verifier to rank and score candidates
6. create pivot preference groups or reward-weighted batches
7. run DPO or RLVR on those verifier-labeled pivot examples

## Recommended Shape

The strongest version of the track should keep a high-capacity post-transition
verifier as the source of truth:

- `V_post(s_t, thinking_t, a_t, s_{t+1})`

Optionally, distill that into a cheaper pre-transition scorer for pruning or
best-of-`N` inference-time reranking:

- `V_pre(s_t, thinking_t, a_t)`

`V_post` is the primary artifact for dataset labeling even if `V_pre` is later
used operationally.

## Baseline Notes

The baseline verifier should follow the same Qwen 3.5 + vLLM native tool-call
pattern used elsewhere in NanoHorizon:

- supervise the verifier to emit one native tool call, not free-form JSON
- run vLLM with Qwen reasoning and tool-call parsers enabled
- use thinking-budget controls so the verifier can spend bounded private tokens
  before producing the tool call

## Likely Files

- borrow patterns from:
  - `submissions/synth/pivotrl.py`
  - `submissions/synth/pivotrl_core.py`
- verifier baseline runner:
  - `src/nanohorizon/baselines/pivot_verifier.py`
- verifier baseline config and script:
  - `configs/pivot_verifier_qwen35_4b_spct.yaml`
  - `configs/pivot_verifier_qwen35_4b_spct_modal_smoke.yaml`
  - `scripts/run_craftax_pivot_verifier_qwen35_4b_spct.sh`
  - `submissions/synth/pivot_verifier.py`
- `docs/tracks/pivot_verifier_qwen35_4b_plan.md`

## Next Steps

- scale beyond the smoke sample to real pivot rollouts from stronger Craftax
  traces
- expand the golden eval set again with harder negatives and more ambiguous
  near-ties
- improve `total_reward` calibration, which is weaker than parse reliability
- compare downstream verifier-labeled preference tuning against verifier-shaped
  RLVR
- decide whether the next verifier-improvement stage should stay reward-weighted
  or move to a true online RL objective
