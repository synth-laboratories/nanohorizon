# Pivot Verifier Qwen3.5 4B Plan

This document proposes a new NanoHorizon track family built around a
post-transition verifier for pivot-point optimization.

The central object is a verifier over:

- state before the pivot: `s_t`
- reasoning at the pivot: `thinking_t`
- action or tool call at the pivot: `a_t`
- state after executing that pivot: `s_{t+1}`

This began as a planning document and now also records the current baseline
status.

## Baseline Progress Snapshot

Completed baseline pieces:

- separate track, config, script, and record surfaces from `pivotrl`
- `Qwen/Qwen3.5-4B` verifier with native Qwen tool-calling supervision
- vLLM evaluation with thinking-budget support
- rubric-backed golden eval traces for `collect_wood`, `place_table`,
  `make_wood_pickaxe`, `collect_stone`, and `make_stone_pickaxe`
- verifier SFT plus sampled reward-weighted refinement
- downstream reward-row, candidate-group, and preference-pair export
- first cheap-GPU downstream preference-policy baseline

Latest smoke result on Modal `L4`:

- parse reliability solved: `16 / 16` parsed tool-call outputs
- `mae_progress_reward = 0.045`
- `mae_total_reward = 0.2791`
- downstream preference pairs exported: `5`
- downstream preference-policy smoke completed successfully

What still feels baseline-grade rather than strong:

- verifier training data is still tiny and smoke-oriented
- refinement is not yet a true online RL verifier procedure
- downstream policy training uses a small preference set
- score calibration, especially on `total_reward`, still needs work

## Thesis

The best downstream DPO or RLVR procedure here likely comes from building the
strongest possible verifier first, then using that verifier as the labeler for
policy improvement.

The verifier should do two things at once:

- assign a process reward to the reasoning trace at the pivot
- assign a progress reward to the action that changed the environment

The downstream learner should not be asked to infer these distinctions from
terminal outcome alone.

## Why This Track

NanoHorizon already has online RLVR and emerging pivot-style work in:

- `src/nanohorizon/baselines/rlvr.py`
- `submissions/synth/pivotrl.py`
- `submissions/synth/pivotrl_core.py`

What is missing is a track where verifier quality is the first-class target.

This new lane should borrow patterns from the current `pivotrl` work, but it
should stay operationally separate from it.

Separation requirements:

- separate track ID
- separate configs and runner scripts
- separate record bundles
- separate evaluation questions
- no assumption that improvements to `pivotrl` and improvements to the verifier
  track are the same thing

This track is inspired by the same high-level bet as
[Digi-Q](https://arxiv.org/abs/2502.15760): use an offline-trained value-style
scorer to extract better actions from candidate sets without relying purely on
fresh online interaction. In Digi-Q, the scorer is an action-value function used
for best-of-`N` policy extraction. Here, the analogous scorer is a
post-transition pivot verifier that ranks pivot candidates and supplies labels
for DPO or rewards for RLVR.

Inference from that paper:

- offline verifier quality can substitute for some expensive online exploration
- best-of-`N` or ranked candidate extraction is a natural fit for pivot points
- the scorer should be optimized for decision quality, not just token-level loss

## Track Proposal

Proposed new track family:

- `pivot_verifier_qwen35_4b`

Recommended first regime:

- base policy: `Qwen/Qwen3.5-4B`
- environment: Craftax
- verifier model: `Qwen/Qwen3.5-4B`
- candidate construction: pivot groups from teacher rollouts plus sampled
  alternatives
- labeler: high-capacity post-transition verifier
- downstream learner: DPO first, RLVR second

Implementation stance:

- reuse pivot extraction, branching, and rollout-shaping patterns from
  `pivotrl` where useful
- keep verifier training, verifier evaluation, and downstream policy recipes in
  their own track-owned surface

Reason for the DPO-first recommendation:

- DPO gives the cleanest read on verifier ranking quality
- RLVR adds credit assignment and on-policy instability on top of verifier error
- once verifier quality is validated, RLVR can use the same labels or rewards

Recommended first baseline:

- SPCT-style generative verifier training on `Qwen/Qwen3.5-4B`
- supervised pseudo-labels first, then later online refinement
- native tool-call critique output with explicit `process_reward`,
  `progress_reward`, and `total_reward`

## Formal Objects

### Pivot tuple

A pivot tuple is:

- `prefix`: rollout context before the decision
- `s_t`: environment state at the decision point
- `thinking_t`: reasoning produced before acting
- `a_t`: chosen action, tool call, or short action burst
- `s_{t+1}`: immediate successor state after execution
- `aux`: short-horizon continuation signals such as `n`-step return, inventory
  delta, achieved prerequisites, validity, and safety flags

### Candidate group

A candidate group contains multiple pivot tuples that share the same rollout
prefix and `s_t`, but differ in `thinking_t` and/or `a_t`.

This group is the unit used for:

- pairwise preferences for DPO
- reward-weighted samples for RLVR
- top-`1` / best-of-`N` extraction analysis

### Verifier outputs

Recommended verifier outputs:

- `r_process`: how helpful, grounded, and non-spurious `thinking_t` was for this
  pivot
- `r_progress`: how much `a_t` improved the trajectory after execution
- `r_total`: combined score for ranking
- `u_total`: uncertainty or confidence for filtering low-certainty labels

Recommended combination:

- `r_total = w_p * r_process + w_a * r_progress`

with trackable weights rather than a single opaque scalar.

## Strongest-Verifier Recommendation

The strongest setup is likely a two-model arrangement:

1. `V_post(s_t, thinking_t, a_t, s_{t+1})`
2. optional distilled `V_pre(s_t, thinking_t, a_t)`

`V_post` is the canonical labeler.

Why:

- it can use direct evidence about the action's local effect
- it reduces ambiguity around whether bad reasoning led to good luck
- it can separate reasoning quality from action impact

Why also keep `V_pre` as optional:

- some downstream uses need a score before execution
- it can cheaply prune candidates before expensive environment branching
- it can imitate `V_post` and narrow the train-test gap

Important track rule:

- the gold labels come from `V_post` or environment-backed supervision
- `V_pre` is a distillation artifact, not the source of truth

## Data Strategy

### Stage 1: executed pivot corpus

Start from real executed rollouts from:

- the current RLVR baseline
- the current pivot pipeline
- stronger teacher policies
- failed trajectories, not just successful ones

For each rollout, extract pivot points where one of the following is true:

- a new subgoal becomes available
- the agent changes strategy
- the action space is materially ambiguous
- progress stalls or reverses

### Stage 2: counterfactual expansion

For each pivot point, sample additional candidates:

- multiple reasoning traces for the same state
- multiple actions conditioned on the same reasoning
- multiple reason-and-act bundles from the same prefix

Then execute enough of each candidate to obtain:

- the immediate `s_{t+1}`
- short-horizon progress evidence
- a compact continuation score

This creates the discrimination signal that pure executed data lacks.

### Stage 3: hard negatives

Actively include:

- locally valid but strategically harmful actions
- verbose but unhelpful reasoning
- reasoning that predicts the wrong affordance
- actions that increase apparent activity but do not unlock progress

The verifier will only be strong if these are common in training.

## Supervision Design

Recommended supervision sources, in priority order:

1. environment-backed deltas
2. short-horizon continuation outcomes
3. pairwise comparisons within a pivot group
4. teacher preferences or rubric labels
5. terminal return only as a weak auxiliary

Environment-backed progress features should include task-specific deltas such as:

- inventory gain
- prerequisite completion
- survivability change
- distance-to-subgoal improvement
- action validity and reversibility

Reasoning-process labels should emphasize:

- state grounding
- subgoal relevance
- action justification quality
- non-redundancy
- predictive usefulness

The verifier should not reward mere verbosity.

## Training Objectives

The first strong baseline should combine multiple objectives.

### Objective A: pairwise ranking

Within a shared pivot group, train the verifier to prefer the better candidate.

This is the most direct objective for downstream DPO data construction.

### Objective B: scalar regression

Predict `r_process`, `r_progress`, and `r_total` so RLVR can consume dense
rewards and so the decomposition stays inspectable.

### Objective C: TD-style consistency

Borrow the key value-learning intuition from Digi-Q and add local temporal
consistency targets where feasible:

- the score assigned to a pivot should align with the value of the resulting
  state under short lookahead or bootstrapped continuation estimates

This matters because the best pivot is not always the one with the largest
immediate visible change.

### Objective D: uncertainty-aware filtering

Teach the verifier to expose low-confidence cases and drop or downweight them
for downstream DPO/RLVR.

The strongest verifier is not just accurate. It is selective.

## Downstream Policy Procedures

### Option 1: state-to-action DPO

Recommended first downstream procedure:

- build groups from a shared `s_t`
- keep the highest-scoring and lowest-scoring candidates
- train the policy on chosen vs rejected actions

Recommended variants:

- reason-and-act DPO: preference over the whole `(thinking_t, a_t)` bundle
- action-only DPO: use verifier labels but train only on `a_t`
- mixed DPO: chosen reasoning from the winner, action from the executed branch
  only when reasoning/action disentanglement is needed

### Option 2: verifier-rewarded RLVR

Use verifier outputs as dense pivot rewards:

- `r_process` rewards reasoning quality
- `r_progress` rewards environment improvement
- `r_total` or a weighted mix is used in the learner objective

Recommended use:

- start with verifier reward as an additive shaping term beside environment
  reward
- only move to pure verifier reward after calibration proves robust

### Option 3: best-of-N extraction

Before or alongside learning, use the verifier as a best-of-`N` extractor:

- sample `N` pivot candidates
- execute or simulate to obtain post-transition evidence
- keep the top-ranked candidate

This provides a low-risk way to measure verifier utility before policy updates.

## Metrics

### Verifier metrics

- pairwise accuracy within pivot groups
- top-1 success lift over random or policy sampling
- calibration of `r_total`
- decomposition stability of `r_process` vs `r_progress`
- robustness on hard-negative slices

### Downstream metrics

- final `mean_outcome_reward`
- reward lift over the base RLVR baseline
- reward lift over the existing pivot baseline
- sample efficiency per labeled pivot
- stability across seeds

### Acceptance rule

Treat the verifier as successful only if both are true:

- it wins on held-out pivot ranking metrics
- it improves downstream policy reward, not just offline rank accuracy

## Likely In-Repo Additions

Likely ownership split:

- borrowed reference patterns for pivot extraction and counterfactual generation:
  `submissions/synth/pivotrl_core.py`
- verifier model, training loop, and evaluation:
  new modules under `src/nanohorizon/`
- configs and runners:
  `configs/` and `scripts/`
- public track notes:
  `docs/tracks/`

Likely new artifacts:

- pivot dataset manifest
- verifier training config
- verifier eval report
- downstream DPO config
- downstream RLVR config

## Implementation Sequence

We should build this in strict stages.

### Phase 0: schema and metrics

1. define the pivot tuple schema
2. define group-level ranking metrics
3. define `r_process`, `r_progress`, and `r_total`

### Phase 1: dataset builder

1. extract executed pivots from current traces
2. build grouped candidate examples at shared states
3. write manifests and held-out splits

### Phase 2: verifier bootstrap

1. train a first post-transition verifier on executed pivots
2. add pairwise ranking and scalar heads
3. measure held-out ranking accuracy

### Phase 3: hard-negative and counterfactual expansion

1. sample alternative candidates per pivot
2. execute short-horizon branches
3. retrain the verifier on harder groups

### Phase 4: downstream DPO

1. build chosen-vs-rejected pairs from verifier scores
2. train a policy on pivot preferences
3. evaluate reward uplift

### Phase 5: downstream RLVR

1. inject verifier-derived dense rewards into pivot training
2. compare mixed reward vs verifier-only shaping
3. evaluate reward uplift and stability

## First Coding Milestones

The first milestones should be executed in this order:

1. add a pivot schema and manifest writer
2. emit grouped pivot examples from the current pivot pipeline
3. train a bootstrap post-transition verifier
4. run verifier-ranked best-of-`N` evaluation without policy training
5. run verifier-labeled DPO on top-vs-bottom candidates
6. run verifier-shaped RLVR only after DPO establishes signal quality

## Main Risks

Main modeling risks:

- the verifier may reward plausible-looking reasoning instead of useful
  reasoning
- `s_{t+1}` may let the model overfit to local artifacts that do not transfer to
  better long-horizon policy learning
- scalar collapse may hide whether gains came from reasoning quality or action
  progress

Main systems risks:

- branching candidate execution can become too expensive
- pivot extraction may undersample the hard decisions that matter most
- DPO pairs may become noisy if candidate groups are not tightly matched

Main experimental risks:

- high offline ranking accuracy may not improve policy reward
- RLVR may amplify verifier mistakes faster than DPO
- the best verifier for labeling may be too slow to use directly online

## Immediate Next Step

Implement the smallest real artifact:

- a pivot dataset builder that emits grouped
  `(s_t, thinking_t, a_t, s_{t+1})` examples plus basic progress labels

After that, train the first post-transition verifier and use it for best-of-`N`
ranking before committing to downstream DPO or RLVR.
