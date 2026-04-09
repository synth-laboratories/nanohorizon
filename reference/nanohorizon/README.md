# NanoHorizon Go-Explore Prompt Optimization

This lane is for open-ended algorithm research on Go-Explore for Craftax, aimed
at hill-climbing the NanoHorizon-style leaderboard objective.

The intended end state is not just "find a nicer prompt." The intended end state
is "produce a better Go-Explore algorithm/prompt package for Craftax under a
fixed Gemini 2.5 Flash Lite rollout budget."

## Primary goals

1. Lift
- Improve held-out Craftax reward on the final held-out rollout set.
- Improve or at least avoid materially regressing held-out achievements.

2. Scientific quality
- Behave like real algorithm research, not random prompt fiddling.
- Leave behind a useful experiment trail with both wins and losses.
- Show fidelity to Go-Explore ideas such as archive reuse, checkpoint/cell
  selection, stepping-stone discovery, and hill-climbing from prior progress.

3. Throughput / engineering
- Make the implementation useful for repeated hill-climb cycles.
- Estimate how long 500 rollouts takes and whether the throughput is practical.

## Canonical source roots

The canonical implementation split is:

- `nanolong/go_explore`:
  canonical Go-Explore implementation home
- `nanohorizon`:
  Craftax benchmark/runtime framing, container setup, and NanoHorizon-side task
  context

This staged task includes a curated reference packet under `reference/`, but the
real sibling repos are also available locally on this machine:

- `/Users/joshpurtell/Documents/GitHub/nanolong`
- `/Users/joshpurtell/Documents/GitHub/nanohorizon`

## Most important reference files in this packet

Read these first:

- `reference/nanolong/go_explore/README.md`
- `reference/nanolong/go_explore/run_full_go_explore.py`
- `reference/nanolong/go_explore/full_service.py`
- `reference/nanolong/go_explore/full_runtime.py`
- `reference/nanolong/go_explore/full_config.py`
- `reference/nanolong/go_explore/storage.py`
- `reference/nanolong/go_explore/real_crafter.py`
- `reference/nanolong/go_explore/legacy/algorithm_dev/pseudocode_summary.txt`
- `reference/nanolong/go_explore/legacy/waypoint_operationalization_notes.txt`
- `reference/nanolong/go_explore/legacy/optimizer.py`
- `reference/nanolong/go_explore/legacy/search/archive.py`
- `reference/nanolong/examples/go_explore_crafter/README.md`
- `reference/nanohorizon/docker/craftax_go_explore.Dockerfile`
- `reference/nanohorizon/src/nanohorizon/craftax_core/runner.py`
- `reference/nanohorizon/src/nanohorizon/baselines/prompt_opt.py`
- `reference/nanohorizon/docs/task-craftax.md`

## Working assumption for this lane

`workspace/run_go_explore.py` is only a starter harness.

The worker should treat the `reference/nanolong/go_explore` material as the real
algorithm baseline and improve the active workspace implementation so it better
matches the spirit of Go-Explore while still satisfying this lane's artifact
contract.

## High-level Go-Explore algorithm spec

At a high level, the algorithm should look like this:

1. Initialize runtime, archive, scoring, and candidate frontier.
2. Run fresh-start exploration from a seed set.
3. Ingest rollout results, checkpoints, labels, and summaries into an archive.
4. Select promising branch points from the archive.
5. Resume from those branch points or checkpoint-like cells.
6. Explore continuations that can discover new progress beyond what fresh starts found.
7. Re-score archive items using progress, novelty, frontier expansion, and
   reusable stepping-stone value.
8. Use the accumulated evidence to update the prompt/algorithm frontier.
9. Re-evaluate the best candidate on a held-out rollout set.

Important algorithm ideas to preserve or improve:

- archive-backed search, not pure random restart
- stepping stones / waypoints / cells / checkpoints as optimizer-side objects
- branching from promising prior states or discoveries
- comparing fresh-start vs resumed exploration
- using experiment evidence to choose the next prompt/algorithm variant

This lane does not require literal game-state waypoint supervision to the agent,
but it should preserve waypoint-like thinking at the optimizer/archive level.

## What counts as success

A good final run should show:

- positive held-out reward lift
- clear evidence of which variants were tried
- a final choice supported by the experiment trail
- plausible Go-Explore fidelity at the pseudocode/algorithm level
- acceptable throughput for repeated 500-rollout hill-climbing

## Starting ideas for improvement

- Improve archive selection:
  choose branch points using novelty, progress gain, and under-explored frontier value instead of only mean score.

- Add stronger stepping-stone logic:
  model cells or pseudo-waypoints from achievements, inventory progression, crafting milestones, or checkpoint depth.

- Distinguish fresh vs resumed budgets:
  spend some budget on broad exploration and some on continuation from promising states.

- Improve prompt mutation quality:
  organize mutations into strategy families such as early resource gathering,
  crafting order, exploration policy, combat/risk posture, and checkpoint continuation behavior.

- Improve experiment logging:
  make it easy to recover which hypotheses were tried and why they won or lost.

- Improve throughput:
  reduce unnecessary serial work, batch evaluation more effectively, and surface
  rollouts-per-minute or time-to-500-rollouts clearly.

- Improve held-out reliability:
  avoid selecting a winner that only beats baseline on a few lucky seeds.

- Improve achievement shaping:
  treat achievements as stepping-stone signals rather than only final score.

## Deliverable mindset

The final prompt is allowed to be the main artifact, but the verifier should be
able to say that it represents a better algorithmic Go-Explore policy for
Craftax, not merely a lucky wording change.
