# NanoHorizon PR103 Place-Plant Fallback Candidate

## Context & objective

This run updated only `submission/agent.py` to add one compact fallback sentence to the Craftax system prompt:
after wood and sapling are secured, try `place_plant` if available; otherwise seek drink/water rather than wandering.

Success for this turn was a minimal reviewable submission change plus a lightweight honest train-seed evaluation record.

## Experiments cited

1. `submission/agent.py`
   - Question: does the candidate surface contain only the intended PR103 fallback change?
   - Outcome: supporting.
   - Evidence: the only code delta is the added system-prompt sentence in `define()`.

2. Train-seed eval probe against `nanohorizon.shared.eval_model.evaluate_model`
   - Question: can the candidate be exercised on the curated train seeds with the existing local rollout harness?
   - Outcome: resource-blocked.
   - Evidence: the first direct attempt through `submission.agent.eval()` failed because that wrapper forwards unsupported kwargs to `evaluate_model()`. A second direct probe through the shared evaluator reached Craftax import and texture bootstrap, then was killed by the workspace while processing textures.

## Insights

1. The submission change stayed inside the intended surface: one prompt sentence, no train/eval logic rewrite.
2. The local honest-eval path is blocked by workspace/runtime constraints, not by the prompt change itself.
3. The current repo state is sufficient for PR review, but not for a trustworthy numeric lift claim from this machine.

## Research artifacts produced

- Code: `submission/agent.py`
- Report: `eval_report.md`

## Quality & validation

- Validation attempted:
  - direct `submission.agent.eval()` on the curated train seeds
  - direct shared-evaluator probe using the checked-in prompt-opt inference URL
- Validation result:
  - `submission.agent.eval()` failed immediately because it passes `target_action_batch_size` and `min_action_batch_size` to `evaluate_model()`, which does not accept them.
  - the direct shared-evaluator probe loaded Craftax from the classic dependency group, but the process was killed during texture bootstrap before any rollout reward could be recorded.
- Validation skipped / not completed:
  - no trustworthy train-seed score
  - no held-out evaluation
  - no leaderboard lift claim

## Reproduction & handoff

- Candidate file: `submission/agent.py`
- Report file: `eval_report.md`
- Suggested next step for a fuller check: rerun the train-seed probe in a larger runtime that already has Craftax textures cached and enough memory to survive the rollout bootstrap.

