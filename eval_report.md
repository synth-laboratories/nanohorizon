# NanoHorizon Submission Eval

## Context

This run is limited to `submission/agent.py`. The candidate fixes compatibility with the current `evaluate_model` signature and keeps the policy on the 8-action batch / thinking-enabled setting while tightening the Craftax prompt.

## Eval setup

- Command shape:
  - `python submission/agent.py eval --data-dir <temp>/data --out-dir <temp>/out --checkpoint-dir <temp>/out`
- Remote inference:
  - `NANOHORIZON_EVAL_INFERENCE_URL=https://api.openai.com/v1/chat/completions`
  - `NANOHORIZON_EVAL_REQUEST_MODEL=gpt-4.1-mini`
  - `NANOHORIZON_EVAL_API_KEY=$OPENAI_API_KEY`
- Slice:
  - train seeds: `0, 1, 2`
  - `NANOHORIZON_SUBMISSION_MAX_STEPS=3`
  - `NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS=128`

## Comparison

Baseline reconstruction:

- mean outcome reward: `0.6666666666666666`
- mean LLM calls per rollout: `3.3333333333333335`

Candidate:

- mean outcome reward: `1.3333333333333333`
- mean LLM calls per rollout: `6.0`

## Readout

- The candidate improved reward on this small repeated-seed slice.
- The candidate also increased LLM call count, so the gain is not free.
- This is still a small remote eval slice, not the official leaderboard harness.
