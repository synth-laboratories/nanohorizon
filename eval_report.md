# NanoHorizon publication smoke eval report

## Context & objective

This run was scoped to a single-file publication-smoke change in [`submission/agent.py`](submission/agent.py). The goal was to keep the candidate diff inside the allowed submission surface while making the eval path more robust for transient rollout failures.

## Experiments cited

### Baseline

- Question: does the pre-change `submission/agent.py eval` path survive a transient rollout failure on repeated seeds?
- Method: `python /tmp/baseline_agent.py eval --checkpoint-dir /tmp/nh_eval_baseline/checkpoint --data-dir /tmp/nh_eval_baseline/data --out-dir /tmp/nh_eval_baseline/out`
- Setup: `sitecustomize.py` patched `nanohorizon.shared.eval_model.evaluate_model` to raise `RuntimeError("simulated transient deadlock")` on the first call and succeed afterward.
- Outcome: negative. The run aborted on the first seed with the simulated deadlock.
- Evidence: process exit `1`; traceback showed the uncaught `RuntimeError` from the stubbed evaluator.

### Candidate

- Question: does the updated eval path retry the first transient failure and finish the same repeated-seed run?
- Method: `PYTHONPATH="$tmpdir:/workspace/src" python submission/agent.py eval --checkpoint-dir /tmp/nh_eval_candidate/checkpoint --data-dir /tmp/nh_eval_candidate/data --out-dir /tmp/nh_eval_candidate/out`
- Setup: same stubbed evaluator and the same repeated seed list `[101, 101]`.
- Outcome: supporting. The run completed and produced two successful rollouts after one retry.
- Evidence: process exit `0`; output reported `num_eval_rollouts: 2`, `num_rollout_errors: 0`, `primary_score: 1.0`, and both details had `seed: 101`.

## Insights

1. The retry wrapper makes the publication-smoke eval path resilient to a single transient deadlock-like failure on the first seed.
2. Removing the unsupported batch-size keywords from the `evaluate_model` call was necessary for the eval entrypoint to execute against the live evaluator contract.
3. The candidate does not change successful rollout scoring in the stubbed comparison; it changes failure handling and completion behavior.

## Research artifacts produced

- Environment: repo workspace at `/workspace`, with eval stubs created outside the repo under `/tmp`.
- Data: synthetic repeated-seed inputs for the controlled comparison, `seeds.json = [101, 101]`.
- Models / checkpoints: no model training or checkpoint promotion was performed in this run.

## Quality & validation

- Validated: baseline abort behavior vs candidate completion behavior under an injected transient evaluator failure.
- Not validated: real Craftax reward lift, live container behavior, or long-horizon task quality.

## Reproduction & handoff

- Candidate commit will be created from the current workspace state after this report is written.
- Reproduce the comparison with the commands above and the same stubbed `sitecustomize.py`.
- Open risk: the validation is a smoke test only; it demonstrates retry robustness, not actual policy performance.
