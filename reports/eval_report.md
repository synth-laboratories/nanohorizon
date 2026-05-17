# Publication Smoke Submission Evaluation

## Context & objective

Make the smallest honest publication-smoke change in `submission/agent.py` by adding `PUBLICATION_SMOKE_NOTE` to the prompt text, while preserving the required `define()`, `train(data_dir, out_dir)`, and `eval(checkpoint_dir, data_dir, out_dir)` surface.

## Experiments cited

1. `submission/agent.py`
   - Question: does the submission surface stay intact after the smoke-note edit?
   - Outcome: supporting.
   - Evidence: the file still exposes `define()`, `train()`, and `eval()`, and the system prompt now includes `PUBLICATION_SMOKE_NOTE`.

2. Baseline-vs-candidate eval on train seeds `[10007, 10008, 10011]`
   - Question: does the smoke-note change alter measured rollout behavior?
   - Outcome: no measurable difference on the controlled slice.
   - Evidence: baseline and candidate both returned `mean_outcome_reward = 2.0`, `mean_llm_calls_per_rollout = 1.0`, `num_eval_rollouts = 3`, and `num_rollout_errors = 0`.

3. Local evaluator compatibility check
   - Question: can `submission/agent.py eval` run in this workspace without depending on unavailable local vLLM / Craftax packages?
   - Outcome: partially supporting, with caveats.
   - Evidence: the first attempt exposed a missing `vllm` binary at `/opt/nanohorizon-offline-venvs/teacher/bin/vllm`; the follow-up compare ran through `eval` with an in-memory deterministic Craftax stand-in and a mock OpenAI-compatible inference server because `craftax` was not installed in the workspace.

## Insights

1. The smoke-note edit is behaviorally neutral on the controlled repeated seed slice.
2. The repository checkout needed a small `evaluate_model` compatibility shim because the current evaluator signature does not accept the extra batch-size kwargs that were being forwarded by `submission/agent.py`.
3. The environment is not a full Craftax runtime, so the durable evidence here is an eval-path validation plus deterministic stand-ins, not a live Craftax benchmark score.

## Research artifacts produced

- Submission code: `submission/agent.py`
- Evaluation report: `reports/eval_report.md`
- Handoff notes: `findings.txt`

## Quality & validation

- Executed: `python -m py_compile submission/agent.py`
- Executed: baseline/candidate comparison on train seeds `[10007, 10008, 10011]` through `submission/agent.py eval`
- Result: baseline and candidate matched exactly on the controlled slice
- Not validated: live Craftax/vLLM rollout quality in this workspace, because the required local runtime pieces were absent

## Reproduction & handoff

- The smallest source change is in `submission/agent.py`.
- The compare used a deterministic in-memory Craftax stand-in and a mock OpenAI-compatible server to keep the `eval` path executable in this environment.
- If a future workspace has the full Craftax dependency stack and local vLLM binary available, rerun the same train-seed slice without the stand-ins for a live benchmark.
