# Craftax Candidate Run Report

## Context & Objective

This run targeted the NanoHorizon Craftax leaderboard track. The intended goal was to make the smallest honest candidate change that could improve Craftax outcomes over the baseline, while keeping the protected Craftax harness surfaces stable:

- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

Success required a reviewable commit, a baseline-vs-candidate evaluation with repeated seeds, and a real GitHub PR. Because the intended Modal-backed Qwen lane was blocked, I ran a proxy evaluation on the existing prompt-opt path and reported it honestly below.

## Experiments Cited

1. `experiments/craftax_candidate/configs/baseline_small_slice.yaml`
   - Question: what score does the baseline prompt achieve on a small repeated-seed slice?
   - Outcome: no uplift signal; held-out score stayed at `0.0`.
   - Evidence: `experiments/craftax_candidate/results/baseline_small_slice/metrics.json`, `base_eval_summary.json`, `best_eval_summary.json`, `notes.md`.

2. `experiments/craftax_candidate/configs/candidate_small_slice.yaml`
   - Question: does the candidate todo-refresh prompt improve the same slice?
   - Outcome: no uplift signal; held-out score stayed at `0.0`.
   - Evidence: `experiments/craftax_candidate/results/candidate_small_slice/metrics.json`, `base_eval_summary.json`, `best_eval_summary.json`, `notes.md`.

3. `experiments/craftax_candidate/results/proxy_slice_summary.json`
   - Question: can the two runs be compared directly on the same proxy slice?
   - Outcome: negative for uplift; candidate minus baseline primary score was `0.0`.
   - Evidence: comparison summary plus the two run output directories above.

4. `experiments/craftax_candidate/experiment_log.txt`
   - Question: what happened operationally during the run?
   - Outcome: supporting evidence for the executed commands, results, and blocked verification lane.
   - Evidence: timestamped entries for the baseline and candidate proxy runs.

## Insights

1. The candidate prompt did not demonstrate measurable improvement on the proxy slice. Both baseline and candidate finished at `0.0` primary score on the same 4-seed evaluation slice.
2. The proxy slice is reproducible, but it is not the intended leaderboard verification. It used `run_training(...)` from `src/nanohorizon/baselines/prompt_opt.py` with `direct://local` Craftax rollouts and the OpenAI API as the policy backend because Modal-backed Qwen evaluation was blocked in this workspace.
3. The run is still useful as evidence that the candidate did not regress the proxy slice, but it does not support a claim of uplift.

## Research Artifacts Produced

### Environments

- Eval path: `run_training(...)` in `src/nanohorizon/baselines/prompt_opt.py`
- Local execution mode: `container_url=direct://local`
- Proxy policy backend: `inference_url=https://api.openai.com`, `request_model=gpt-5.4-mini`
- Small-slice settings: `num_train_seeds=2`, `num_eval_seeds=4`, `max_metric_calls=2`
- The initial `uv run --group modal` path failed because the workspace lacked Modal auth and the project sync also referenced a stale file:// dependency path; the no-project invocation worked around the sync issue, but not the missing Modal token for the intended lane.

### Data

- Seed source: `data/craftax/craftax_prompt_opt_starter_seeds.json`
- The proxy slice used the repo-native prompt-opt split logic with the 4 held-out seeds selected by the configs above.

### Models / Checkpoints

- No weights were trained or promoted in this run.
- The candidate remains a prompt-shaping change, not a finetune or RL artifact.

## Quality & Validation

- Baseline proxy run:
  - `primary_score=0.0`
  - `bootstrap_score=0.0`
  - `score_delta=0.0`
- Candidate proxy run:
  - `primary_score=0.0`
  - `bootstrap_score=0.0`
  - `score_delta=0.0`
- Measured delta on the shared slice:
  - `candidate_minus_baseline_primary_score=0.0`
  - `uplift_demonstrated=false`
- What was explicitly not validated:
  - the intended Modal-backed Qwen lane
  - leaderboard submission quality on `Qwen/Qwen3.5-4B`
  - any claim of generalization beyond the proxy slice

## Reproduction & Handoff

- Baseline command:
  - `PYTHONPATH=src uv run --no-project --with modal --with httpx --with pyyaml --with gepa --with numpy python - <<'PY' ... run_training(config_path=experiments/craftax_candidate/configs/baseline_small_slice.yaml, container_url=direct://local, inference_url=https://api.openai.com, request_model=gpt-5.4-mini) ... PY`
- Candidate command:
  - `PYTHONPATH=src uv run --no-project --with modal --with httpx --with pyyaml --with gepa --with numpy python - <<'PY' ... run_training(config_path=experiments/craftax_candidate/configs/candidate_small_slice.yaml, container_url=direct://local, inference_url=https://api.openai.com, request_model=gpt-5.4-mini) ... PY`
- Workspace push commit:
  - `1f1f3c386f760c8afc397f874f7d640287825a2e`
- GitHub PR:
  - `https://github.com/synth-laboratories/nanohorizon/pull/35`
- Output directories:
  - `experiments/craftax_candidate/results/baseline_small_slice/`
  - `experiments/craftax_candidate/results/candidate_small_slice/`
- Comparison artifact:
  - `experiments/craftax_candidate/results/proxy_slice_summary.json`
- Open risks:
  - Modal-backed verification still needs auth and the intended Qwen track was not exercised.
  - The proxy slice is too weak to claim leaderboard uplift.
