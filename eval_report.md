# NanoHorizon Submission Eval Report

## Context & objective

Candidate: `codex-20260418T105003Z`

Goal for this run:
- strengthen the single-file Craftax submission in `submission/agent.py`
- keep the diff inside the allowed submission surface
- run the repo's real `train(...)` then `eval(...)` path on train seeds
- capture honest score evidence before publishing

## Changes made

- Replaced the generic Craftax prompt with a prioritized resource-ladder prompt.
- Made the policy prefer a private 3-item checklist, short 3-4 action batches, and explicit early-game priorities.
- Wired the submission `eval(...)` path to drive rollouts directly so the batch-size settings in the checkpoint are actually honored.
- Kept edits confined to `submission/agent.py`.

## Validation attempts

### Syntax sanity

Command:

```bash
python -m py_compile submission/agent.py
```

Result:

- passed

### Honest eval attempt on a single train seed

Command:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
JAX_PLATFORM_NAME=cpu \
NANOHORIZON_SUBMISSION_MAX_STEPS=1 \
NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS=64 \
NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS=0 \
NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE=4 \
NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE=3 \
NANOHORIZON_EVAL_INFERENCE_URL='https://api.openai.com/v1/chat/completions' \
NANOHORIZON_EVAL_REQUEST_MODEL='gpt-5.4-mini' \
NANOHORIZON_EVAL_API_KEY="$OPENAI_API_KEY" \
python - <<'PY'
from pathlib import Path
import json, shutil
from submission.agent import train, eval

repo = Path('.').resolve()
base_data = repo / 'data'
ckpt_dir = Path('/tmp/nanohorizon_submission_ckpt')
seed_dir = Path('/tmp/nanohorizon_submission_seed_10007')
out_dir = Path('/tmp/nanohorizon_submission_eval_10007')

for path in (ckpt_dir, seed_dir, out_dir):
    if path.exists():
        shutil.rmtree(path)
seed_dir.mkdir(parents=True)
(seed_dir / 'seeds.json').write_text(json.dumps({'seeds': [10007]}), encoding='utf-8')

train(base_data, ckpt_dir)
result = eval(ckpt_dir, seed_dir, out_dir)
print(json.dumps(result, indent=2, sort_keys=True))
PY
```

Observed output:

```text
Loading Craftax textures from cache.
Textures successfully loaded from cache.
/bin/bash: line 21: 20067 Killed
```

Interpretation:

- the submission code itself compiles
- the honest rollout path could not complete in this workspace because the process was killed by the environment during Craftax startup / rollout execution
- I therefore do not have a real train-seed score to report from this run

### Remote runtime checks

- `modal_synth` is importable in the workspace, but the underlying `modal` client is not installed here, so I could not move the eval to a Modal sandbox from this environment.
- The recorded Modal rollout URLs from prior records returned `404 modal-http: invalid function call` when probed directly, so they were not usable as a fallback inference endpoint.

## Reproduction notes

- Source of the candidate: `submission/agent.py`
- The commit should remain reviewable because only the submission file changed.
- To reproduce the current failure mode, rerun the one-seed command above in this workspace.
- The remaining blocker is environment capacity for running the full Craftax rollout path, not a syntax error in the submission file.
