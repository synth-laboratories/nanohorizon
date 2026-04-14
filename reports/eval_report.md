# NanoHorizon submission smoke

## Context

Goal: make `submission/agent.py` publishable with a minimal, reviewable change, then verify the train-seed behavior before opening a PR.

## Validation

### Baseline attempt

Command:

```bash
python - <<'PY'
from pathlib import Path
from submission.agent import eval as run_eval
run_eval(Path('/tmp/nano_baseline_ckpt'), Path('/workspace/data/craftax'), Path('/tmp/nano_baseline_eval'))
PY
```

Result:

- failed immediately with `TypeError: evaluate_model() got an unexpected keyword argument 'target_action_batch_size'`
- this confirmed the current submission surface was stale against the live evaluator

### Candidate comparison

Command:

```bash
python - <<'PY'
# temporary local proxy used because the workspace has no craftax package or local vLLM binary
# baseline source loaded from `git show HEAD:submission/agent.py`
# candidate loaded from the edited working tree
PY
```

Result on the six train seeds from `data/craftax/craftax_prompt_opt_starter_seeds.json`:

- baseline mean outcome reward: `0.5`
- candidate mean outcome reward: `0.5`
- delta: `0.0`
- baseline rollouts: `6/6` successful
- candidate rollouts: `6/6` successful

## Caveats

- The workspace does not provide a usable local Craftax package or `vllm` binary, so the comparison used a deterministic local proxy runner and chat stub to exercise the submission code path.
- Because the prompt-only smoke note did not change the proxy behavior, the measured delta was neutral.
