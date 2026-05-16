# NanoHorizon Submission Eval Report

## Context

This run updated only `submission/agent.py` to turn the submission into a stronger Craftax leaderboard candidate. The goal was to preserve the previous PR #103 behavior that reliably reached `collect_wood` and `collect_sapling`, then add one narrow legal rule for `place_plant` when the local state makes it valid.

## Experiments

1. PR #103 behavior recovery
- Question: what did the previous candidate actually do?
- Outcome: supporting.
- Evidence: GitHub PR #103 (`Craftax submission heuristic candidate`) reported `primaryScore: 1.75` with `collect_wood 20/20`, `collect_sapling 11/20`, `collect_drink 4/20`, and `max reward 3.0`.

2. Submission contract smoke
- Question: does `submission/agent.py` compile after the rewrite?
- Outcome: supporting.
- Evidence: `python -m py_compile submission/agent.py` passed after replacing the broken `evaluate_model` call path with a direct rollout collector and updating the prompt/defaults.

3. Live OpenAI eval attempt on a train seed
- Question: can the new candidate be run honestly against the real OpenAI chat-completions route on a train seed?
- Outcome: blocked in this workspace.
- Evidence: I attempted a real `gpt-4.1-nano` rollout through `https://api.openai.com/v1/chat/completions` using the repo’s local Craftax path and a train seed (`10000`, then `10007` in a reduced smoke). The run never completed because Craftax texture-cache initialization was killed in this environment before a result file was produced.

4. Runtime environment probe
- Question: is there a live local Craftax container already running?
- Outcome: negative.
- Evidence: `http://127.0.0.1:8903/health` and `http://127.0.0.1:8913/health` both returned connection refused.

## Insights

1. The previous candidate’s useful behavior was not a full policy solution; it was a narrow heuristic that mostly secured wood and sapling, with sparse drink coverage.
2. The current submission file needed more than a prompt tweak. It had a contract bug: the old `evaluate_model` call passed unsupported batch-size arguments.
3. The rewritten candidate now keeps the wood/sapling priority and adds an explicit `place_plant` pivot rule, which is the smallest legal expansion aligned with the task brief.
4. Local honest evaluation is blocked by Craftax initialization in this workspace, so score lift is currently unmeasured here.

## Research Artifacts Produced

### Environments

- `submission/agent.py` now uses the repo’s rollout collector directly for remote OpenAI-compatible inference.
- The workspace classic dependency group was materialized with `uv sync --group classic --frozen` to make Craftax importable.

### Data

- Train seeds default from `data/craftax/craftax_prompt_opt_starter_seeds.json`.
- A reduced one-seed smoke used `10007` locally; the full train manifest remains the intended evaluation set.

### Models / Checkpoints

- No model weights were trained or promoted in this run.
- The submission checkpoint now records the candidate focus list in `train()`.

## Quality & Validation

- Passed: `python -m py_compile submission/agent.py`
- Passed: direct runtime inspection of the prior PR #103 behavior from GitHub
- Failed: local honest Craftax rollout on `gpt-4.1-nano` could not complete because the Craftax texture-cache step was killed in this workspace
- Not validated: full train-seed score, held-out score, and any claim that primary score exceeds `2.5`

## Reproduction & Handoff

Relevant file:

- [`submission/agent.py`](/synth/state/.out/smr/projects/12afd559-ee0b-4583-a0d6-15ffa948e95a/runs/51621002-b468-4076-861e-0d960452c01d/workspace/submission/agent.py)

Suggested reproduction command once the Craftax runtime is healthy:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json, os, tempfile
import submission.agent as agent

base = Path(tempfile.mkdtemp(prefix='nh_eval_'))
data = base / 'data'
ckpt = base / 'ckpt'
out = base / 'out'
data.mkdir(); ckpt.mkdir(); out.mkdir()
(data / 'seeds.json').write_text(json.dumps({'seeds':[10007]}), encoding='utf-8')
agent.train(data, ckpt)
os.environ['NANOHORIZON_EVAL_INFERENCE_URL'] = 'https://api.openai.com/v1/chat/completions'
os.environ['NANOHORIZON_EVAL_REQUEST_MODEL'] = 'gpt-4.1-nano'
print(agent.eval(ckpt, data, out)['primary_score'])
PY
```

Open risk:

- The current workspace cannot complete a real Craftax rollout locally, so the submission is code-complete but not score-verified here.
