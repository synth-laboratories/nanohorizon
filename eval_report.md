# NanoHorizon Submission Eval Report

## Context

This run updated only `submission/agent.py` to make the submission candidate more explicit about early-game Craftax priorities: collect sapling and wood first, then pivot to `place_plant` or `collect_drink`. The file also had to remain a valid train/eval entrypoint and write `eval_summary.json`.

## Experiments

1. `submission/agent.py` contract smoke
   - Question: does the submission file import cleanly and does `eval()` write an evaluation summary?
   - Outcome: supporting after fixes.
   - Evidence: the smoke exposed missing imports in `submission/agent.py` (`asyncio`, `LocalVLLMEvalConfig`, `local_vllm_server`, `collect_rollouts_concurrently_with_summary`), which were added. The eval path now reaches summary writing.

2. Live one-seed eval on train seed `10000`
   - Question: can the candidate produce more than the two-achievement baseline on a real rollout?
   - Outcome: negative so far.
   - Evidence: live eval through the direct managed route at `/chat/completions` returned `primary_score: 2.0` with achievements `["collect_sapling", "collect_wood"]` and `llm_call_count: 20`.

3. Managed inference routing attempts
   - Question: which inference surface is actually usable for the live eval branch?
   - Outcome: mixed.
   - Evidence:
     - direct route root and `/v1/chat/completions` returned 404 `modal-http: invalid function call`
     - proxy base URL returned 401 `API key not found`
     - direct route with `/chat/completions` worked and produced a valid `eval_summary.json`

## Quality Notes

- The candidate no longer has the earlier `NameError`/import-path failures in the eval path.
- The current prompt still appears capped at two achievements on seed `10000`, so the target `primaryScore > 2.5` remains unproven.
- The live result is still useful as a reproducible baseline for the next iteration.

## Reproduction

```bash
python - <<'PY'
from pathlib import Path
import json, os, tempfile
import submission.agent as agent

base = Path(tempfile.mkdtemp(prefix='nh_eval_'))
data = base / 'data'
ckpt = base / 'ckpt'
out = base / 'out'
data.mkdir(); ckpt.mkdir(); out.mkdir()
(data / 'seeds.json').write_text(json.dumps({'seeds':[10000]}), encoding='utf-8')
agent.train(data, ckpt)
os.environ['NANOHORIZON_EVAL_INFERENCE_URL'] = os.environ['SMR_MANAGED_INFERENCE_DIRECT_ROUTE'].rstrip('/') + '/chat/completions'
os.environ['NANOHORIZON_EVAL_API_KEY'] = ''
print(agent.eval(ckpt, data, out)['primary_score'])
PY
```

