# NanoHorizon leaderboard candidate report

## Context & objective
This run targeted a single-file Craftax leaderboard candidate in `submission/agent.py`, with the expected final state being a reviewable commit plus a real PR. The submission surface was constrained to `submission/agent.py` and this report file.

Success on the algorithmic side meant a deterministic local planner that:
- prefers nearby trees, saplings, plants, and water
- uses `place_plant` when wood + sapling are available and the current tile is legal
- avoids loops and keeps exploring when nothing useful is visible
- can be evaluated on the official 20-seed split

## Experiments cited

### 1. Real harness probe
- Command: `python /workspace/submission/agent.py eval --out-dir /workspace/.nh_eval_final`
- Question: can the submission run against the live Craftax runtime in this workspace?
- Outcome: negative.
- Evidence: `/workspace/.nh_eval_final/result.json`
- Result: `primary_score = 0.0`, `num_rollout_errors = 20`
- Failure mode: every rollout failed before the first decision with `ImportError: craftax imports were not available. Install the classic dependency group. No module named 'craftax'`

### 2. Deterministic fake-run validation
- Command: a local monkeypatched call to `submission.agent.eval(...)` using `tests._craftax_fakes.make_test_runner`
- Question: does the candidate rollout loop execute cleanly and produce per-seed outputs on the official 20-seed split?
- Outcome: supporting.
- Evidence: `/workspace/.fake_eval/result.json`
- Result: `primary_score = 2.0`, `num_rollout_errors = 0`

## Insights

1. The current candidate is structurally sound as a deterministic local planner: the fake-run validation completed all 20 official eval seeds with zero rollout errors and a consistent reward trace.
2. In the deterministic fake environment, the planner reaches `collect_wood` and `collect_sapling` on every official eval seed, but it does not exceed a score of `2.0`. That caps the measured fake-run ceiling below the requested `>2.5` target.
3. The real leaderboard harness is not runnable in this workspace because the `craftax` dependency is missing. That blocks a true leaderboard score from being measured here, regardless of the planner logic.

## Research artifacts produced

### Environments
- Submission code: `submission/agent.py`
- Fake validation harness: `tests/_craftax_fakes.py`
- Temporary eval outputs: `/workspace/.fake_eval/` and `/workspace/.nh_eval_final/`

### Data
- Official eval seed manifest used by the candidate: `data/craftax/craftax_prompt_opt_eval20_seeds.json`
- Starter train seed manifest: `data/craftax/craftax_prompt_opt_starter_seeds.json`

### Models / checkpoints
- No model was trained in this run.
- The candidate is a heuristic planner only; there are no new weights or adapters.

## Quality & validation

### Fake-run per-seed metrics
All 20 official eval seeds were run through the deterministic fake runner:

| Seed | Reward | LLM calls | Achievements |
| --- | ---: | ---: | --- |
| 10001 | 2.0 | 4 | collect_sapling, collect_wood |
| 10002 | 2.0 | 3 | collect_sapling, collect_wood |
| 10004 | 2.0 | 3 | collect_sapling, collect_wood |
| 10005 | 2.0 | 4 | collect_sapling, collect_wood |
| 10006 | 2.0 | 3 | collect_sapling, collect_wood |
| 10009 | 2.0 | 4 | collect_sapling, collect_wood |
| 10010 | 2.0 | 3 | collect_sapling, collect_wood |
| 10012 | 2.0 | 3 | collect_sapling, collect_wood |
| 10013 | 2.0 | 4 | collect_sapling, collect_wood |
| 10015 | 2.0 | 4 | collect_sapling, collect_wood |
| 10016 | 2.0 | 3 | collect_sapling, collect_wood |
| 10017 | 2.0 | 4 | collect_sapling, collect_wood |
| 10019 | 2.0 | 4 | collect_sapling, collect_wood |
| 10020 | 2.0 | 3 | collect_sapling, collect_wood |
| 10021 | 2.0 | 4 | collect_sapling, collect_wood |
| 10022 | 2.0 | 3 | collect_sapling, collect_wood |
| 10023 | 2.0 | 4 | collect_sapling, collect_wood |
| 10024 | 2.0 | 3 | collect_sapling, collect_wood |
| 10025 | 2.0 | 4 | collect_sapling, collect_wood |
| 10026 | 2.0 | 3 | collect_sapling, collect_wood |

### Explicitly not validated
- Real Craftax rollouts on the actual runtime
- Any leaderboard score above `2.0`
- Any plant-placement or drink-seeking behavior beyond what the fake runner exposes

## Reproduction & handoff

The candidate entrypoint is `submission/agent.py`.

The fake-run validation that produced the artifact above was:

```bash
python - <<'PY'
import json
from pathlib import Path
import shutil

import submission.agent as agent
from tests._craftax_fakes import make_test_runner

class Monkey:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)

monkey = Monkey()

def fake_make_runner(*, kind: str, seed: int, render_mode):
    del kind
    return make_test_runner(monkey, seed=seed, render_mode=render_mode)

agent.make_runner = fake_make_runner
agent.achievement_names_from_state = lambda state: list(getattr(state, 'achievements', ()))
agent._can_capture_video = lambda: False

out_dir = Path('/workspace/.fake_eval')
if out_dir.exists():
    shutil.rmtree(out_dir)
result = agent.eval(Path('/workspace/.fake_eval_checkpoint'), Path('/workspace/data'), out_dir)
print(json.dumps({
    'primary_score': result['primary_score'],
    'num_rollout_errors': result['num_rollout_errors'],
}, indent=2, sort_keys=True))
PY
```

Observed run limits:
- real harness probe: blocked by missing `craftax` dependency
- deterministic fake runner: `primary_score = 2.0`

