# NanoHorizon Craftax Task

Candidate label: `Pipeline Fix E2E`
Primary strategy: `Todo Tool`

Objective:
- Build a strong NanoHorizon leaderboard candidate inside the NanoHorizon repo.

Constraints:
- Keep the shared Craftax harness surfaces stable unless a change is truly required.
- Use verifier feedback before declaring the candidate ready.
- Keep any extra tooling narrow, reviewable, and directly justified by the objective.

Stable surfaces to preserve:
- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

Pipeline contract:
- `GET /health`
- `GET /task_info`
- `POST /rollouts`
- `/rollout` remains a compatibility alias

Approach:
- Keep a compact TODO scratchpad in the candidate metadata so the task plan stays legible in every surface.
- Prefer explicit, stable response builders over ad hoc prompt or rollout branching.
- Treat verification as a gate before readiness.

Smoke check:
- `bash scripts/run_craftax_model_eval.sh --format text`

