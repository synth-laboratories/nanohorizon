# NanoHorizon Craftax Task

Candidate label: `Video E2E Retry`

Objective:
- Build the strongest small honest NanoHorizon candidate inside this repo.
- Keep the Craftax harness surfaces stable unless a change is truly required.
- Prefer verifier-driven review and compact tool/context shaping over broad system edits.
- Do not use SFT or RL.

Strategy:
- Add a compact todo/scratchpad primitive that helps the agent track subgoals while optimizing the harness.

Constraints:
- Work only in the NanoHorizon repository.
- Keep `docs/task-craftax.md`, `src/nanohorizon/craftax_core/http_shim.py`, `src/nanohorizon/craftax_core/runner.py`, `src/nanohorizon/craftax_core/metadata.py`, and `scripts/run_craftax_model_eval.sh` stable unless a change is truly required.
- Use verifier feedback before declaring the candidate ready.

