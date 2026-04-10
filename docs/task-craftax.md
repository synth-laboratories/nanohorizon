# Task: Craftax Candidate

Candidate label: `Test Candidate`

Objective:
- implement a strong NanoHorizon Craftax submission candidate
- keep the shared Craftax harness surfaces stable unless a change is truly required
- avoid SFT and RL
- prefer the smallest honest harness change that improves the approach

Chosen strategy:
- custom harness optimization
- compact working-memory buffer that carries the latest subgoals and resource state across steps

Intended harness effect:
- reduce state loss between steps
- keep the model aware of the current goal, recent actions, and scarce resources
- expose the buffer through a stable prompt/context payload so the agent can reuse it without broader system changes

Stable surfaces to preserve when possible:
- `docs/task-craftax.md`
- `src/nanohorizon/craftax_core/http_shim.py`
- `src/nanohorizon/craftax_core/runner.py`
- `src/nanohorizon/craftax_core/metadata.py`
- `scripts/run_craftax_model_eval.sh`

