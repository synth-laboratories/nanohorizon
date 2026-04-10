# Craftax Test Candidate

## Context & objective

The objective for this run was to produce a reviewable NanoHorizon Craftax candidate with the smallest honest change that plausibly improves custom-strategy leaderboard performance, while keeping the shared Craftax harness surfaces stable unless a change was narrowly justified.

The candidate chosen here is a compact working-memory buffer:
- the rollout prompt now carries forward a bounded summary of recent plan, actions, state, reward, and achievements
- the prompt-opt candidate explicitly asks the policy to treat that information as a private three-item working-memory buffer
- the rollout request now forwards a `working_memory_capacity` value instead of relying on an implicit default

No live Craftax score run was executed in this task, so reward impact remains unmeasured.

## Experiments cited

1. `tests/test_craftax_core_contract.py`
   - Question: does the shared Craftax rollout surface actually expose prior-turn memory to the next prompt?
   - Outcome: supporting.
   - Evidence: the new regression test stubs a two-turn rollout and asserts the follow-up prompt contains `Working memory from previous turns:` plus the earlier plan, compact state summary, and achievements.

2. `tests/test_craftax_core_runtime_guarantees.py`
   - Question: did the bounded-memory change preserve the rollout contract and existing runtime guarantees?
   - Outcome: supporting.
   - Evidence: the existing runtime-guarantee coverage still passes after the buffer change, including checkpointing, deterministic replay, and action parsing behavior.

3. `tests/test_codex_durable_intent_candidate.py`
   - Question: did the durable-intent prompt candidate remain structurally valid after the buffer vocabulary update?
   - Outcome: supporting.
   - Evidence: the candidate config and source string checks still pass.

4. `tests/test_codex_todo_refresh_gate_candidate.py`
   - Question: did the updated prompt-opt candidate and record bundle remain consistent after making the buffer capacity explicit?
   - Outcome: supporting.
   - Evidence: the config now includes the working-memory buffer wording and `working_memory_capacity: 4`, and the record bundle remains marked as not run.

5. `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
   - Question: is the candidate reproducible and self-describing for future runs?
   - Outcome: supporting.
   - Evidence: the bundle now records the updated config, notes, command, metrics placeholder, and a stable `run_config.yaml`.

## Insights

1. A compact working-memory buffer is the smallest harness-side change that still gives later turns explicit access to prior subgoals and resource state.
2. The shared rollout surface stayed honest: the tool schema and action catalog were unchanged; only the prompt context and turn metadata were augmented.
3. Making `working_memory_capacity` explicit in the rollout request keeps the candidate reproducible and avoids relying on hidden defaults.
4. The verifier results show the change is structurally sound, but there is still no measured leaderboard improvement in this run.

## Research artifacts produced

### Environments

- Shared Craftax harness:
  - `src/nanohorizon/craftax_core/metadata.py`
  - `src/nanohorizon/craftax_core/rollout.py`
- Prompt-opt candidate shaping:
  - `src/nanohorizon/baselines/prompt_opt.py`
- Candidate config:
  - `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
- Candidate record bundle:
  - `records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate/`

### Data

- No new training or rollout dataset was generated in this task.
- The candidate continues to reference `../data/craftax/craftax_prompt_opt_starter_seeds.json` through the prompt-opt config.

### Models / checkpoints

- No model weights were trained, promoted, or checkpointed.
- The candidate remains a prompt-level modification around `Qwen/Qwen3.5-4B`.

## Quality & validation

- Passed: `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_durable_intent_candidate.py tests/test_codex_todo_refresh_gate_candidate.py`
- Passed: `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with uvicorn --with pillow --with numpy --with craftax python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_core_runtime_guarantees.py`
- Passed: `PYTHONPATH=src uv run --no-project python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_durable_intent_fix`
- Passed: `PYTHONPATH=src uv run --no-project python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-07_codex_todo_refresh_gate`
- Known blocker during full-project `uv run`: the repo still contains a stale local `synth-ai` file dependency in optional cloud metadata, so project-discovering `uv` commands tried to resolve `file:///Users/joshpurtell/Documents/GitHub/synth-ai`. The validation above worked around that by using `--no-project`.
- Known non-validation: no live Craftax or Modal reward run was executed, so the leaderboard effect is still a hypothesis.

## Reproduction & handoff

- Candidate branch: `test-candidate-final`
- GitHub PR: https://github.com/synth-laboratories/nanohorizon/pull/19
- Implementation commit: `f9484e8` (`Add compact Craftax working memory`)
- Current reviewable head: the published tip of `test-candidate-final`
- Main files to inspect:
  - `src/nanohorizon/craftax_core/metadata.py`
  - `src/nanohorizon/craftax_core/rollout.py`
  - `src/nanohorizon/baselines/prompt_opt.py`
  - `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
  - `tests/test_craftax_core_contract.py`
  - `tests/test_codex_todo_refresh_gate_candidate.py`
- Reproduction commands:
  - `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml python -m pytest tests/test_codex_durable_intent_candidate.py tests/test_codex_todo_refresh_gate_candidate.py`
  - `PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with httpx --with fastapi --with uvicorn --with pillow --with numpy --with craftax python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_core_runtime_guarantees.py`
- Open risk:
  - the candidate has not been compared against a live Craftax baseline score, so any leaderboard benefit is still unproven
