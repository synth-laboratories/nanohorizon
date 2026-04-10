# Craftax Test Candidate

## Context & objective

Implement the smallest honest NanoHorizon Craftax improvement for the `Test Candidate` submission. The constraint was to keep the shared Craftax harness surfaces stable unless a change was strictly necessary, avoid SFT/RL, and use a compact custom harness optimization strategy that improves long-horizon decision making.

The chosen strategy is a narrow working-memory buffer: each rollout turn now carries forward a compact summary of recent plan, actions, state, reward, and achievements so the next prompt can condition on prior progress without changing the tool contract or HTTP rollout surface.

## Experiments cited

1. `src/nanohorizon/craftax_core/metadata.py`
   - Question: can the harness keep a compact, bounded memory of prior turns without changing the shared rollout contract?
   - Outcome: supporting.
   - Evidence: `compact_state_summary`, `CraftaxWorkingMemoryEntry`, and `WorkingMemoryBuffer` were added to store and render recent turn summaries in a bounded deque.

2. `src/nanohorizon/craftax_core/rollout.py`
   - Question: can follow-up Craftax prompts receive the previous turns' working memory without altering the tool schema?
   - Outcome: supporting.
   - Evidence: `_observation_prompt` now injects rendered working memory text, `run_rollout` threads a bounded `WorkingMemoryBuffer`, and turn/metadata outputs include the memory snapshot.

3. `tests/test_craftax_core_contract.py`
   - Question: does the second prompt actually include the rendered memory from the earlier turn?
   - Outcome: supporting.
   - Evidence: the new regression test stubs a two-turn rollout and asserts the follow-up prompt contains `Working memory from previous turns:` plus the first turn's plan, compact state summary, and achievements.

4. `docs/task-craftax.md`
   - Question: is the candidate strategy documented in the task-facing note without expanding the contract?
   - Outcome: supporting.
   - Evidence: a short candidate note records that `Test Candidate` uses a compact working-memory buffer while keeping the tool schema and rollout surface stable.

## Insights

1. A compact working-memory buffer is the smallest harness-side change that still gives later turns explicit access to prior subgoals and resource state.
2. The change stays honest to the original Craftax contract because it only augments prompt context and result metadata; it does not alter the tool schema, action catalog, or rollout endpoints.
3. The regression test is the most important evidence here because it proves the memory actually reaches the follow-up prompt instead of only existing in metadata.
4. Live reward improvement remains unmeasured in this run, so the candidate should be read as a structurally improved baseline rather than a scored win.

## Research artifacts produced

- Code:
  - `src/nanohorizon/craftax_core/metadata.py`
  - `src/nanohorizon/craftax_core/rollout.py`
  - `src/nanohorizon/craftax_core/__init__.py`
- Tests:
  - `tests/test_craftax_core_contract.py`
- Task note:
  - `docs/task-craftax.md`
- Handoff log:
  - `findings.txt`

## Quality & validation

- Structural validation added: the new contract test checks that the second rollout prompt includes the rendered working-memory block from turn 0.
- Executed and passed: `uv run --no-project --with pytest --with fastapi --with httpx --with pillow --with pyyaml --with numpy python -m pytest tests/test_craftax_core_contract.py -q`
- Known failure mode: if the renderer emits sparse state text, the compact summary may be short; the buffer still preserves the raw observation and the turn-level metadata for inspection.

## Reproduction & handoff

- Candidate branch: `test-candidate-final`
- Main implementation entrypoints:
  - `src/nanohorizon/craftax_core/rollout.py`
  - `src/nanohorizon/craftax_core/metadata.py`
- Intended verification command:

- Executed verification command:

```bash
uv run --no-project --with pytest --with fastapi --with httpx --with pillow --with pyyaml --with numpy python -m pytest tests/test_craftax_core_contract.py -q
```

- Open risk: the harness-side memory may help planning, but it has not been compared against a live baseline score in this run.
