# Video Validation Run

## Context & objective

The task was to package a minimal, reviewable Craftax leaderboard candidate in the prompt-opt lane using the smallest honest change that improves the todo/scratchpad strategy. The protected Craftax runtime surfaces were left unchanged; the candidate lives in a new prompt-opt config, a matching not-run record bundle, and a focused verifier.

## Experiments cited

1. `configs/craftax_prompt_opt_qwen35_4b_codex_video_validation_run.yaml`
   - Question: does a slightly sharper private todo contract help the model keep a short auditable plan during rollout?
   - Outcome: supporting as a candidate specification.
   - Evidence: seed prompt asks for three bounded todo items, explicit stale-target replacement, and a short final batch aligned with the first todo item.

2. `tests/test_codex_video_validation_run_candidate.py`
   - Question: does the candidate config and record bundle preserve the intended scratchpad wording and packaging shape?
   - Outcome: supporting as a structural verifier.
   - Evidence: the test pins the config text and checks the record bundle metadata, metrics, and launch command.

3. `records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_video_validation_run/`
   - Question: is the candidate packaged reproducibly for future execution?
   - Outcome: supporting for packaging, inconclusive for reward.
   - Evidence: `command.txt`, `metadata.json`, `metrics.json`, `notes.md`, `run_config.yaml`, and `system_info.json` are present with `candidate_not_run` / `not_run` markers.

4. `scripts/verify_video_validation_run.py`
   - Question: can the candidate be checked without relying on the broken optional cloud dependency path in `uv`?
   - Outcome: supporting.
   - Evidence: the script validates the config and record bundle with only stdlib plus `yaml`.

## Insights

1. The prompt-opt baseline already centralizes the todo-tool contract; the safest improvement is a narrower seed-prompt variant rather than touching the Craftax runtime.
2. Making the scratchpad explicitly short and visually inspectable is a reasonable fit for a video validation run, because the candidate is meant to stay auditable while still constraining loop behavior.
3. Structural validation is enough for this packaging task, but it does not measure leaderboard lift.

## Research artifacts produced

- Config: `configs/craftax_prompt_opt_qwen35_4b_codex_video_validation_run.yaml`
- Record bundle: `records/prompt_opt_1usd_gpt54_family/2026-04-11_codex_video_validation_run/`
- Verifier: `scripts/verify_video_validation_run.py`
- Handoff notes: `findings.txt`

## Quality & validation

- Ran `python scripts/verify_video_validation_run.py` successfully.
- The verifier confirmed the scratchpad wording, the not-run record bundle, and the candidate command.
- Not validated: any live Craftax rollout, GEPA search result, or reward delta.

## Reproduction & handoff

- Candidate command:
  - `NANOHORIZON_PROMPT_OPT_CONFIG=configs/craftax_prompt_opt_qwen35_4b_codex_video_validation_run.yaml ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`
- Open risk: the scratchpad wording may be neutral if it does not materially improve loop breaking under the real Craftax reward signal.
- Branch/PR: see the GitHub branch and PR created for this candidate.

