## Smoke Test B Candidate

Design intent:
- Keep the same prompt-optimization harness and GEPA + verifier flow from `prompt_opt_1usd_gpt54_family`, but reduce runtime load versus full eval by using a tighter smoke rollout budget and an explicit anti-loop, target-refresh contract.
- Strengthen loop recovery language in the seed prompt while preserving the private todo-tool contract and shared tool-calling constraints.

Verification and risk notes:
- `uv run python -m nanohorizon.shared.validate_record records/prompt_opt_1usd_gpt54_family/2026-04-09_smoke_test_b` was planned as the packaging verifier.
- Full GEPA execution is not started in this task to keep the workspace at smoke scope; reward impact is therefore unmeasured.
