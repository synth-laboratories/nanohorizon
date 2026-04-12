# Bootstrap Verify Report

## Context & Objective

This run targeted the Craftax prompt-opt track in NanoHorizon. The objective was to make the smallest honest candidate change that could plausibly improve the `Qwen/Qwen3.5-4B` leaderboard candidate, keep the shared Craftax harness stable, and back the change with a repeated-seed comparison.

The candidate change is prompt-shaped rather than weight-based:

- keep the private three-item todo contract
- prefer short non-redundant 3-action batches by default
- avoid duplicate actions in a batch because the parser collapses repeats
- keep the batch ending adjacent to a useful target when safe

## Experiments Cited

1. `records/prompt_opt_1usd_gpt54_family/2026-04-12_bootstrap_verify/`
   - Question: does the candidate prompt outperform the baseline on a repeated-seed slice through the repo rollout path?
   - Outcome: supporting on the proxy metric, inconclusive on the official leaderboard metric.
   - Evidence: `metrics.json`, `notes.md`, `prompt_bundle.json`, and `reports/bootstrap_verify_proxy_eval.json`.
   - Result: repeated-seed proxy slice over seeds `10001`, `10010`, `10017`, and `10019` repeated twice. `mean_outcome_reward` tied at `2.0`, but `mean_native_env_reward_total` improved from `18.25` to `23.25` (`+5.0`).
   - Caveat: deterministic proxy policy and fake runner, not a live Qwen rollout.

2. `tests/test_bootstrap_verify_candidate.py`
   - Question: are the candidate config and prompt contract aligned with the short-batch / non-redundant guidance?
   - Outcome: supporting.
   - Evidence: the test passes and checks the config seed prompt and the centralized source contract.

3. `tests/test_craftax_core_contract.py` and `tests/test_craftax_interface.py`
   - Question: did restoring the Craftax HTTP surface accidentally break the existing contract?
   - Outcome: supporting.
   - Evidence: `11 passed` under `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml python -m pytest tests/test_bootstrap_verify_candidate.py tests/test_craftax_interface.py tests/test_craftax_core_contract.py`.

4. `uv run` with the default project environment
   - Question: can the normal project env be used directly for verification?
   - Outcome: negative.
   - Evidence: the inherited `cloud` dependency points at `file:///Users/joshpurtell/Documents/GitHub/synth-ai`, which does not exist in this workspace. Verification therefore used `--no-project` plus explicit packages.

## Insights

1. The Craftax shim had a real contract gap: `create_app` was missing, so the existing HTTP smoke tests could not import the module. Restoring a minimal FastAPI app factory fixed that without changing the rollout semantics.
2. The prompt-opt candidate improves the repo’s own rollout proxy when the prompt explicitly says the action batch must be non-redundant, not just short. That matters because repeated actions are collapsed before execution.
3. The official leaderboard metric remains unmeasured in this run. The repeated-seed result is therefore a proxy signal, not proof of a submission lift.
4. The default `uv` project environment is currently blocked by an absolute local dependency path. Any future verification in this workspace should keep using `--no-project` unless that dependency is normalized.

## Research Artifacts Produced

### Environments

- Proxy benchmark: inline `uv run --no-project` Python script exercising `nanohorizon.craftax_core.rollout.run_rollout`
- Smoke-test environment: `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml`

### Data

- Repeated proxy seeds: `10001`, `10010`, `10017`, `10019` repeated twice
- Baseline config: `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_bootstrap_verify.yaml`

### Models / Checkpoints

- No weights or checkpoints were trained or promoted in this run.
- The candidate is prompt/config based only.

## Quality & Validation

- Passed: `tests/test_bootstrap_verify_candidate.py`
- Passed: `tests/test_craftax_core_contract.py`
- Passed: `tests/test_craftax_interface.py`
- Passed: repeated-seed proxy comparison over the existing `run_rollout` path
- Not validated: live Craftax rollout reward, Modal execution, or a real OpenAI-compatible policy endpoint

Known failure modes and caveats:

- The proxy result is informative but not equivalent to a leaderboard run.
- The default `uv` project environment is currently blocked by an absolute file dependency path, so the verification path here is intentionally `--no-project`.

## Reproduction & Handoff

Commands:

```bash
PYTHONPATH=src:. uv run --no-project --with numpy --with fastapi --with httpx --with pyyaml python - <<'PY'
PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml python -m pytest tests/test_bootstrap_verify_candidate.py tests/test_craftax_interface.py tests/test_craftax_core_contract.py
```

Artifacts to inspect:

- `reports/bootstrap_verify_proxy_eval.json`
- `reports/bootstrap_verify_proxy_eval.md`
- `records/prompt_opt_1usd_gpt54_family/2026-04-12_bootstrap_verify/`
- `findings.txt`

Commit / PR:

- Local reviewable commit: `e85a62a0478f0e7a8d1b9c8a5a4f5d8f4d91aa0b`
- GitHub PR: `https://github.com/synth-laboratories/nanohorizon/pull/31`
