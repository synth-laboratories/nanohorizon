# Bootstrap Verify Proxy Eval

- Baseline config: `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_bootstrap_verify.yaml`
- Source update: `src/nanohorizon/baselines/prompt_opt.py`

Proxy evaluation:

- Repeated seeds: `10001`, `10001`, `10010`, `10010`, `10017`, `10017`, `10019`, `10019`
- Baseline mean outcome reward: `2.0`
- Candidate mean outcome reward: `2.0`
- Baseline mean native env reward total: `18.25`
- Candidate mean native env reward total: `23.25`
- Delta native env reward total: `+5.0`
- Caveat: deterministic proxy policy routed through the repo rollout loop, not a live Qwen rollout

Contract smoke:

- Command: `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml python -m pytest tests/test_bootstrap_verify_candidate.py tests/test_craftax_interface.py tests/test_craftax_core_contract.py`
- Result: `11 passed`
