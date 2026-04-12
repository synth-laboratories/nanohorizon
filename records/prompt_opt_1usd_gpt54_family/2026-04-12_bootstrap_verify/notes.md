Bootstrap-verify candidate for the Craftax prompt-opt track.

- Baseline config: `configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_bootstrap_verify.yaml`
- Source update: `src/nanohorizon/baselines/prompt_opt.py`

Proxy evaluation:

- Command: `PYTHONPATH=src uv run --no-project --with httpx --with numpy --with pyyaml python experiments/prompt_opt_bootstrap_verify/compare_proxy_eval.py`
- Repeated seeds: `10000`, `10001`, `10002`, `10003`, `10004`, `10005`
- Baseline mean outcome reward: `1.0`
- Candidate mean outcome reward: `2.0`
- Delta: `+1.0`
- Caveat: deterministic proxy only; no live Craftax model endpoint was available in this workspace.

Contract smoke:

- Command: `PYTHONPATH=src uv run --no-project --with pytest --with fastapi --with httpx --with numpy --with pyyaml python -m pytest tests/test_craftax_core_contract.py tests/test_craftax_interface.py`
- Result: `9 passed`

Branch / commit:

- `worker/run-5ff60fe2-d772-49ef-965b-c6b347e25794`
- `https://github.com/synth-laboratories/nanohorizon/pull/31`
