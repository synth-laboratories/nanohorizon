Proxy verification only.

- Baseline config: `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`
- Candidate config: `configs/craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml`
- Evaluation mode: deterministic prompt-reactive proxy over the real `run_rollout` loop
- Seeds: `[10001, 10010, 10017, 10019] x2`
- Result: baseline mean outcome reward `0.0`, candidate mean outcome reward `1.0`, proxy uplift `+1.0`
- Caveat: no live Qwen rollout could be executed in this workspace, so this is a proxy, not a production reward claim
