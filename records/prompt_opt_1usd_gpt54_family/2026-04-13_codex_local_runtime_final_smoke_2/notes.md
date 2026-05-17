Proxy smoke for the Local Runtime Final Smoke 2 prompt-opt candidate.

Baseline: `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`.
Candidate: `configs/craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml`.
Evaluation path: `nanohorizon.shared.craftax_data.collect_rollouts_concurrently_with_summary` with a deterministic proxy rollout responder over the held-out starter eval seeds.
Result: the wood-first bootstrap prompt improved proxy mean outcome reward and kept the early-game resource ladder explicit.
