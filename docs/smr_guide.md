# Using SMR to Improve NanoHorizon Baselines

[Synth Managed Research (SMR)](https://docs.usesynth.ai/managed-research) is an autonomous research platform that can iterate on training algorithms. This guide explains how to point SMR at a NanoHorizon track so it autonomously tries to beat the baseline.

## Quick Start

### 1. Install the MCP endpoint

```bash
# Codex
codex mcp add managed-research --url https://api.usesynth.ai/mcp

# Claude Code
claude mcp add --transport http managed-research https://api.usesynth.ai/mcp
```

### 2. Create a project for your track

Ask your MCP client:

> Create a new SMR project called "NanoHorizon SFT Improvement" with directed effort mode.

### 3. Configure the NanoHorizon repo

The ReportBench tasks are configured to clone the NanoHorizon repo (`https://github.com/synth-laboratories/nanohorizon.git`) as the project workspace. SMR workers get the full repo — all dependencies, scripts, configs, shared modules, and the Craftax environment bindings.

If setting up manually, tell your MCP client to use the NanoHorizon repo as the project's git workspace:

> Set the project repo to https://github.com/synth-laboratories/nanohorizon.git

The workers will then be able to run scripts directly (e.g., `./scripts/run_craftax_offline_qwen35_4b_1xa100_20min.sh`) and make commits with their changes.

### 4. Set the project spec

Tell SMR what the task is:

> **SFT**: "Improve the Craftax offline SFT training algorithm in offline_sft.py to achieve higher held-out mean reward than the baseline of 0.5. You may change the loss function, data selection, curriculum, or any algorithmic component. Do NOT change configs. Budget: 10 min on 1x A100 40GB. Model: Qwen/Qwen3.5-4B. Run via: ./scripts/run_craftax_offline_qwen35_4b_1xa100_20min.sh"

> **RLVR**: "Improve the Craftax GRPO training loop in rlvr.py to achieve higher reward than the baseline of 2.5. You may change the RL loss, reward shaping, rollout strategy, exploration. Budget: 10 min on 2x A100. Run via: ./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh"

> **Classic**: "Improve the Craftax-Classic PPO algorithm in classic.py to achieve higher mean reward. Budget: 1M steps, 100M params, JAX. Run via: ./scripts/run_classic_craftax_1m_modal.sh"

### 5. Trigger a run

Ask your MCP client:

> Trigger a directed effort run on the project.

Or use the frontend: navigate to your project page and click **New Run**.

## What SMR Does

SMR will:

1. **Plan** — The orchestrator reads the baseline code and task description, identifies potential algorithmic improvements
2. **Implement** — Workers modify the training algorithm file, following the constraints
3. **Test** — Workers run the training script on Modal, collect reward metrics
4. **Iterate** — Based on results, try different approaches (different losses, curricula, data strategies)
5. **Report** — Produce a summary of what was tried, what worked, and final metrics

## Monitoring

- **Frontend**: Go to the project page → click the run → see workers, tool calls, messages
- **Messages**: Send guidance mid-run via the message input (e.g., "focus on data augmentation" or "try a curriculum that starts with easier seeds")

## Tips

- **SFT track**: The throughput-optimized baseline already shows the FBC algorithmic ceiling. Try fundamentally different approaches — DPO, reward-weighted regression, self-play data generation.
- **RLVR track**: The current baseline does 3 GRPO iterations. Try different advantage estimation, KL penalties, replay buffers, or reward decomposition.
- **Classic track**: Pure RL on Craftax. Architecture changes, intrinsic motivation, and skill discovery are open territory.

## ReportBench Integration

These tracks have matching ReportBench task specs in `evals/smr/reportbench/`:

- `nanohorizon_sft` — SFT/FBC track
- `nanohorizon_rlvr` — RLVR track
- `nanohorizon_classic` — Classic RL track

To run as a formal ReportBench evaluation:

```bash
cd evals
python -m evals.smr.reportbench.runner --task nanohorizon_sft --backend-url http://localhost:8000
```
