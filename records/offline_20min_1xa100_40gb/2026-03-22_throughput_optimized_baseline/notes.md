# Throughput-optimized baseline

Throughput-focused iteration on the reference SFT/FBC baseline. Goal: reduce wall time while preserving reward uplift. This is now the **updated reference baseline** for the offline track.

## Best result

| Metric | Value |
|---|---|
| **Total wall time** | **8.2 min** (was ~20 min) |
| Base reward | 0.4 |
| Finetuned reward | **0.6** |
| **Delta** | **+0.2** |
| SFT examples | 23 |
| SFT tokens | 40,459 (29k prompt, 11k completion) |
| Mean sequence length | 1,759 |
| Training steps | 16 |
| Training loss | 0.35 |

### Achievement deltas (best run)

| Achievement | Base freq | Finetuned freq | Delta |
|---|---|---|---|
| collect_sapling | 0.10 | 0.20 | **+0.10** |
| collect_wood | 0.30 | 0.40 | **+0.10** |
| *(all other 20 achievements)* | 0.00 | 0.00 | 0.00 |

## Changes from original reference baseline

**Infrastructure (pure speed, no quality impact):**
- Teacher GPU: A10G → **A100-40GB** (~2x inference throughput)
- Rollout concurrency: 4 → **32** (fills vLLM batch queue)
- `max_inputs` on Modal teacher: 32 → **64** (matches vLLM `max-num-seqs`)
- `buffer_containers=1` on teacher (warm container, eliminates cold-start)
- `device_map="auto"` removed — explicit `.to("cuda")` instead
- FA2/SDPA attention backend
- `tf32=True`, `optim="adamw_torch_fused"`, `logging_steps=10`
- `dataloader_num_workers=4`, `persistent_workers=True`, `prefetch_factor=2`
- Data collator pads to multiple of 8 for tensor cores
- `per_device_train_batch_size`: 1 → 2

**Inference changes:**
- `thinking_budget_tokens`: 2000 → 512
- `max_tokens`: 2200 → 712

**Note:** Removing gradient checkpointing OOM'd on A100-40GB. It was re-enabled.

## Step-by-step timing (best run, 8.2 min total)

| Step | Time | % of total |
|---|---|---|
| Teacher rollout collection (32 rollouts) | **3.8 min** | **46%** |
| Modal SFT deploy + train (16 steps) | ~0.5 min | ~6% |
| Base eval (20 rollouts) | ~2 min | ~24% |
| Finetuned eval (20 rollouts) | ~2 min | ~24% |

### Rollout throughput progression

| Config | Rollouts | Wall time | Rate | GPU |
|---|---|---|---|---|
| Original (A10G, conc=4) | 32 | 10.3 min | 3.1/min | A10G |
| This baseline (A100, conc=32) | 32 | 3.1-3.8 min | 8.5-10.3/min | A100-40GB |
| Scaled run (A100, conc=32) | 240 | 10.6 min | 22.7/min | A100-40GB |
| Large run (A100, conc=32) | 800 | 52.6 min | 15.2/min | A100-40GB |

## Scaling experiments

We ran several experiments to test whether more/better data improves the baseline SFT:

| Run | Tokens | LR | Steps | Examples | Filter | Delta |
|---|---|---|---|---|---|---|
| **Best (this baseline)** | **40k** | **5e-5** | **16** | **23** | **reward≥1** | **+0.2** |
| More data, same LR | 346k | 5e-5 | 170 | 201 | reward≥1 | -0.05 |
| Few examples, high LR | 21k | 5e-4 | 100 | 12 | reward≥3 | +0.05 |
| Large data, high LR | 1.3M | 5e-4 | 100 | 727 | reward≥2 | -0.15 |

### Teacher reward distribution (800 rollouts, 2000 thinking, 16 steps)

| Reward | Count | % |
|---|---|---|
| 0 | 167 | 20.9% |
| 1 | 457 | 57.1% |
| 2 | 173 | 21.6% |
| 3 | 3 | 0.4% |

### Key findings

1. **Naive SFT does not scale with more data.** The best result came from the smallest run (23 examples, 40k tokens). Every attempt to scale up — more rollouts, higher reward filters, larger LR — either matched or regressed from the base model.

2. **The bottleneck is algorithmic, not infrastructure.** Rollout throughput is solved (22.7/min at scale). Training is fast. The issue is that behavior cloning on a narrow distribution (only collect_wood and collect_sapling) doesn't teach the student new capabilities.

3. **Teacher quality ceiling.** The 9B teacher with 8-16 steps rarely achieves more than 2 unique achievements. All training data comes from the same 2 behaviors. A stronger teacher (27B on A100-80GB) or more game steps would help data diversity.

4. **LR=5e-4 (10x, per TM blog LoRA guidance) was too aggressive** at scale. At 727 examples it caused catastrophic forgetting of collect_wood. The TM recommendation assumes a different regime (longer training, larger models).

5. **This is exactly the ceiling the benchmark is designed to expose.** Participants should change the training algorithm — DPO, reward-weighted loss, curriculum learning, advantage-filtered BC — not just add more data to naive SFT.

## Reproduce

```bash
./scripts/run_offline_training.sh
```

## Source artifacts

- Best run: `artifacts/offline_reference_20260322T235119Z/`
- Throughput v1 (12.1 min): `artifacts/offline_reference_20260322T210818Z/`
- 240-rollout scaled run: `artifacts/offline_reference_20260323T003535Z/`
- 800-rollout reward≥3 run: `artifacts/offline_reference_20260323T012810Z/`
- 727-example reward≥2 run: `artifacts/offline_reference_20260323T030549Z/`
