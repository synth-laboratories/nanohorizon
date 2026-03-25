# NanoHorizon

<p align="center">
  <img src="assets/craftax_gpt54nano.gif" width="200" alt="Real NanoHorizon Craftax rollout captured after each in-game action using gpt-5.4-nano via the OpenAI API" />
</p>

**Fast, cheap iteration for long-horizon and classic RL agents.** Improve agents under hard time, hardware, and budget caps with reproducible runs, pinned metrics, and public records anyone can verify.

*GIF: real NanoHorizon Craftax rollout captured after each in-game action with `gpt-5.4-nano` via the OpenAI API.*

[Join the Synth Discord](https://discord.gg/cjfAMcCZef)

---

### What this benchmark is about

NanoHorizon is about **changing the training algorithm**, not tuning hyperparameters. Each track gives you a single Python file containing a baseline algorithm (SFT, GRPO, GEPA). Your job is to write a better one — a different loss function, a smarter data selection strategy, a new rollout scheme, a curriculum, whatever you think will work. The config exists to set resource caps, not as the edit surface.

There are two ways to win:

1. **Higher score** — get more reward out of the same time and hardware budget.
2. **Higher throughput** — get the same reward in less time, or more lift per minute of training. A method that reaches the current best score 3x faster is a real contribution.

Base model target for the Craftax tracks: `Qwen/Qwen3.5-4B` unless a track doc states otherwise.

## Leaderboard

Status: **three checked-in training reference baselines** plus a checked-in pure 4B Modal eval baseline — all reproducible from `records/`.

| Track | Rank | Run | Score | Summary | Record |
| --- | ---: | --- | --- | --- | --- |
| `offline_20min_1xa100_40gb` | 1 | `modal_4b_nochange_baseline` | `0.7` | Pure no-change 4B baseline via Modal inference on the 20 held-out seeds; includes raw rewards and 22-achievement frequencies | [info](records/offline_20min_1xa100_40gb/2026-03-22_modal_4b_nochange_baseline/) |
| `offline_20min_1xa100_40gb` | 2 | `throughput_optimized_baseline` | `0.6` | FBC on 4B with 9B teacher; +0.2 delta in **8.2 min** (was ~20 min); 40k tokens, 23 examples, A100 teacher at 32x concurrency; naive SFT scaling experiments show this is the algorithmic ceiling | [info](records/offline_20min_1xa100_40gb/2026-03-22_throughput_optimized_baseline/) |
| `offline_20min_1xa100_40gb` | 3 | `reference_baseline` | `0.5` | Craftax FBC on 4B with 9B teacher; held-out compare gives `+0.2` reward delta | [info](records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) |
| `rlvr_20min_2xa100_40gb` | 1 | `throughput_baseline` | `2.5` | 3 GRPO iterations, 69 rollouts in 18.7 min, +0.25 reward lift from bootstrap (2.25 → 2.5); V1 engine + CUDA graphs + local Craftax + internal networking; 470 tok/s steady-state, 1451 tok/s peak | — |
| `rlvr_20min_2xa100_40gb` | 2 | `reference_baseline` | `0.0` | Topology validation only (enforce-eager, 13 tok/s) | [info](records/rlvr_20min_2xa100_40gb/2026-03-21_reference_baseline/) |
| `prompt_opt_1usd_gpt54_family` | 1 | `reference_baseline` | `0.35` | GEPA prompt search on 4B under honest Craftax reward accounting; 20-rollout held-out probe regressed `-0.25` from the seed prompt | [info](records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline/) |

`classic` is now an official track, but it does not have a checked-in reference record yet.

New rows: add `records/<track>/<YYYY-MM-DD>_<name>/` and update this table in the **same PR**.

## Change And Run

Each track has **one Python file** containing the training algorithm and **one shell script** to run it. Change the algorithm, run the script, check your score.

### RLVR track

1. Change the training algorithm in [rlvr.py](src/nanohorizon/baselines/rlvr.py)
2. Run: `./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh`

Budget: `20` minutes on `2x A100 40GB` · Model: `Qwen/Qwen3.5-4B`

This is the first-track long-horizon RL lane: LLM-in-the-loop, Craftax, and the existing repo container/runtime abstractions.

<details>
<summary>Config override (for smoke tests, not the main edit surface)</summary>

```bash
NANOHORIZON_RLVR_CONFIG=configs/craftax_rlvr_qwen35_4b_validation_smoke.yaml \
./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh
```
</details>

What that bash script handles for you:

- builds and uploads the Craftax Python runtime into Modal
- starts a Synth-compatible Craftax HTTP shim in the same Modal app
- starts a clustered learner-plus-inference runtime and forwards a stable inference URL from the inference worker
- runs grouped online Craftax rollouts
- runs the GRPO-style LoRA update loop from `src/nanohorizon/baselines/rlvr.py`
- reloads adapters into inference between rollout waves
- writes periodic eval, final eval, and record-bundle outputs

### Classic track

1. Run locally: `./scripts/run_classic_craftax_1m.sh --train`
2. Or run on Modal: `./scripts/run_classic_craftax_1m_modal.sh`

Task: Craftax-Classic via `craftax` · Regime: `1M` · Policy: random-init RL under `100M` params

This is a separate classic-RL lane. It does not use the in-repo Craftax runtime, container abstractions for environment interaction, or OpenAI-compatible policy serving. The first baseline is a JAX PPO-RNN setup for `Craftax-Classic-Symbolic-v1` at `1M` frames, followed by a fast parallelized eval pass.

### Offline (SFT) track

1. Change the training algorithm in [offline_sft.py](src/nanohorizon/baselines/offline_sft.py)
2. Run: `./scripts/run_offline_training.sh`

Budget: `20` minutes on `1x A100 40GB` · Student: `Qwen/Qwen3.5-4B` · Teacher: `Qwen/Qwen3.5-9B`

What that bash script handles for you:

- starts the local Craftax runtime
- runs teacher inference on Modal
- runs SFT on Modal
- collects rollouts locally against the local Craftax runtime
- runs held-out base vs finetuned evals locally against the local Craftax runtime
- writes the comparison summary

Default offline path:

- no tunnel
- local Craftax runtime at `http://127.0.0.1:8903`
- remote Modal inference for teacher, base student, and finetuned student
- remote Modal SFT for the student LoRA

Reference record to compare against:

- [2026-03-20_reference_baseline](records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline)
- score: `0.5`
- reward delta over base: `+0.2`

Pure no-change 4B Modal baseline:

- [2026-03-22_modal_4b_nochange_baseline](records/offline_20min_1xa100_40gb/2026-03-22_modal_4b_nochange_baseline)
- mean reward over 20 held-out seeds: `0.7`
- checked-in fields include raw rewards and full-Craftax achievement frequencies

Known issue / current focus for submissions:

- Simply scaling teacher volume does not reliably improve score.
- Recent quality-filtered runs still admitted mostly `collect_wood` traces,
  with weak coverage of stronger target behaviors like `eat_cow`,
  `defeat_zombie`, and `defeat_skeleton`.
- For this SFT track, improving high-quality data filtering and trace
  selection should be treated as a primary optimization target.

### Prompt-opt track

1. Change the search/optimization algorithm in [prompt_opt.py](src/nanohorizon/baselines/prompt_opt.py)
2. Run: `./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh`

Budget: `$1` optimizer spend (GPT-5.4 family) · Model: `Qwen/Qwen3.5-4B`

### Managed Research (SMR) — agent-assisted progress (optional)

Use **[Synth Managed Research](https://docs.usesynth.ai/managed-research/quickstart)** to drive **managed runs** against this repo: MCP + API key, project onboarding, optional GitHub org link, **starting data** uploads, and a written **spec** that tells agents how to help your NanoHorizon submission (e.g. which track, which files to edit, record layout).

**Full walkthrough** (signup, credits / Codex entitlements, MCP login, spec + uploads): **[docs/smr/README.md](docs/smr/README.md)**.

SMR complements local `./scripts/...` runs. To **submit your solution**, follow [records/README.md](records/README.md): validate `records/<track>/...`, then **open a pull request** that includes your record and intended code changes — see [docs/smr/README.md](docs/smr/README.md#submitting-your-solution-pull-request).

## Reference baseline

Canonical Craftax SFT baseline:

```bash
./scripts/run_offline_training.sh
```

Single Python file to modify:

```bash
src/nanohorizon/baselines/offline_sft.py
```

Default reference settings:

- student: `Qwen/Qwen3.5-4B`
- teacher: `Qwen/Qwen3.5-9B`
- tool-calling-only Craftax traces
- `thinking_budget_tokens = 2000`
- async local rollout collection, reward filtering, Modal LoRA SFT, then held-out base-vs-finetuned compare
- held-out compare defaults: `20` eval rollouts at concurrency `10`

Most recent proved comparison:

- base mean reward: `0.3`
- finetuned mean reward: `0.5`
- reward delta: `+0.2`

## RLVR reference baseline

Canonical Craftax RLVR baseline:

```bash
./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh
```

Single Python file to modify:

```bash
src/nanohorizon/baselines/rlvr.py
```

Default reference settings:

- model: `Qwen/Qwen3.5-4B`
- topology: Craftax runs locally on controller (rank 0) via the in-repo Python runtime, vLLM on rank 1, connected via cluster-internal networking
- tool-calling-only Craftax interaction
- `thinking_budget_tokens = 2000`
- `max_tokens = 3072`
- grouped rollout size: `4`
- rollout concurrency: `32`, semaphore limit: `16`
- vLLM V1 engine with torch.compile and CUDA graphs (LoRA bounds-check patch applied automatically)
- LoRA targets: all attention + MLP + GDN layers (12 modules)
- periodic eval at bootstrap and after each learner iteration

Checked-in baseline results (3 iterations, 48-step rollouts, 2x A100-40GB):

| Metric | Value |
| --- | --- |
| **Score** (`final_mean_outcome_reward`) | `2.5` |
| **Bootstrap score** | `2.25` |
| **Reward lift** | `+0.25` |
| **Total rollouts** | `69` |
| **Iterations completed** | `3` |
| **Wall time** | `18.7 min` |
| **Rollouts/min** | `3.7` |
| **Peak generation throughput** | `1,451 tok/s` |
| **Steady-state generation throughput** | `~470 tok/s` |
| **vLLM concurrent requests** | `4–16` |

Key throughput optimizations over the initial topology-validation baseline:

1. vLLM V1 engine + torch.compile + CUDA graphs (was `--enforce-eager`)
2. In-place vLLM LoRA patch for Qwen3.5 hybrid Mamba-attention CUDA graph profiling (`src/nanohorizon/custom_vllm/lora_patch.py`)
3. Local Craftax on controller + cluster-internal inference networking (was Modal web proxy + tunnel, which serialized all LLM requests to concurrency 1)
4. Full-layer LoRA: added GDN projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`) per [Tinker LoRA recommendations](https://thinkingmachines.ai/blog/lora/)

How to interpret RLVR results:

- primary score: `metrics.json -> final_mean_outcome_reward`
- bootstrap baseline: `metrics.json -> step0_mean_outcome_reward`
- reward lift: `metrics.json -> reward_delta_from_bootstrap`
- periodic eval checkpoints: `periodic_eval/step_000`, `step_001`, ... — hillclimbing means later steps trend above bootstrap
- per-iteration training data: `iteration_summaries.json` and `iterations/iter_XXX/rollouts.jsonl`
- a run is only a clean reference record when it writes `metrics.json` and `final_eval_summary.json`

## Offline / SFT Records

| Run | Score | Student | Teacher | Summary | Date | Info |
| --- | ---: | --- | --- | --- | --- | --- |
| `reference_baseline` | `0.5` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` | Craftax FBC with tool-calling traces, 2k thinking budget, and held-out compare (`+0.2` delta over base) | `2026-03-20` | [info](records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) |
| `modal_4b_nochange_baseline` | `0.7` | `Qwen/Qwen3.5-4B` | `-` | Pure no-change 4B baseline via Modal inference on the 20 held-out seeds; raw rewards and 22-achievement frequencies are checked in | `2026-03-22` | [info](records/offline_20min_1xa100_40gb/2026-03-22_modal_4b_nochange_baseline/) |

---

## Tracks

| Track | Wall / budget | What you run |
| --- | --- | --- |
| `offline_20min_1xa100_40gb` | 20 min · 1× A100 40GB | `./scripts/run_offline_training.sh` |
| `rlvr_20min_2xa100_40gb` | 20 min · 2× A100 40GB | `./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh` |
| `prompt_opt_1usd_gpt54_family` | **$1** optimizer spend (GPT-5.4 family) | `./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh` |
| `classic` | Craftax-Classic `1M` · random-init RL under `100M` params | `./scripts/run_classic_craftax_1m_modal.sh` |

Craftax reference offline baseline uses **`Qwen/Qwen3.5-4B`** with a **`Qwen/Qwen3.5-9B`** teacher. The Craftax tracks are the repo's long-horizon RL lane; `classic` is the separate classic-RL lane. Track rules: [docs/tracks/](docs/tracks/) · tasks: [docs/task-craftax.md](docs/task-craftax.md), [docs/task-craftax-classic.md](docs/task-craftax-classic.md).

## Prompt-opt reference baseline

- checked-in record: [2026-03-21_reference_baseline](records/prompt_opt_1usd_gpt54_family/2026-03-21_reference_baseline)
- current checked-in score: `0.35`
- baseline seed-prompt eval: `0.6`
- optimizer family: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` (default: `gpt-5.4-mini`)
- search backend: `GEPA`
- score reported in the leaderboard: actual Craftax `mean_outcome_reward`, not GEPA search score

How to interpret prompt-opt results:

- primary score: `metrics.json -> primary_score`
- seed prompt baseline: `metrics.json -> bootstrap_score`
- prompt search objective: `metrics.json -> best_gepa_val_score`
- real hillclimbing means `primary_score > bootstrap_score`
- GEPA search score movement alone does not count as reward improvement

---

## Compete

1. **Install the Modal toolchain**:

   ```bash
   uv sync --group modal
   uv run modal setup
   ```

   The in-repo Craftax runtime lives under `src/nanohorizon/craftax_core/`.

2. **Clone and run** (from repo root):

   ```bash
   git clone https://github.com/synth-laboratories/nanohorizon.git && cd nanohorizon
   ./scripts/run_offline_training.sh   # or another track command above
   ```

3. **Default Modal GPUs**:
   - Offline default: `A10G`
   - RLVR default: `A100-40GB`
   - Prompt-opt default: `L4`
   - Eval default: `L4`

   Override per track with:

   ```bash
   NANOHORIZON_MODAL_GPU_OFFLINE=L4
   NANOHORIZON_MODAL_GPU_RLVR=A100-40GB
   NANOHORIZON_MODAL_GPU_PROMPT_OPT=T4
   NANOHORIZON_MODAL_GPU_EVAL=L4
   ```

4. **Submit**: add `records/<track>/<date>_<slug>/` with at least `metadata.json`, `metrics.json`, `system_info.json`, `command.txt`, `run_config.yaml` — see [records/README.md](records/README.md). Open a PR that includes the record **and** an updated leaderboard row.

5. **Check the bundle**:

   ```bash
   uv sync && uv run python -m nanohorizon.shared.validate_record records/<track>/<your_record_dir>
   ```

   No `uv`: `PYTHONPATH=src python3 -m nanohorizon.shared.validate_record records/<track>/<your_record_dir>`.

## Modal layout

- Shared Modal substrate: `src/nanohorizon/shared/modal_common.py`
- Shared Craftax eval entrypoint: `src/nanohorizon/shared/modal_eval.py`
- Shared Craftax runtime + HTTP shim: `src/nanohorizon/craftax_core/`
- Offline/FBC core logic: `src/nanohorizon/baselines/offline_sft.py`
- Offline/FBC Modal SFT entrypoint: `src/nanohorizon/baselines/offline_sft.py`
- Shared teacher / student vLLM entrypoint: `src/nanohorizon/shared/modal_teacher.py`
- RLVR track entrypoint: `src/nanohorizon/baselines/rlvr.py`
- RLVR single-script training logic: `src/nanohorizon/baselines/rlvr.py`
- Prompt-opt track entrypoint: `src/nanohorizon/baselines/prompt_opt.py`
- Classic track starter scaffold: `src/nanohorizon/baselines/classic.py`

Classic deliberately does **not** use the shared Modal/container stack above.
