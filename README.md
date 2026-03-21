# NanoHorizon

<p align="center">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" alt="Top-down survival / crafting gameplay in the same long-horizon spirit as Crafter" />
</p>

**Fast, cheap iteration for post-training on long-horizon agents.** Train Crafter agents under fixed time, hardware, and budget caps; publish a verifiable record and climb the leaderboard.

*GIF: [JoshuaPurtell/craftaxlm](https://github.com/JoshuaPurtell/craftaxlm); asset [Craftax](https://github.com/MichaelTMatthews/Craftax) (`images/dungeon_crawling.gif`).*

---

Base model target: `Qwen/Qwen3.5-4B` unless a track doc states otherwise.

## Leaderboard

Status: **bootstrap baselines** — replace `TBD` with your `metrics.json` score and open a PR.

| Track | Rank | Run | Score | Summary | Record |
| --- | ---: | --- | --- | --- | --- |
| `offline_20min_1xa100_40gb` | 1 | `reference_baseline` | `0.5` | Crafter FBC on 4B with 9B teacher; held-out compare gives `+0.2` reward delta | [info](records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) |
| `rlvr_20min_2xa100_40gb` | 1 | `bootstrap_baseline` | TBD | RLVR-style LoRA from rollout JSONL | [info](records/rlvr_20min_2xa100_40gb/2026-03-19_bootstrap_baseline/) |
| `prompt_opt_1usd_gpt54_family` | 1 | `bootstrap_baseline` | TBD | Prompt search ($1 optimizer budget) | [info](records/prompt_opt_1usd_gpt54_family/2026-03-19_bootstrap_baseline/) |

New rows: add `records/<track>/<YYYY-MM-DD>_<name>/` and update this table in the **same PR**.

## Change And Run

For the offline reference baseline, there are only two files to care about:

1. Change the learning logic in [offline_training.py](/Users/joshpurtell/Documents/GitHub/nanohorizon/src/nanohorizon/offline_training.py)
2. Run the full pipeline with [run_offline_training.sh](/Users/joshpurtell/Documents/GitHub/nanohorizon/scripts/run_offline_training.sh)

Reference config and default sizes:

- config: [crafter_offline_reference.yaml](/Users/joshpurtell/Documents/GitHub/nanohorizon/configs/crafter_offline_reference.yaml)
- student: `Qwen/Qwen3.5-4B`
- teacher: `Qwen/Qwen3.5-9B`
- eval: `20` held-out rollouts at concurrency `10`

Replication command:

```bash
./scripts/run_offline_training.sh
```

What that bash script handles for you:

- starts the local Crafter service
- runs teacher inference on Modal
- runs SFT on Modal
- collects rollouts locally against the local Crafter container
- runs held-out base vs finetuned evals locally against the local Crafter container
- writes the comparison summary

Default offline path:

- no tunnel
- local Crafter container at `http://127.0.0.1:8903`
- remote Modal inference for teacher, base student, and finetuned student
- remote Modal SFT for the student LoRA

Reference record to compare against:

- [2026-03-20_reference_baseline](/Users/joshpurtell/Documents/GitHub/nanohorizon/records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline)
- score: `0.5`
- reward delta over base: `+0.2`

## Reference baseline

Canonical Crafter SFT baseline:

```bash
./scripts/run_offline_training.sh
```

Single Python file to modify:

```bash
src/nanohorizon/offline_training.py
```

Default reference settings:

- student: `Qwen/Qwen3.5-4B`
- teacher: `Qwen/Qwen3.5-9B`
- tool-calling-only Crafter traces
- `thinking_budget_tokens = 2000`
- async local rollout collection, reward filtering, Modal LoRA SFT, then held-out base-vs-finetuned compare
- held-out compare defaults: `20` eval rollouts at concurrency `10`

Most recent proved comparison:

- base mean reward: `0.3`
- finetuned mean reward: `0.5`
- reward delta: `+0.2`

## Offline / SFT Records

| Run | Score | Student | Teacher | Summary | Date | Info |
| --- | ---: | --- | --- | --- | --- | --- |
| `reference_baseline` | `0.5` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-9B` | Crafter FBC with tool-calling traces, 2k thinking budget, and held-out compare (`+0.2` delta over base) | `2026-03-20` | [info](records/offline_20min_1xa100_40gb/2026-03-20_reference_baseline/) |

---

## Tracks

| Track | Wall / budget | What you run |
| --- | --- | --- |
| `offline_20min_1xa100_40gb` | 20 min · 1× A100 40GB | `./scripts/run_offline_training.sh` |
| `rlvr_20min_2xa100_40gb` | 20 min · 2× A100 40GB | `./scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh` |
| `prompt_opt_1usd_gpt54_family` | **$1** optimizer spend (GPT-5.4 family) | `./scripts/run_crafter_prompt_opt_qwen35_4b_gpt54_budget.sh` |

Reference offline baseline uses **`Qwen/Qwen3.5-4B`** with a **`Qwen/Qwen3.5-9B`** teacher. Track rules: [docs/tracks/](docs/tracks/) · task: [docs/task-crafter.md](docs/task-crafter.md).

---

## Compete

1. **Install the Modal toolchain**:

   ```bash
   uv sync --group modal
   uv run modal setup
   ```

   Crafter prerequisite for the offline reference path:

   ```text
   ~/Documents/GitHub/nanohorizon
   ~/Documents/GitHub/crafter-rs
   ```

2. **Clone and run** (from repo root):

   ```bash
   git clone https://github.com/synth-laboratories/nanohorizon.git && cd nanohorizon
   ./scripts/run_offline_training.sh   # or another track command above
   ```

3. **Cheap-by-default Modal GPUs**:
   - Offline default: `A10G`
   - RLVR default: `A10G`
   - Prompt-opt default: `L4`
   - Eval default: `L4`

   Override per track with:

   ```bash
   NANOHORIZON_MODAL_GPU_OFFLINE=L4
   NANOHORIZON_MODAL_GPU_RLVR=A10G
   NANOHORIZON_MODAL_GPU_PROMPT_OPT=T4
   NANOHORIZON_MODAL_GPU_EVAL=L4
   ```

4. **Submit**: add `records/<track>/<date>_<slug>/` with at least `metadata.json`, `metrics.json`, `system_info.json`, `command.txt`, `run_config.yaml` — see [records/README.md](records/README.md). Open a PR that includes the record **and** an updated leaderboard row.

5. **Check the bundle**:

   ```bash
   uv sync && uv run python -m nanohorizon.validate_record records/<track>/<your_record_dir>
   ```

   No `uv`: `PYTHONPATH=src python3 -m nanohorizon.validate_record records/<track>/<your_record_dir>`.

## Modal layout

- Shared Modal substrate: `src/nanohorizon/modal_common.py`
- Shared Crafter eval entrypoint: `src/nanohorizon/modal_eval.py`
- Offline/FBC core logic: `src/nanohorizon/offline_training.py`
- Offline/FBC SFT entrypoint: `src/nanohorizon/modal_sft.py`
- Shared teacher / student vLLM entrypoint: `src/nanohorizon/modal_teacher.py`
- RLVR track entrypoint: `src/nanohorizon/modal_rlvr.py`
- Prompt-opt track entrypoint: `src/nanohorizon/modal_prompt_opt.py`
