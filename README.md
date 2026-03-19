# NanoHorizon

**Crafter post-training under hard time, hardware, and budget caps.** Improve `Qwen/Qwen3.5-0.8B` on [Crafter](https://danijar.com/project/crafter/) with reproducible runs, pinned metrics, and public records anyone can verify.

NanoHorizon is the public competition surface for this problem. It is intentionally narrow: broader research lives elsewhere; here the task, tracks, and submission format are fixed so results stay comparable.

---

## Leaderboard

Status: **bootstrap baselines** — scores are placeholders until first verified runs land.

| Track | Rank | Run | Score | Summary | Record |
| --- | ---: | --- | --- | --- | --- |
| `rlvr_20min_2xa100_40gb` | 1 | `bootstrap_baseline` | TBD | Reward-weighted LoRA from Crafter rollout JSONL | [info](records/rlvr_20min_2xa100_40gb/2026-03-19_bootstrap_baseline/) |
| `offline_20min_1xa100_40gb` | 1 | `bootstrap_baseline` | TBD | Teacher SFT + TRL fine-tune on 1×A100 | [info](records/offline_20min_1xa100_40gb/2026-03-19_bootstrap_baseline/) |
| `prompt_opt_1usd_gpt54_family` | 1 | `bootstrap_baseline` | TBD | GEPA-style prompt search (default proposer: `gpt-5.4-mini`) | [info](records/prompt_opt_1usd_gpt54_family/2026-03-19_bootstrap_baseline/) |

Rankings follow **pinned metrics** in each record’s `metrics.json`, not screenshots or ad hoc claims. New SOTA rows should add a dated directory under `records/<track>/` and update this table in the same PR.

---

## Tracks

Each track has **one blessed entrypoint script** and a fixed resource envelope.

| Track | Objective | Wall clock | Hardware / budget |
| --- | --- | --- | --- |
| `rlvr_20min_2xa100_40gb` | RLVR-style training for Crafter | 20 min | 2× A100 40GB |
| `offline_20min_1xa100_40gb` | Purely offline training (e.g. SFT / distillation) | 20 min | 1× A100 40GB |
| `prompt_opt_1usd_gpt54_family` | Prompt / policy scaffolding for the same base model | — | **$1** total optimizer spend across `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` |

**Base model target:** `Qwen/Qwen3.5-0.8B` unless a [track doc](docs/tracks/) states otherwise.

---

## Getting started

1. **Clone** this repository.
2. **Pick a track** from the table above.
3. **Run** the matching script from the repo root.
4. **Publish** outputs as a record under `records/<track>/<YYYY-MM-DD>_<name>/` (see [records/README.md](records/README.md)).

### One-command entrypoints

| Track | Command |
| --- | --- |
| RLVR | `./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh` |
| Offline | `./scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh` |
| Prompt optimization | `./scripts/run_crafter_prompt_opt_qwen35_08b_gpt54_budget.sh` |

Example:

```bash
git clone https://github.com/synth-laboratories/nanohorizon.git
cd nanohorizon
./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh
```

### Validate a record

```bash
PYTHONPATH=src python3 -m nanohorizon.validate_record records/rlvr_20min_2xa100_40gb/2026-03-19_bootstrap_baseline
```

---

## Submissions

Submissions are **directories**, not issues.

1. Add `records/<track>/<date>_<slug>/` with at least: `metadata.json`, `metrics.json`, `system_info.json`, `command.txt`, `run_config.yaml` (full checklist: [records/README.md](records/README.md)).
2. Open a PR that includes the record and an updated **Leaderboard** row pointing at it.
3. Prefer enough logging and config detail that an independent rerun can reproduce the claim within the track’s rules.

Records use **stable paths**, **explicit commands**, and **system metadata** so comparisons stay fair and auditable.

---

## Offline / RunPod reference path

For the offline track, the most hands-off cloud path is the reference launcher (RunPod + API key):

```bash
RUNPOD_API_KEY=... ./scripts/run_crafter_offline_reference.sh
```

Editable knobs live at the top of [scripts/run_crafter_offline_reference.sh](scripts/run_crafter_offline_reference.sh). That pipeline can generate teacher SFT rows with `Qwen/Qwen3.5-27B` (vLLM) and fine-tune `Qwen/Qwen3.5-0.8B` with TRL inside the track’s time budget.

**Container images** (built via [`.github/workflows/build-track-images.yml`](.github/workflows/build-track-images.yml)):

| Image | Role |
| --- | --- |
| `ghcr.io/synth-laboratories/nanohorizon-offline:latest` | Offline track |
| `ghcr.io/synth-laboratories/nanohorizon-rlvr:latest` | RLVR track |
| `ghcr.io/synth-laboratories/nanohorizon-prompt-opt:latest` | Prompt optimization |
| `ghcr.io/synth-laboratories/nanohorizon-eval:latest` | Evaluation |

Override the default tag with `NANOHORIZON_RUNPOD_IMAGE=...` on any RunPod launcher.

**Build locally** (optional push with `NANOHORIZON_DOCKER_PUSH=1`):

```bash
./scripts/build_track_image.sh base
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh offline
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh rlvr
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh prompt_opt
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh eval
```

---

## Repository layout

Layout reference: [docs/repo-structure.md](docs/repo-structure.md). Track rules: [docs/tracks.md](docs/tracks.md).

| Path | Purpose |
| --- | --- |
| [docs/task-crafter.md](docs/task-crafter.md) | Crafter task definition, eval shape, starter assets |
| [docs/tracks/](docs/tracks/) | Per-track contracts (`<track_id>.md`) |
| [runtime/crafter_rs/](runtime/crafter_rs/) | In-repo Rust Crafter runtime (rollout contract) |
| [scripts/](scripts/) | Blessed baseline launchers |
| [configs/](configs/) | Pinned configs for those scripts |
| [src/nanohorizon/](src/nanohorizon/) | Python package (baselines, eval, `validate_record`, `runpod_training_launcher`) |
| [records/](records/) | Public baselines and submissions |

**Starter data** (small bootstrap bundle, not the final benchmark corpus):

- `data/crafter/crafter_ft_starter.jsonl`
- `data/crafter/crafter_ft_seed_prompts.jsonl`
- `data/crafter/crafter_rlvr_starter_seeds.json`
- Rollout samples for local smoke tests

---

## FAQ

**What counts as the score?**  
Whatever the track pins in `metrics.json` and documents in the record; the leaderboard should name that field explicitly as scores stabilize.

**Can I change hyperparameters?**  
Yes, within the track’s time, hardware, and (for prompt optimization) dollar caps. Document everything in `run_config.yaml` and `command.txt`.

**Is this an official OpenAI project?**  
No.

---

## Current status

- Task, tracks, and record layout are defined; baseline scripts are real and runnable from this repo.
- Crafter runtime is vendored under `runtime/crafter_rs/`.
- Checked-in data is **bootstrap-scale**; benchmark bundles and stricter verification rules may tighten over time.
