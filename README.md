# NanoHorizon

NanoHorizon is a focused ML engineering golf repo for Crafter post-training under hard runtime and hardware budgets.

The initial goal is simple:

- make `Qwen/Qwen3.5-0.8B` better at Crafter
- keep the task reproducible and comparable
- publish records in a format that is easy to verify

NanoHorizon is related to the private `nanolong` lab, but it is intentionally narrower. `nanolong` remains the broader research workspace. `nanohorizon` is the public-facing competition surface.

## Get Started

NanoHorizon is meant to have one obvious entrypoint per competition track.

1. Clone `nanohorizon`.
2. Pick a track.
3. Run the single blessed script for that track.
4. Save the resulting artifacts as a record under `records/<track>/...`.

Track entrypoints:

| Track | What it is | One-command entrypoint |
|---|---|---|
| `rlvr_20min_2xa100_40gb` | RLVR-style Crafter training in 20 minutes on 2x A100 40GB | `./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh` |
| `offline_20min_1xa100_40gb` | Purely offline Crafter training in 20 minutes on 1x A100 40GB | `./scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh` |
| `prompt_opt_1usd_gpt54_family` | Prompt optimization for Crafter with a $1 optimizer budget across `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano` | `./scripts/run_crafter_prompt_opt_qwen35_08b_gpt54_budget.sh` |

Example:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh
```

## Offline Reference Run

The main user-editable surface for the offline track is:

- [run_crafter_offline_reference.sh](/Users/joshpurtell/Documents/GitHub/nanohorizon/scripts/run_crafter_offline_reference.sh)

That script is intended to be the easiest end-to-end path:

1. edit the knobs at the top if you want to change repo/ref/config/GPU
2. run it with `RUNPOD_API_KEY`
3. let RunPod execute the full pipeline

Command:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
RUNPOD_API_KEY=... ./scripts/run_crafter_offline_reference.sh
```

Runtime images:

- `ghcr.io/synth-laboratories/nanohorizon-offline:latest`
- `ghcr.io/synth-laboratories/nanohorizon-rlvr:latest`
- `ghcr.io/synth-laboratories/nanohorizon-prompt-opt:latest`
- `ghcr.io/synth-laboratories/nanohorizon-eval:latest`
- built remotely by GitHub Actions in `.github/workflows/build-track-images.yml`

Build and optionally push them with:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
./scripts/build_track_image.sh base
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh base
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh offline
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh rlvr
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh prompt_opt
NANOHORIZON_DOCKER_PUSH=1 ./scripts/build_track_image.sh eval
```

Every RunPod launcher accepts `NANOHORIZON_RUNPOD_IMAGE=...` to override the default image tag.

If local Docker is unreliable or too slow, run the GitHub Actions image workflow and wait for the GHCR tags to publish before launching RunPod.

What it does:

- launches a RunPod A100 pod
- starts `vllm` for `Qwen/Qwen3.5-27B`
- generates teacher SFT rows
- keeps the top half of generated rows by heuristic reward
- fine-tunes `Qwen/Qwen3.5-0.8B` with TRL
- evaluates the finetuned adapter

Baseline note:

- each track now has a self-contained baseline entrypoint in this repo
- the offline baseline is a RunPod-oriented `vllm` + `trl` pipeline
- the reference offline path may generate fresh SFT rows with `Qwen/Qwen3.5-27B` inside the same 20-minute budget window before training `Qwen/Qwen3.5-0.8B`
- the RLVR baseline is reward-weighted LoRA training from Crafter rollout JSONL
- the prompt optimization baseline is GEPA-style prompt search with `gpt-5.4-mini` as the default proposer
- the repo includes starter benchmark assets:
  - FT starter data: `data/crafter/crafter_ft_starter.jsonl`
  - FT seed prompts for `Qwen/Qwen3.5-27B`: `data/crafter/crafter_ft_seed_prompts.jsonl`
  - RLVR starter seeds: `data/crafter/crafter_rlvr_starter_seeds.json`
  - tiny rollout bootstrap data for local smoke tests

## Leaderboard

Current leaderboard status: bootstrap baselines.

| Track | Rank | Entry | Score | Status | Record |
|---|---|---|---|---|---|
| `rlvr_20min_2xa100_40gb` | 1 | `bootstrap_baseline` | `TBD` | reward-weighted LoRA | [record](/Users/joshpurtell/Documents/GitHub/nanohorizon/records/rlvr_20min_2xa100_40gb/2026-03-19_bootstrap_baseline) |
| `offline_20min_1xa100_40gb` | 1 | `bootstrap_baseline` | `TBD` | simple SFT | [record](/Users/joshpurtell/Documents/GitHub/nanohorizon/records/offline_20min_1xa100_40gb/2026-03-19_bootstrap_baseline) |
| `prompt_opt_1usd_gpt54_family` | 1 | `bootstrap_baseline` | `TBD` | GEPA-style prompt search | [record](/Users/joshpurtell/Documents/GitHub/nanohorizon/records/prompt_opt_1usd_gpt54_family/2026-03-19_bootstrap_baseline) |

The intended long-term model is the same as Parameter Golf:

- every serious run gets a durable record directory
- the README leaderboard points at those records
- rankings are derived from pinned metrics, not ad hoc screenshots

## Initial Tracks

- `rlvr_20min_2xa100_40gb`
  - train `Qwen/Qwen3.5-0.8B` for Crafter with RLVR-style methods
  - hard budget: 20 minutes wall clock
  - hardware budget: 2x A100 40GB
- `offline_20min_1xa100_40gb`
  - train `Qwen/Qwen3.5-0.8B` for Crafter with purely offline methods
  - hard budget: 20 minutes wall clock
  - hardware budget: 1x A100 40GB
- `prompt_opt_1usd_gpt54_family`
  - improve the Crafter prompting and policy scaffolding for `Qwen/Qwen3.5-0.8B`
  - optimization budget: $1 total compute spend
  - allowed optimizers: any mix of `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`

## Repo Layout

- `tasks/crafter/`
  - task definition, evaluation notes, and allowed data assumptions
- `reference/crafter/crafter_rs_container/`
  - in-repo Rust Crafter runtime exposing the rollout container contract
- `tracks/`
  - one folder per official competition track
- `scripts/`
  - blessed launcher scripts for the official baselines
- `configs/`
  - pinned configs used by the blessed scripts
- `tools/`
  - record validation and metadata helpers
- `records/`
  - reproducible public submissions and baselines

## Current Status

This repository is a first public benchmark scaffold with real baseline entrypoints.

That means:

- the task, tracks, and record layout are defined here
- the starter scripts are real and executable
- the baselines live fully inside `nanohorizon`
- the Crafter runtime now lives in-repo under `reference/crafter/crafter_rs_container`
- the checked-in data is small bootstrap data, not the final benchmark bundle

## Starter Commands

RLVR track bootstrap:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
./scripts/run_crafter_rlvr_qwen35_08b_2xa100_20min.sh
```

Offline track bootstrap:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
./scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh
```

Offline track RunPod launch:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
RUNPOD_API_KEY=... ./scripts/run_crafter_offline_reference.sh
```

Prompt optimization track bootstrap:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
./scripts/run_crafter_prompt_opt_qwen35_08b_gpt54_budget.sh
```

Validate a record bundle:

```bash
cd /Users/joshpurtell/Documents/GitHub/nanohorizon
python3 tools/validate_record.py records/rlvr_20min_2xa100_40gb/2026-03-19_bootstrap_baseline
```

## Relationship To Parameter Golf

NanoHorizon is inspired by the reproducibility norms in OpenAI's Parameter Golf records:

- every submission should live in a stable directory
- every submission should include exact config and command metadata
- every submission should carry enough metrics and system information to compare fairly

Reference:
- [OpenAI Parameter Golf records](https://github.com/openai/parameter-golf/tree/main/records)

## Next Steps

- extract the minimum Crafter runtime and eval harness needed for open sourcing
- publish one baseline record for each track
- tighten the track rules around timing, allowed inputs, and verification
