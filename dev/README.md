# Maintainer tooling

## RunPod GPU defaults

Launch scripts under `scripts/` source [`scripts/lib_runpod_gpu.sh`](../scripts/lib_runpod_gpu.sh): unless `NANOHORIZON_RUNPOD_GPU_TYPE` is set, they pass `--gpu-profile l4` (NVIDIA L4 — see `GPU_PROFILES` in `src/nanohorizon/shared/runpod_training_launcher.py`) for Qwen3.5-4B inference. Use `NANOHORIZON_RUNPOD_GPU_PROFILE=mid24` for a wider GPU pool; set `NANOHORIZON_RUNPOD_GPU_TYPE` for an exact RunPod GPU name (e.g. track-matched A100).

Prompt track: [`scripts/run_craftax_prompt_opt_reference.sh`](../scripts/run_craftax_prompt_opt_reference.sh) (image `nanohorizon-prompt-opt`; default **`--no-interruptible`** for stable public IP / log proxy — set `NANOHORIZON_RUNPOD_INTERRUPTIBLE=1` for spot). After `launch`, stderr lists **`training_log` / `job_status` proxy URLs**; JSON `payload.env` is redacted. Offline/SFT: [`scripts/run_craftax_offline_reference.sh`](../scripts/run_craftax_offline_reference.sh).

## Git hooks (pre-commit)

Hooks run **Ruff** (format + lint) and **ty** when you commit.

### One-time setup

From the repository root:

```bash
uv sync
uv run pre-commit install
```

Optional (run hooks on `git push` as well):

```bash
uv run pre-commit install --hook-type pre-push
```

### Manual runs

```bash
uv run pre-commit run --all-files          # everything
uv run pre-commit run ruff-format --all-files
uv run pre-commit run ty --all-files
```

### CI

Python **ruff** + **ty** also run in [`.github/workflows/python-checks.yml`](../.github/workflows/python-checks.yml). Pre-commit is the local mirror of those checks plus Rust and generic file hygiene.
