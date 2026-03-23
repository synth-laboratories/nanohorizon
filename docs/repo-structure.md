# Repository layout

Canonical map of **top-level directories** after consolidating docs under `docs/` and moving the Crafter runtime into the public repo. Modal is now the primary cloud execution path; legacy RunPod helpers remain in `scripts/` and `src/`.

## Top level

| Directory | Purpose |
| --- | --- |
| `configs/` | Pinned YAML for blessed baselines |
| `data/` | Small bootstrap datasets |
| `docker/` | Track Dockerfiles (with `scripts/build_track_image.sh`) |
| `docs/` | Task spec, track rules, and meta-docs (e.g. this file, [tracks.md](tracks.md)) |
| `dev/` | Maintainer hooks: [README.md](../dev/README.md), [install-hooks.sh](../dev/install-hooks.sh) |
| `records/` | Leaderboard bundles; path shape `records/<track_id>/<date>_<name>/` |
| `containers/` | Crafter container implementation: [crafter_rs](../containers/crafter_rs) |
| `scripts/` | Bash entrypoints for the three track runners, shared eval, and legacy RunPod helpers |
| `src/` | `nanohorizon` Python package (baselines, eval, Modal entrypoints, record validation, legacy RunPod launcher module) |

**Not committed:** `artifacts/` (default local training outputs), `.out/` (optional scratch), `containers/crafter_rs/target/` (Rust build).

## Documentation map

- **Tasks:** [task-crafter.md](task-crafter.md), [task-craftax-classic.md](task-craftax-classic.md)
- **Tracks index:** [tracks.md](tracks.md)
- **Per-track contracts:** [tracks/](tracks/) (`<track_id>.md` matches `records/<track_id>/`)
- **Synth Managed Research (SMR):** [smr/README.md](smr/README.md) — signup, MCP, spec + uploads for agent-assisted progress

## Future tweaks (optional)

- Point default `OUTPUT_ROOT` env vars at `.out/` if you want a single ignored tree at repo root (today defaults still use `artifacts/`).

## Python tooling

- **uv:** `uv sync --group dev --group modal` then `uv run …` (see root README).
- **Ruff / ty:** configured under `[tool.ruff]` and `[tool.ty]` in `pyproject.toml`.
- **Pre-commit:** [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) — install with `./dev/install-hooks.sh` or `uv run pre-commit install` (see [dev/README.md](../dev/README.md)).
