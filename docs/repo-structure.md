# Repository layout

Canonical map of **top-level directories** after consolidating docs under `docs/` and moving the Crafter runtime + RunPod launcher out of the old `reference/` tree.

## Top level

| Directory | Purpose |
| --- | --- |
| `configs/` | Pinned YAML for blessed baselines |
| `data/` | Small bootstrap datasets |
| `docker/` | Track Dockerfiles (with `scripts/build_track_image.sh`) |
| `docs/` | Task spec, track rules, and meta-docs (e.g. this file, [tracks.md](tracks.md)) |
| `records/` | Leaderboard bundles; path shape `records/<track_id>/<date>_<name>/` |
| `runtime/` | Vendored Crafter Rust service: [crafter_rs](../runtime/crafter_rs) |
| `scripts/` | Bash entrypoints and RunPod launch glue |
| `src/` | `nanohorizon` Python package (baselines, eval, record validation, RunPod launcher module) |

**Not committed:** `artifacts/` (default local training outputs), `.out/` (optional scratch), `runtime/crafter_rs/target/` (Rust build).

## Documentation map

- **Task (Crafter):** [task-crafter.md](task-crafter.md)
- **Tracks index:** [tracks.md](tracks.md)
- **Per-track contracts:** [tracks/](tracks/) (`<track_id>.md` matches `records/<track_id>/`)

## Future tweaks (optional)

- Point default `OUTPUT_ROOT` env vars at `.out/` if you want a single ignored tree at repo root (today defaults still use `artifacts/`).
- Add `[project.scripts]` entry points if you want `pip install` shims instead of `PYTHONPATH=src python3 -m …`.
