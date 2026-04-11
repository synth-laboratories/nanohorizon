#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --python 3.11 python -m nanohorizon.craftax_core.runner "$@"
