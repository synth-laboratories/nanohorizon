#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
exec uv run --no-project --python 3.11 python src/nanohorizon/craftax_core/runner.py "$@"
