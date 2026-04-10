#!/usr/bin/env bash
set -euo pipefail

uv run --python 3.11 python -m nanohorizon.craftax_core.runner --demo "$@"

