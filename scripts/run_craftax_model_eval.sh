#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --no-project --with pyyaml --with pytest --python 3.11 \
  pytest -q tests/test_server_push_e2e_candidate.py
