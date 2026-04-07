#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
uv sync --group dev
uv run pre-commit install
echo "pre-commit installed. Try: uv run pre-commit run --all-files"
