#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${NANOHORIZON_NLE_BIND_HOST:-127.0.0.1}"
PORT="${NANOHORIZON_NLE_BIND_PORT:-8913}"

export NANOHORIZON_NLE_BIND_HOST="$HOST"
export NANOHORIZON_NLE_BIND_PORT="$PORT"

exec /opt/homebrew/bin/uv run --group nle python -m nanohorizon.nle_core.http_shim
