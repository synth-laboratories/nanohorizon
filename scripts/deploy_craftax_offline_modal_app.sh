#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${NANOHORIZON_MODAL_OFFLINE_APP_NAME:-nanohorizon-craftax-offline}"
CODE_VERSION="${NANOHORIZON_CODE_VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

cd "$ROOT"
export NANOHORIZON_MODAL_OFFLINE_APP_NAME="$APP_NAME"
export NANOHORIZON_CODE_VERSION="$CODE_VERSION"
exec /opt/homebrew/bin/uv run --group modal modal deploy src/nanohorizon/shared/modal_offline.py "$@"
