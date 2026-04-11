#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_PATH="${NANOHORIZON_CRAFTAX_CANDIDATE_MANIFEST:-experiments/nanohorizon_leaderboard_candidate/results/candidate_manifest.json}"
PROMPT_PATH="${NANOHORIZON_CRAFTAX_CANDIDATE_PROMPT:-experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt}"

cd "$ROOT"
exec uv run python -m nanohorizon.craftax_core.runner \
  --write "$MANIFEST_PATH" \
  --prompt-out "$PROMPT_PATH" \
  "$@"
