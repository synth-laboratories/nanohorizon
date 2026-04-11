#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NANOHORIZON_PROMPT_OPT_CONFIG="${NANOHORIZON_PROMPT_OPT_CONFIG:-configs/craftax_prompt_opt_qwen35_4b_codex_pipeline_fix_e2e.yaml}"
exec "$ROOT/scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" "$@"
