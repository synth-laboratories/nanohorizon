#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec ./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh "$@"
