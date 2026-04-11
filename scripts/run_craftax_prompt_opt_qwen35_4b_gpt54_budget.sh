#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOHORIZON_PROMPT_OPT_CONFIG:-configs/craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml}"

cd "$ROOT"

uv run python - "$CONFIG" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml


config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"missing prompt-opt config: {config_path}")

payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
seed_prompt = str(payload["prompt"]["seed_prompt"])

print("Craftax prompt-opt wrapper")
print(f"CONFIG={config_path}")
print(f"TRACK={payload['search']['objective']}")
print(f"MODEL={payload['policy']['model']}")
print(f"OUTPUT_ROOT={payload['output']['root_dir']}")
print(f"SEED_PROMPT_HAS_TODO={ 'todo list with exactly three items' in seed_prompt }")
PY
