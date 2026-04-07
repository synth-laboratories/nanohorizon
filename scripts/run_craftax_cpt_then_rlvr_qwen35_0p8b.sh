#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_SCRIPT="${NANOHORIZON_CPT_DATA_SCRIPT:-$ROOT/scripts/run_craftax_cpt_generate_qwen35_27b_100k.sh}"
CPT_SCRIPT="${NANOHORIZON_CPT_SCRIPT:-$ROOT/scripts/run_craftax_cpt_qwen35_0p8b_100k.sh}"
RLVR_SCRIPT="${NANOHORIZON_CPT_RLVR_SCRIPT:-$ROOT/scripts/run_craftax_rlvr_qwen35_0p8b_2xa100_20min.sh}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Stage 1: Generate CPT data (teacher rollouts → text)
# ---------------------------------------------------------------------------
if [[ "${NANOHORIZON_SKIP_CPT_DATA:-0}" != "1" ]]; then
  echo "=== Stage 1: CPT data generation ===" >&2
  "$DATA_SCRIPT"
fi

# ---------------------------------------------------------------------------
# Stage 2: CPT training on Modal (HF Trainer)
# ---------------------------------------------------------------------------
if [[ "${NANOHORIZON_SKIP_CPT:-0}" != "1" ]]; then
  echo "=== Stage 2: CPT training ===" >&2

  CPT_CONFIG="${NANOHORIZON_CPT_CONFIG:-configs/craftax_cpt_qwen35_0p8b_100k.yaml}"
  CPT_SOURCE_JSONL="${NANOHORIZON_CPT_SOURCE_JSONL:-data/craftax/cpt_rollouts_text.jsonl}"

  if [[ ! -f "$CPT_SOURCE_JSONL" ]]; then
    echo "ERROR: CPT source JSONL not found at $CPT_SOURCE_JSONL" >&2
    echo "Run data generation first or set NANOHORIZON_CPT_SOURCE_JSONL" >&2
    exit 1
  fi

  # Run CPT on Modal and capture the output path
  CPT_RESULT="$(uv run --group modal modal run src/nanohorizon/baselines/cpt.py \
    --config "$CPT_CONFIG" 2>&1 | tee /dev/stderr)"

  # Extract hf_export_path from the JSON output
  CPT_HF_EXPORT_PATH="$(echo "$CPT_RESULT" | uv run python -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        if isinstance(data, dict) and 'hf_export_path' in data:
            print(data['hf_export_path'])
            break
    except json.JSONDecodeError:
        continue
" 2>/dev/null || true)"

  if [[ -n "$CPT_HF_EXPORT_PATH" ]]; then
    echo "CPT export path: $CPT_HF_EXPORT_PATH" >&2
    export NANOHORIZON_CPT_RLVR_MODEL_REF="$CPT_HF_EXPORT_PATH"
  fi
fi

# ---------------------------------------------------------------------------
# Stage 3: RLVR training on Modal
# ---------------------------------------------------------------------------
MODEL_REF="${NANOHORIZON_CPT_RLVR_MODEL_REF:-}"
if [[ -z "$MODEL_REF" ]]; then
  cat >&2 <<'EOF'
missing NANOHORIZON_CPT_RLVR_MODEL_REF

Set this to the exported CPT checkpoint path (on the Modal volume or as a
Hugging Face model id) before running the RLVR stage.

Example:
  export NANOHORIZON_CPT_RLVR_MODEL_REF="/vol/records/cpt_rlvr_qwen35_0p8b/.../hf_export"
  # or
  export NANOHORIZON_CPT_RLVR_MODEL_REF="your-org/qwen35-0p8b-craftax-cpt-bootstrap"
EOF
  exit 1
fi

echo "=== Stage 3: RLVR training with model=$MODEL_REF ===" >&2

export NANOHORIZON_RLVR_CONFIG="${NANOHORIZON_RLVR_CONFIG:-configs/craftax_rlvr_qwen35_0p8b_2xa100_20min.yaml}"
TMP_RLVR_CONFIG="$(mktemp "${TMPDIR:-/tmp}/nanohorizon_cpt_rlvr_XXXXXX.yaml")"
cleanup() {
  rm -f "$TMP_RLVR_CONFIG"
}
trap cleanup EXIT

uv run python - <<'PY' "$ROOT/$NANOHORIZON_RLVR_CONFIG" "$TMP_RLVR_CONFIG" "$MODEL_REF"
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
model_ref = sys.argv[3]

payload = yaml.safe_load(source.read_text(encoding="utf-8"))
payload.setdefault("model", {})
payload["model"]["model"] = model_ref
destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

export NANOHORIZON_RLVR_CONFIG="$TMP_RLVR_CONFIG"
exec "$RLVR_SCRIPT"
