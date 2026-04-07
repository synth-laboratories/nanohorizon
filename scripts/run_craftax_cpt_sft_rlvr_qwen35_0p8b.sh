#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Stage 1: CPT data generation (skip if data exists)
# ---------------------------------------------------------------------------
if [[ "${NANOHORIZON_SKIP_CPT_DATA:-1}" != "1" ]]; then
  echo "=== Stage 1: CPT data generation ===" >&2
  bash scripts/run_craftax_cpt_generate_qwen35_27b_100k.sh
fi

# ---------------------------------------------------------------------------
# Stage 2: CPT training on Modal → push to HF Hub
# ---------------------------------------------------------------------------
CPT_MODEL="${NANOHORIZON_CPT_MODEL:-JoshPurtell/qwen35-0p8b-craftax-cpt}"
if [[ "${NANOHORIZON_SKIP_CPT:-1}" != "1" ]]; then
  echo "=== Stage 2: CPT training ===" >&2
  export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required for CPT push}"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --group modal modal run src/nanohorizon/baselines/cpt.py \
      --config configs/craftax_cpt_qwen35_0p8b_100k.yaml
fi

# ---------------------------------------------------------------------------
# Stage 3: SFT (tool calling + thinking) on Modal
# ---------------------------------------------------------------------------
SFT_MODEL="${NANOHORIZON_SFT_MODEL:-}"
if [[ "${NANOHORIZON_SKIP_SFT:-0}" != "1" ]]; then
  echo "=== Stage 3: SFT (tool calling + thinking) ===" >&2
  SFT_CONFIG="${NANOHORIZON_SFT_CONFIG:-configs/craftax_sft_qwen35_0p8b_from_cpt_rollouts.yaml}"

  # Run SFT on Modal via offline_sft pipeline
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --group modal --group classic --group training modal run src/nanohorizon/baselines/offline_sft.py \
      --config "$SFT_CONFIG"

  # TODO: extract SFT adapter path from output and set SFT_MODEL
  echo "SFT complete. Set NANOHORIZON_SFT_MODEL to the adapter/merged model ref for RLVR." >&2
fi

# ---------------------------------------------------------------------------
# Stage 4: RLVR with high-throughput config
# ---------------------------------------------------------------------------
RLVR_MODEL="${NANOHORIZON_CPT_RLVR_MODEL_REF:-${SFT_MODEL:-${CPT_MODEL}}}"
echo "=== Stage 4: RLVR with model=$RLVR_MODEL ===" >&2

RLVR_CONFIG="${NANOHORIZON_RLVR_CONFIG:-configs/craftax_rlvr_qwen35_0p8b_high_throughput.yaml}"
TMP_RLVR="$(mktemp /tmp/nanohorizon_rlvr_XXXXXX.yaml)"
trap "rm -f $TMP_RLVR" EXIT

uv run python - "$ROOT/$RLVR_CONFIG" "$TMP_RLVR" "$RLVR_MODEL" <<'PY'
from pathlib import Path
import sys, yaml
source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
model_ref = sys.argv[3]
payload = yaml.safe_load(source.read_text())
payload.setdefault("model", {})
payload["model"]["model"] = model_ref
destination.write_text(yaml.safe_dump(payload, sort_keys=False))
PY

export NANOHORIZON_RLVR_CONFIG="$TMP_RLVR"
exec bash scripts/run_craftax_rlvr_qwen35_0p8b_2xa100_20min.sh
