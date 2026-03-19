#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_REPO="${NANOHORIZON_GIT_REPO:-}"
GIT_REF="${NANOHORIZON_GIT_REF:-main}"
IMAGE_NAME="${NANOHORIZON_RUNPOD_IMAGE:-ghcr.io/synth-laboratories/nanohorizon-offline:latest}"

if [[ -z "$GIT_REPO" ]]; then
  echo "Set NANOHORIZON_GIT_REPO to the Git URL for this repo before launching RunPod." >&2
  exit 1
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m nanohorizon.runpod_training_launcher launch \
  --image-name "$IMAGE_NAME" \
  --name "nanohorizon-offline-$(date -u +%Y%m%d-%H%M%S)" \
  --gpu-type-id "NVIDIA A100 80GB PCIe" \
  --gpu-count 1 \
  --container-disk-gb 80 \
  --volume-gb 160 \
  --support-public-ip \
  --git-repo "$GIT_REPO" \
  --git-ref "$GIT_REF" \
  --repo-dir nanohorizon \
  --setup-cmd "cd /workspace/nanohorizon && python3 -V && echo using prebuilt offline runtime image" \
  --train-cmd "cd /workspace/nanohorizon && NANOHORIZON_AUTO_INSTALL=0 NANOHORIZON_START_LOCAL_TEACHER=1 bash scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh" \
  --auto-stop \
  "$@"
