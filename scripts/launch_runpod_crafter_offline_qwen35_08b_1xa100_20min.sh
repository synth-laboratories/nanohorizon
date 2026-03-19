#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_REPO="${NANOHORIZON_GIT_REPO:-}"
GIT_REF="${NANOHORIZON_GIT_REF:-main}"

if [[ -z "$GIT_REPO" ]]; then
  echo "Set NANOHORIZON_GIT_REPO to the Git URL for this repo before launching RunPod." >&2
  exit 1
fi

python3 "$ROOT/reference/runpod_training_launcher.py" launch \
  --name "nanohorizon-offline-$(date -u +%Y%m%d-%H%M%S)" \
  --gpu-type-id "NVIDIA A100 40GB PCIe" \
  --gpu-count 1 \
  --container-disk-gb 80 \
  --volume-gb 160 \
  --support-public-ip \
  --install-uv \
  --git-repo "$GIT_REPO" \
  --git-ref "$GIT_REF" \
  --repo-dir nanohorizon \
  --setup-cmd "cd /workspace/nanohorizon && python3 -m pip install --upgrade pip && python3 -m pip install -q 'httpx>=0.28.1' 'pyyaml>=6.0.2' 'accelerate>=1.10.0' 'datasets>=4.1.0' 'peft>=0.17.0' 'transformers>=4.57.0' 'trl>=0.21.0' 'vllm>=0.10.0'" \
  --train-cmd "cd /workspace/nanohorizon && NANOHORIZON_AUTO_INSTALL=0 NANOHORIZON_START_LOCAL_TEACHER=1 bash scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh" \
  --auto-stop \
  "$@"
