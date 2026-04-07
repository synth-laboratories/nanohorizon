# Shared GPU selection for `runpod_training_launcher launch`.
#
# Default: `l4` (NVIDIA L4, ~24GB — see GPU_PROFILES in runpod_training_launcher.py). Good default for
# Qwen3.5-4B inference on RunPod. Override: NANOHORIZON_RUNPOD_GPU_PROFILE=mid24 for a broader pool,
# `small16` for lighter pods, or NANOHORIZON_RUNPOD_GPU_TYPE for an exact RunPod GPU name.
#
# Env:
#   NANOHORIZON_RUNPOD_GPU_TYPE     - if set, --gpu-type-id (wins over profile)
#   NANOHORIZON_RUNPOD_GPU_PROFILE  - if no GPU_TYPE, --gpu-profile (default: l4)

nanoh_runpod_gpu_desc() {
  if [[ -n "${NANOHORIZON_RUNPOD_GPU_TYPE:-}" ]]; then
    printf '%s' "${NANOHORIZON_RUNPOD_GPU_TYPE}"
  else
    printf 'profile %s' "${NANOHORIZON_RUNPOD_GPU_PROFILE:-l4}"
  fi
}

# Call before `runpod_training_launcher launch`; expands as "${NANOH_RUNPOD_GPU_ARGS[@]}".
nanoh_runpod_gpu_load() {
  NANOH_RUNPOD_GPU_ARGS=()
  if [[ -n "${NANOHORIZON_RUNPOD_GPU_TYPE:-}" ]]; then
    NANOH_RUNPOD_GPU_ARGS=(--gpu-type-id "${NANOHORIZON_RUNPOD_GPU_TYPE}")
  else
    NANOH_RUNPOD_GPU_ARGS=(--gpu-profile "${NANOHORIZON_RUNPOD_GPU_PROFILE:-l4}")
  fi
}
