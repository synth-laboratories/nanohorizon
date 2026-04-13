#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${NANOHORIZON_PIVOTRL_ARTIFACT_DIR:-$ROOT/artifacts/pivotrl_reference_$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_OUTPUT_DIR="${NANOHORIZON_MODAL_PIVOTRL_OUTPUT_DIR:-/vol/artifacts/pivotrl/$(basename "$ARTIFACT_DIR")}"
TRAINING_LOG="$ARTIFACT_DIR/pivotrl_training.log"
MODAL_TRAIN_RESULT_PATH="$ARTIFACT_DIR/modal_train_result.json"
BOOTSTRAP_ENDPOINT_STDOUT="$ARTIFACT_DIR/bootstrap_teacher.stdout.log"
BOOTSTRAP_ENDPOINT_STDERR="$ARTIFACT_DIR/bootstrap_teacher.stderr.log"
BOOTSTRAP_ENDPOINT_INFO="$ARTIFACT_DIR/bootstrap_teacher.endpoint.txt"
BASE_ENDPOINT_STDOUT="$ARTIFACT_DIR/base_endpoint.stdout.log"
BASE_ENDPOINT_STDERR="$ARTIFACT_DIR/base_endpoint.stderr.log"
BASE_ENDPOINT_INFO="$ARTIFACT_DIR/base_endpoint.endpoint.txt"
FINETUNED_ENDPOINT_STDOUT="$ARTIFACT_DIR/finetuned_endpoint.stdout.log"
FINETUNED_ENDPOINT_STDERR="$ARTIFACT_DIR/finetuned_endpoint.stderr.log"
FINETUNED_ENDPOINT_INFO="$ARTIFACT_DIR/finetuned_endpoint.endpoint.txt"
BASE_EVAL_DIR="$ARTIFACT_DIR/base_eval"
FINETUNED_EVAL_DIR="$ARTIFACT_DIR/finetuned_eval"
COMPARISON_PATH="$ARTIFACT_DIR/comparison_summary.json"
RESULT_MANIFEST_PATH="$ARTIFACT_DIR/result_manifest.json"

STUDENT_MODEL="${NANOHORIZON_STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
THINKING_BUDGET="${NANOHORIZON_THINKING_BUDGET_TOKENS:-2000}"
MAX_NEW_TOKENS="${NANOHORIZON_MAX_NEW_TOKENS:-2200}"
MAX_MODEL_LEN="${NANOHORIZON_MAX_MODEL_LEN:-8192}"
NUM_EVAL_ROLLOUTS="${NANOHORIZON_NUM_EVAL_ROLLOUTS:-8}"
EVAL_SEED_START="${NANOHORIZON_EVAL_SEED_START:-10000}"
EVAL_MAX_STEPS="${NANOHORIZON_EVAL_MAX_STEPS:-8}"
EVAL_MAX_CONCURRENCY="${NANOHORIZON_EVAL_MAX_CONCURRENCY:-4}"
EVAL_REQUEST_TIMEOUT_SECONDS="${NANOHORIZON_EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
BOOTSTRAP_SEED_COUNT="${NANOHORIZON_PIVOTRL_BOOTSTRAP_SEED_COUNT:-32}"
BOOTSTRAP_MAX_STEPS="${NANOHORIZON_PIVOTRL_BOOTSTRAP_MAX_STEPS:-48}"
BOOTSTRAP_ROLLOUT_CONCURRENCY="${NANOHORIZON_PIVOTRL_BOOTSTRAP_CONCURRENCY:-4}"
BOOTSTRAP_MAX_NEW_TOKENS="${NANOHORIZON_PIVOTRL_BOOTSTRAP_MAX_NEW_TOKENS:-3072}"
BOOTSTRAP_THINKING_BUDGET_TOKENS="${NANOHORIZON_PIVOTRL_BOOTSTRAP_THINKING_BUDGET_TOKENS:-2000}"
PROFILE_K="${NANOHORIZON_PIVOTRL_PROFILE_K:-4}"
LAMBDA_DIFF="${NANOHORIZON_PIVOTRL_LAMBDA_DIFF:-0.75}"
MAX_PIVOTS="${NANOHORIZON_PIVOTRL_MAX_PIVOTS:-128}"
MIN_KEPT_PIVOTS="${NANOHORIZON_PIVOTRL_MIN_KEPT_PIVOTS:-4}"
GROUP_SIZE="${NANOHORIZON_PIVOTRL_GROUP_SIZE:-4}"
TRAIN_ITERATIONS="${NANOHORIZON_PIVOTRL_TRAIN_ITERATIONS:-2}"
TRAIN_STEPS_PER_ITERATION="${NANOHORIZON_PIVOTRL_TRAIN_STEPS_PER_ITERATION:-8}"
MAX_TRAIN_STEPS="${NANOHORIZON_PIVOTRL_MAX_TRAIN_STEPS:-16}"
PIVOTS_PER_ITERATION="${NANOHORIZON_PIVOTRL_PIVOTS_PER_ITERATION:-16}"
PROFILE_MAX_NEW_TOKENS="${NANOHORIZON_PIVOTRL_PROFILE_MAX_NEW_TOKENS:-96}"
SAMPLE_MAX_NEW_TOKENS="${NANOHORIZON_PIVOTRL_SAMPLE_MAX_NEW_TOKENS:-96}"
VLLM_API_KEY="${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"
BOOTSTRAP_CONTAINER_URL_OVERRIDE="${NANOHORIZON_PIVOTRL_BOOTSTRAP_CONTAINER_URL:-}"
BOOTSTRAP_CONTAINER_WORKER_TOKEN_OVERRIDE="${NANOHORIZON_PIVOTRL_BOOTSTRAP_CONTAINER_WORKER_TOKEN:-}"

mkdir -p "$ARTIFACT_DIR" "$BASE_EVAL_DIR" "$FINETUNED_EVAL_DIR"

source "$ROOT/scripts/lib_craftax_tunnel.sh"

cleanup() {
  if [[ -n "${BASE_ENDPOINT_PID:-}" ]]; then
    nanohorizon_stop_modal_endpoint "$BASE_ENDPOINT_PID"
  fi
  if [[ -n "${FINETUNED_ENDPOINT_PID:-}" ]]; then
    nanohorizon_stop_modal_endpoint "$FINETUNED_ENDPOINT_PID"
  fi
  nanohorizon_cleanup_craftax_tunnel
}
trap cleanup EXIT

run_local_eval() {
  local output_dir="$1"
  local inference_url="$2"
  local request_model="$3"
  local -a cmd=(
    "$UV_BIN" run python -m nanohorizon.shared.eval_model
    --base-model "$STUDENT_MODEL"
    --container-url "${NANOHORIZON_CRAFTAX_CONTAINER_URL}"
    --output-dir "$output_dir"
    --seed-start "$EVAL_SEED_START"
    --num-rollouts "$NUM_EVAL_ROLLOUTS"
    --max-steps "$EVAL_MAX_STEPS"
    --max-concurrent-rollouts "$EVAL_MAX_CONCURRENCY"
    --max-length "$MAX_MODEL_LEN"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --thinking-budget-tokens "$THINKING_BUDGET"
    --request-timeout-seconds "$EVAL_REQUEST_TIMEOUT_SECONDS"
    --inference-url "$inference_url"
    --inference-api-key "$VLLM_API_KEY"
    --request-model "$request_model"
  )
  if [[ "$THINKING_BUDGET" -gt 0 ]]; then
    cmd+=(--enable-thinking)
  fi
  "${cmd[@]}"
}

wait_for_inference_chat_url() {
  local chat_url="$1"
  local api_key="$2"
  local base_url="${chat_url%/v1/chat/completions}"
  nanohorizon_wait_for_openai_compat_endpoint "$base_url" "$api_key" 240 2 1
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NANOHORIZON_MODAL_GPU_OFFLINE="${NANOHORIZON_MODAL_GPU_OFFLINE:-A100-40GB}"
export NANOHORIZON_MODAL_GPU_TEACHER="${NANOHORIZON_MODAL_GPU_TEACHER:-A100-40GB}"
export NANOHORIZON_TEACHER_API_KEY="$VLLM_API_KEY"
export NANOHORIZON_VLLM_API_KEY="$VLLM_API_KEY"

nanohorizon_load_synth_auth
nanohorizon_open_craftax_tunnel_if_needed "$ROOT"

BOOTSTRAP_CONTAINER_URL="${BOOTSTRAP_CONTAINER_URL_OVERRIDE:-${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}}"
BOOTSTRAP_CONTAINER_WORKER_TOKEN="${BOOTSTRAP_CONTAINER_WORKER_TOKEN_OVERRIDE:-${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-}}"

if [[ -z "$BOOTSTRAP_CONTAINER_URL" ]]; then
  echo "missing Craftax container URL for PivotRL bootstrap" >&2
  exit 1
fi

if [[ -z "${NANOHORIZON_TEACHER_INFERENCE_URL:-}" ]]; then
  nanohorizon_start_modal_endpoint \
    "$BOOTSTRAP_ENDPOINT_STDOUT" \
    "$BOOTSTRAP_ENDPOINT_STDERR" \
    nanohorizon-craftax-pivotrl-bootstrap-teacher \
    "$TEACHER_MODEL" \
    "$MAX_MODEL_LEN" \
    "" \
    "" \
    "16" >"$BOOTSTRAP_ENDPOINT_INFO"
  NANOHORIZON_MODAL_TEACHER_PID="$(sed -n '1p' "$BOOTSTRAP_ENDPOINT_INFO")"
  export NANOHORIZON_MODAL_TEACHER_PID
  export NANOHORIZON_TEACHER_INFERENCE_URL="$(sed -n '2p' "$BOOTSTRAP_ENDPOINT_INFO")"
fi

wait_for_inference_chat_url "$NANOHORIZON_TEACHER_INFERENCE_URL" "$NANOHORIZON_TEACHER_API_KEY"

"$UV_BIN" run --group modal modal run submissions/synth/pivotrl.py \
  --output-dir "$REMOTE_OUTPUT_DIR" \
  --base-model "$STUDENT_MODEL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-inference-url "$NANOHORIZON_TEACHER_INFERENCE_URL" \
  --teacher-api-key "$NANOHORIZON_TEACHER_API_KEY" \
  --craftax-container-url "$BOOTSTRAP_CONTAINER_URL" \
  --craftax-container-worker-token "$BOOTSTRAP_CONTAINER_WORKER_TOKEN" \
  --bootstrap-rollouts-path "${NANOHORIZON_PIVOTRL_BOOTSTRAP_ROLLOUTS_PATH:-}" \
  --max-length "$MAX_MODEL_LEN" \
  --request-timeout-seconds "$EVAL_REQUEST_TIMEOUT_SECONDS" \
  --bootstrap-seed-count "$BOOTSTRAP_SEED_COUNT" \
  --bootstrap-max-steps "$BOOTSTRAP_MAX_STEPS" \
  --bootstrap-max-new-tokens "$BOOTSTRAP_MAX_NEW_TOKENS" \
  --bootstrap-thinking-budget-tokens "$BOOTSTRAP_THINKING_BUDGET_TOKENS" \
  --bootstrap-rollout-concurrency "$BOOTSTRAP_ROLLOUT_CONCURRENCY" \
  --bootstrap-rollout-semaphore-limit "$BOOTSTRAP_ROLLOUT_CONCURRENCY" \
  --profile-k "$PROFILE_K" \
  --lambda-diff "$LAMBDA_DIFF" \
  --profile-max-new-tokens "$PROFILE_MAX_NEW_TOKENS" \
  --max-pivots "$MAX_PIVOTS" \
  --min-kept-pivots "$MIN_KEPT_PIVOTS" \
  --group-size "$GROUP_SIZE" \
  --train-iterations "$TRAIN_ITERATIONS" \
  --train-steps-per-iteration "$TRAIN_STEPS_PER_ITERATION" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --pivots-per-iteration "$PIVOTS_PER_ITERATION" \
  --sample-max-new-tokens "$SAMPLE_MAX_NEW_TOKENS" \
  --enable-thinking \
  --local-result-path "$MODAL_TRAIN_RESULT_PATH" | tee "$TRAINING_LOG"

if [[ -n "${NANOHORIZON_MODAL_TEACHER_PID:-}" ]]; then
  nanohorizon_stop_modal_endpoint "$NANOHORIZON_MODAL_TEACHER_PID"
  unset NANOHORIZON_MODAL_TEACHER_PID
fi

ADAPTER_DIR="$("$UV_BIN" run python3 - <<'PY' "$MODAL_TRAIN_RESULT_PATH"
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = str(payload.get("adapter_dir") or "").strip()
if not value:
    raise SystemExit(1)
print(value)
PY
)"

nanohorizon_start_modal_endpoint \
  "$BASE_ENDPOINT_STDOUT" \
  "$BASE_ENDPOINT_STDERR" \
  nanohorizon-craftax-pivotrl-base \
  "$STUDENT_MODEL" \
  "$MAX_MODEL_LEN" \
  "" \
  "" \
  "16" >"$BASE_ENDPOINT_INFO"
BASE_ENDPOINT_PID="$(sed -n '1p' "$BASE_ENDPOINT_INFO")"
BASE_INFERENCE_URL="$(sed -n '2p' "$BASE_ENDPOINT_INFO")"
wait_for_inference_chat_url "$BASE_INFERENCE_URL" "$VLLM_API_KEY"
run_local_eval "$BASE_EVAL_DIR" "$BASE_INFERENCE_URL" "$STUDENT_MODEL"
nanohorizon_stop_modal_endpoint "$BASE_ENDPOINT_PID"
unset BASE_ENDPOINT_PID

nanohorizon_start_modal_endpoint \
  "$FINETUNED_ENDPOINT_STDOUT" \
  "$FINETUNED_ENDPOINT_STDERR" \
  nanohorizon-craftax-pivotrl-finetuned \
  "$STUDENT_MODEL" \
  "$MAX_MODEL_LEN" \
  "policy-lora" \
  "$ADAPTER_DIR" \
  "16" >"$FINETUNED_ENDPOINT_INFO"
FINETUNED_ENDPOINT_PID="$(sed -n '1p' "$FINETUNED_ENDPOINT_INFO")"
FINETUNED_INFERENCE_URL="$(sed -n '2p' "$FINETUNED_ENDPOINT_INFO")"
wait_for_inference_chat_url "$FINETUNED_INFERENCE_URL" "$VLLM_API_KEY"
run_local_eval "$FINETUNED_EVAL_DIR" "$FINETUNED_INFERENCE_URL" "policy-lora"

"$UV_BIN" run python3 - <<'PY' "$BASE_EVAL_DIR/eval_summary.json" "$FINETUNED_EVAL_DIR/eval_summary.json" "$COMPARISON_PATH" "$ADAPTER_DIR" "$REMOTE_OUTPUT_DIR"
from pathlib import Path
import json
import sys

base_summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
finetuned_summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
comparison = {
    "adapter_dir": sys.argv[4],
    "remote_output_dir": sys.argv[5],
    "base_mean_outcome_reward": float(base_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "finetuned_mean_outcome_reward": float(finetuned_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "reward_delta": float(finetuned_summary.get("mean_outcome_reward", 0.0) or 0.0)
    - float(base_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "base_num_eval_rollouts": int(base_summary.get("num_eval_rollouts", 0) or 0),
    "finetuned_num_eval_rollouts": int(finetuned_summary.get("num_eval_rollouts", 0) or 0),
}
Path(sys.argv[3]).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, indent=2, sort_keys=True))
PY

"$UV_BIN" run python3 - <<'PY' "$MODAL_TRAIN_RESULT_PATH" "$COMPARISON_PATH" "$RESULT_MANIFEST_PATH" "$ARTIFACT_DIR" "$BASE_EVAL_DIR/eval_summary.json" "$FINETUNED_EVAL_DIR/eval_summary.json"
from pathlib import Path
import json
import sys

train_result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
comparison = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
base_eval = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
finetuned_eval = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
manifest = {
    "method_name": "pivotrl_preachievement_offline",
    "artifact_dir": sys.argv[4],
    "remote_output_dir": train_result.get("output_root"),
    "adapter_dir": train_result.get("adapter_dir"),
    "bootstrap_summary": train_result.get("bootstrap_summary", {}),
    "pivot_profile_summary": train_result.get("pivot_profile_summary", {}),
    "training_summary": train_result.get("training_summary", {}),
    "base_eval_summary": base_eval,
    "finetuned_eval_summary": finetuned_eval,
    "comparison_summary": comparison,
}
Path(sys.argv[3]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
