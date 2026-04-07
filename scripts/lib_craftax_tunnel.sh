#!/usr/bin/env bash

nanohorizon_load_synth_auth() {
  if [[ -z "${SYNTH_API_KEY:-}" && -f "/Users/joshpurtell/Documents/GitHub/synth-ai/.env" ]]; then
    export SYNTH_API_KEY="$(awk -F= '/^SYNTH_API_KEY=/{print $2; exit}' /Users/joshpurtell/Documents/GitHub/synth-ai/.env)"
  fi
  export SYNTH_BACKEND_URL="${SYNTH_BACKEND_URL:-https://api.usesynth.ai}"
}


nanohorizon_craftax_local_port() {
  printf '%s\n' "${NANOHORIZON_CRAFTAX_LOCAL_PORT:-8913}"
}


nanohorizon_craftax_local_shim_count() {
  if [[ -n "${NANOHORIZON_CRAFTAX_LOCAL_SHIMS:-}" ]]; then
    printf '%s\n' "${NANOHORIZON_CRAFTAX_LOCAL_SHIMS}"
    return 0
  fi
  "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY'
import os
import subprocess

def _sysctl(name: str) -> int | None:
    try:
        value = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        return int(value)
    except Exception:
        return None

perf = _sysctl("hw.perflevel0.physicalcpu_max")
logical = _sysctl("hw.ncpu") or (os.cpu_count() or 1)
candidate = perf or logical
print(max(1, min(int(candidate), 8)))
PY
}


nanohorizon_craftax_local_urls() {
  local base_port
  base_port="$(nanohorizon_craftax_local_port)"
  local count
  count="$(nanohorizon_craftax_local_shim_count)"
  "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$base_port" "$count"
import sys

base = int(sys.argv[1])
count = max(1, int(sys.argv[2]))
print(",".join(f"http://127.0.0.1:{base + offset}" for offset in range(count)))
PY
}


nanohorizon_craftax_service_ready() {
  local port="$1"
  "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$port"
import json
import sys
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/health"
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)

service = str(payload.get("service") or "").strip().lower()
if service == "craftax_core_http_shim":
    raise SystemExit(0)
raise SystemExit(1)
PY
}


nanohorizon_check_craftax_core_layout() {
  return 0
}


nanohorizon_start_local_craftax_if_needed() {
  local root="$1"
  if [[ -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" || -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" ]]; then
    return 0
  fi
  local local_port
  local_port="$(nanohorizon_craftax_local_port)"
  local shim_count
  shim_count="$(nanohorizon_craftax_local_shim_count)"
  local ready_count=0
  for offset in $(seq 0 $((shim_count - 1))); do
    if nanohorizon_craftax_service_ready "$((local_port + offset))"; then
      ready_count=$((ready_count + 1))
    fi
  done
  if [[ "$ready_count" -eq "$shim_count" ]]; then
    return 0
  fi

  local artifact_dir="${NANOHORIZON_CRAFTAX_TUNNEL_ARTIFACT_DIR:-$root/.out/craftax_tunnel}"
  mkdir -p "$artifact_dir"
  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
  local pids=()
  for offset in $(seq 0 $((shim_count - 1))); do
    local port=$((local_port + offset))
    local container_log="$artifact_dir/craftax_core_http_shim_${port}.log"
    (
      cd "$root"
      NANOHORIZON_CRAFTAX_BIND_HOST="127.0.0.1" \
      NANOHORIZON_CRAFTAX_BIND_PORT="$port" \
      NANOHORIZON_CRAFTAX_UVICORN_WORKERS="1" \
      PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$uv_bin" run python -m nanohorizon.craftax_core.http_shim
    ) >"$container_log" 2>&1 &
    pids+=("$!")
  done
  NANOHORIZON_LOCAL_CRAFTAX_PIDS="${pids[*]}"
  export NANOHORIZON_LOCAL_CRAFTAX_PIDS

  for _ in $(seq 1 180); do
    ready_count=0
    for offset in $(seq 0 $((shim_count - 1))); do
      local port=$((local_port + offset))
      if nanohorizon_craftax_service_ready "$port"; then
        ready_count=$((ready_count + 1))
      fi
    done
    if [[ "$ready_count" -eq "$shim_count" ]]; then
      return 0
    fi
    for pid in "${pids[@]}"; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        for offset in $(seq 0 $((shim_count - 1))); do
          tail -n 80 "$artifact_dir/craftax_core_http_shim_$((local_port + offset)).log" >&2 || true
        done
        return 1
      fi
    done
    sleep 1
  done

  for offset in $(seq 0 $((shim_count - 1))); do
    tail -n 80 "$artifact_dir/craftax_core_http_shim_$((local_port + offset)).log" >&2 || true
  done
  return 1
}


nanohorizon_open_craftax_tunnel_if_needed() {
  local root="$1"
  if [[ -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" || -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" ]]; then
    return 0
  fi

  nanohorizon_load_synth_auth
  nanohorizon_start_local_craftax_if_needed "$root"
  local local_port
  local_port="$(nanohorizon_craftax_local_port)"

  local artifact_dir="${NANOHORIZON_CRAFTAX_TUNNEL_ARTIFACT_DIR:-$root/.out/craftax_tunnel}"
  local tunnel_env="$artifact_dir/tunnel.env"
  local tunnel_json="$artifact_dir/tunnel.json"
  local tunnel_stdout="$artifact_dir/tunnel.stdout.log"
  local tunnel_stderr="$artifact_dir/tunnel.stderr.log"
  mkdir -p "$artifact_dir"
  rm -f "$tunnel_env" "$tunnel_json"

  local backend="${NANOHORIZON_CRAFTAX_TUNNEL_BACKEND:-synthtunnel}"
  local managed_ngrok_url="${NANOHORIZON_MANAGED_NGROK_URL:-}"
  local quick_tunnel_wait_seconds="${NANOHORIZON_CRAFTAX_TUNNEL_QUICK_WAIT_SECONDS:-30}"
  local tunnel_script_path="${NANOHORIZON_TUNNEL_SCRIPT_PATH:-$root/scripts/open_craftax_tunnel.py}"
  local synth_ai_root="${NANOHORIZON_SYNTH_AI_ROOT:-$root/../synth-ai}"
  local synth_ai_python="${NANOHORIZON_SYNTH_AI_PYTHON:-$synth_ai_root/.venv/bin/python}"
  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
  local tunnel_python="${NANOHORIZON_TUNNEL_PYTHON:-}"
  if [[ -z "$tunnel_python" ]]; then
    if [[ -x "$synth_ai_python" ]]; then
      tunnel_python="$synth_ai_python"
    elif [[ -x "$root/.venv/bin/python" ]]; then
      tunnel_python="$root/.venv/bin/python"
    else
      tunnel_python="$(command -v python3 || true)"
    fi
  fi
  if [[ ! -f "$tunnel_script_path" ]]; then
    echo "missing tunnel script: $tunnel_script_path" >&2
    return 1
  fi
  if [[ -z "$tunnel_python" || ! -x "$tunnel_python" ]]; then
    echo "missing python interpreter for tunnel helper; set NANOHORIZON_TUNNEL_PYTHON" >&2
    return 1
  fi

  local cmd=()
  if [[ -x "$uv_bin" && -z "${NANOHORIZON_TUNNEL_FORCE_PLAIN_PYTHON:-}" ]]; then
    cmd=(
      "$uv_bin" run python "$tunnel_script_path"
      --backend "$backend"
      --backend-url "$SYNTH_BACKEND_URL"
      --local-port "$local_port"
      --local-base-url "http://127.0.0.1:${local_port}"
      --env-file "$tunnel_env"
      --json-file "$tunnel_json"
      --hold
    )
  else
    cmd=(
      "$tunnel_python" "$tunnel_script_path"
      --backend "$backend"
      --backend-url "$SYNTH_BACKEND_URL"
      --local-port "$local_port"
      --local-base-url "http://127.0.0.1:${local_port}"
      --env-file "$tunnel_env"
      --json-file "$tunnel_json"
      --hold
    )
  fi
  local requested_ttl="${NANOHORIZON_CRAFTAX_TUNNEL_TTL_SECONDS:-7200}"
  if [[ "$requested_ttl" -gt 0 ]]; then
    cmd+=(--requested-ttl-seconds "$requested_ttl")
  fi
  if [[ -n "$managed_ngrok_url" ]]; then
    cmd+=(--managed-ngrok-url "$managed_ngrok_url")
  fi
  if [[ "$backend" == "cloudflare_quick" ]]; then
    cmd+=(--quick-tunnel-wait-seconds "$quick_tunnel_wait_seconds")
  fi

  PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
  if [[ -d "$synth_ai_root" ]]; then
    PYTHONPATH="$synth_ai_root:$PYTHONPATH"
  fi

  env PYTHONPATH="$PYTHONPATH" "${cmd[@]}" >"$tunnel_stdout" 2>"$tunnel_stderr" &
  NANOHORIZON_CRAFTAX_TUNNEL_PID=$!
  export NANOHORIZON_CRAFTAX_TUNNEL_PID

  for _ in $(seq 1 120); do
    if [[ -s "$tunnel_env" ]]; then
      # shellcheck disable=SC1090
      source "$tunnel_env"
      if [[ -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" || -n "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" ]]; then
        export NANOHORIZON_CRAFTAX_CONTAINER_URL="${NANOHORIZON_CRAFTAX_CONTAINER_URL:-${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}}"
        export NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-}}"
        export NANOHORIZON_CRAFTAX_CONTAINER_URL="${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}"
        export NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-}"
        return 0
      fi
    fi
    if ! kill -0 "$NANOHORIZON_CRAFTAX_TUNNEL_PID" >/dev/null 2>&1; then
      cat "$tunnel_stderr" >&2 || true
      return 1
    fi
    sleep 1
  done

  cat "$tunnel_stderr" >&2 || true
  return 1
}


nanohorizon_cleanup_craftax_tunnel() {
  if [[ -n "${NANOHORIZON_MODAL_TEACHER_PID:-}" ]]; then
    nanohorizon_stop_modal_endpoint "$NANOHORIZON_MODAL_TEACHER_PID"
  fi
  if [[ -n "${NANOHORIZON_CRAFTAX_TUNNEL_PID:-}" ]]; then
    kill "$NANOHORIZON_CRAFTAX_TUNNEL_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NANOHORIZON_LOCAL_CRAFTAX_PIDS:-}" ]]; then
    for pid in ${NANOHORIZON_LOCAL_CRAFTAX_PIDS}; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}


nanohorizon_wait_for_openai_compat_endpoint() {
  local base_url="$1"
  local api_key="$2"
  local startup_attempts="${3:-180}"
  local sleep_seconds="${4:-2}"
  local skip_warmup="${5:-0}"

  local normalized_base="${base_url%/}"
  if [[ "$normalized_base" == */v1 ]]; then
    local models_url="${normalized_base}/models"
    local warmup_url="${normalized_base}/chat/completions"
  else
    local models_url="${normalized_base}/v1/models"
    local warmup_url="${normalized_base}/v1/chat/completions"
  fi
  local warmup_model="${NANOHORIZON_TEACHER_SERVED_MODEL_NAME:-${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-9B}}"

  for _ in $(seq 1 "$startup_attempts"); do
    if "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$models_url" "$api_key" >/dev/null 2>&1
import sys
import urllib.request

url = sys.argv[1]
api_key = sys.argv[2]
request = urllib.request.Request(
    url,
    method="GET",
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(request, timeout=15) as response:
    if int(getattr(response, "status", 0)) != 200:
        raise SystemExit(1)
PY
    then
      if [[ "$skip_warmup" != "1" ]]; then
        "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$warmup_url" "$api_key" "$warmup_model" >/dev/null 2>&1 || true
import json
import sys
import urllib.request

url = sys.argv[1]
api_key = sys.argv[2]
model = sys.argv[3]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 8,
    "temperature": 0.0,
}
request = urllib.request.Request(
    url,
    method="POST",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=30):
    pass
PY
      fi
      return 0
    fi
    sleep "$sleep_seconds"
  done

  return 1
}


nanohorizon_extract_modal_endpoint_url() {
  local log_stdout="$1"
  "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$log_stdout"
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
for raw in text.splitlines():
    if raw.startswith("vLLM endpoint: "):
        print(raw.split("vLLM endpoint: ", 1)[1].strip())
        raise SystemExit(0)
match = re.search(r"https://[A-Za-z0-9._/-]*modal\.run", text.replace("\n", ""))
if match:
    print(match.group(0).strip())
PY
}


nanohorizon_extract_modal_app_id() {
  local log_stdout="$1"
  "${UV_BIN:-/opt/homebrew/bin/uv}" run python3 - <<'PY' "$log_stdout"
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
match = re.search(r"/(ap-[A-Za-z0-9]+)\b", text)
if match:
    print(match.group(1))
PY
}


nanohorizon_stop_modal_endpoint() {
  local handle="${1:-}"
  if [[ -z "$handle" ]]; then
    return 0
  fi
  if [[ "$handle" =~ ^ap-[A-Za-z0-9]+$ ]]; then
    local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
    "$uv_bin" run --group modal modal app stop "$handle" >/dev/null 2>&1 || true
    return 0
  fi
  if [[ "$handle" =~ ^[0-9]+$ ]]; then
    kill "$handle" >/dev/null 2>&1 || true
  fi
}


nanohorizon_start_modal_endpoint() {
  local log_stdout="$1"
  local log_stderr="$2"
  local app_name="$3"
  local model_ref="$4"
  local max_model_len="$5"
  local lora_name="${6:-}"
  local lora_path="${7:-}"
  local max_lora_rank="${8:-16}"

  : >"$log_stdout"
  : >"$log_stderr"

  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
  local api_key="${NANOHORIZON_TEACHER_API_KEY:-${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}}"
  local keepalive_s="${NANOHORIZON_MODAL_TEACHER_KEEPALIVE_S:-7200}"
  local launch_mode="${NANOHORIZON_MODAL_TEACHER_LAUNCH_MODE:-detach}"

  local -a cmd=(
    "$uv_bin" run --group modal modal run
  )
  if [[ "$launch_mode" == "detach" ]]; then
    cmd+=(--detach)
  fi
  cmd+=(src/nanohorizon/shared/modal_teacher.py --keepalive-s "$keepalive_s")

  env \
    COLUMNS=200 \
    NANOHORIZON_MODAL_TEACHER_APP_NAME="$app_name" \
    NANOHORIZON_TEACHER_MODEL="$model_ref" \
    NANOHORIZON_TEACHER_API_KEY="$api_key" \
    NANOHORIZON_TEACHER_MAX_MODEL_LEN="$max_model_len" \
    NANOHORIZON_TEACHER_LORA_NAME="$lora_name" \
    NANOHORIZON_TEACHER_LORA_PATH="$lora_path" \
    NANOHORIZON_TEACHER_MAX_LORA_RANK="$max_lora_rank" \
    "${cmd[@]}" >"$log_stdout" 2>"$log_stderr" &
  local pid=$!
  for _ in $(seq 1 240); do
    if [[ -f "$log_stdout" ]]; then
      local url
      url="$(nanohorizon_extract_modal_endpoint_url "$log_stdout")"
      if [[ -n "$url" ]]; then
        if ! nanohorizon_wait_for_openai_compat_endpoint "$url" "$api_key" 240 2; then
          cat "$log_stderr" >&2 || true
          cat "$log_stdout" >&2 || true
          return 1
        fi
        local handle="$pid"
        local app_id
        app_id="$(nanohorizon_extract_modal_app_id "$log_stdout")"
        if [[ -n "$app_id" ]]; then
          handle="$app_id"
        fi
        printf '%s\n%s\n' "$handle" "${url%/}/v1/chat/completions"
        return 0
      fi
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      cat "$log_stderr" >&2 || true
      cat "$log_stdout" >&2 || true
      return 1
    fi
    sleep 2
  done

  cat "$log_stderr" >&2 || true
  cat "$log_stdout" >&2 || true
  return 1
}


nanohorizon_deploy_modal_endpoint() {
  local log_stdout="$1"
  local log_stderr="$2"
  local app_name="$3"
  local model_ref="$4"
  local max_model_len="$5"
  local lora_name="${6:-}"
  local lora_path="${7:-}"
  local max_lora_rank="${8:-16}"

  : >"$log_stdout"
  : >"$log_stderr"

  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
  local api_key="${NANOHORIZON_TEACHER_API_KEY:-${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}}"

  if ! env \
    COLUMNS=200 \
    NANOHORIZON_MODAL_TEACHER_APP_NAME="$app_name" \
    NANOHORIZON_TEACHER_MODEL="$model_ref" \
    NANOHORIZON_TEACHER_API_KEY="$api_key" \
    NANOHORIZON_TEACHER_MAX_MODEL_LEN="$max_model_len" \
    NANOHORIZON_TEACHER_LORA_NAME="$lora_name" \
    NANOHORIZON_TEACHER_LORA_PATH="$lora_path" \
    NANOHORIZON_TEACHER_MAX_LORA_RANK="$max_lora_rank" \
    "$uv_bin" run --group modal modal deploy src/nanohorizon/shared/modal_teacher.py --name "$app_name" >"$log_stdout" 2>"$log_stderr"
  then
    cat "$log_stderr" >&2 || true
    cat "$log_stdout" >&2 || true
    return 1
  fi

  local url
  url="$(nanohorizon_extract_modal_endpoint_url "$log_stdout")"
  if [[ -z "$url" ]]; then
    cat "$log_stdout" >&2 || true
    return 1
  fi
  printf 'deployed\n%s/v1/chat/completions\n' "${url%/}"
}


nanohorizon_start_modal_teacher_if_needed() {
  local root="$1"
  if [[ -n "${NANOHORIZON_TEACHER_INFERENCE_URL:-}" ]]; then
    return 0
  fi

  local artifact_dir="${NANOHORIZON_CRAFTAX_TUNNEL_ARTIFACT_DIR:-$root/.out/craftax_tunnel}"
  local teacher_stdout="$artifact_dir/teacher.stdout.log"
  local teacher_stderr="$artifact_dir/teacher.stderr.log"
  mkdir -p "$artifact_dir"
  local teacher_model="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
  local teacher_api_key="${NANOHORIZON_TEACHER_API_KEY:-${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}}"
  local teacher_app_name="${NANOHORIZON_MODAL_TEACHER_APP_NAME:-nanohorizon-craftax-teacher}"
  local teacher_max_model_len="${NANOHORIZON_TEACHER_MAX_MODEL_LEN:-4096}"
  local teacher_launch_mode="${NANOHORIZON_MODAL_TEACHER_LAUNCH_MODE:-detach}"

  local endpoint_raw
  if [[ "$teacher_launch_mode" == "deploy" ]]; then
    endpoint_raw="$(nanohorizon_deploy_modal_endpoint \
      "$teacher_stdout" \
      "$teacher_stderr" \
      "$teacher_app_name" \
      "$teacher_model" \
      "$teacher_max_model_len" \
      "" \
      "" \
      "16")"
  else
    endpoint_raw="$(nanohorizon_start_modal_endpoint \
      "$teacher_stdout" \
      "$teacher_stderr" \
      "$teacher_app_name" \
      "$teacher_model" \
      "$teacher_max_model_len" \
      "" \
      "" \
      "16")"
  fi
  local teacher_pid_raw
  teacher_pid_raw="$(printf '%s\n' "$endpoint_raw" | sed -n '1p')"
  if [[ "$teacher_pid_raw" =~ ^[0-9]+$ ]]; then
    NANOHORIZON_MODAL_TEACHER_PID="$teacher_pid_raw"
  else
    NANOHORIZON_MODAL_TEACHER_PID=""
  fi
  export NANOHORIZON_MODAL_TEACHER_PID
  export NANOHORIZON_TEACHER_INFERENCE_URL="$(printf '%s\n' "$endpoint_raw" | sed -n '2p')"
  export NANOHORIZON_TEACHER_API_KEY="$teacher_api_key"
  if [[ -n "${NANOHORIZON_TEACHER_INFERENCE_URL:-}" ]]; then
    nanohorizon_wait_for_openai_compat_endpoint "${NANOHORIZON_TEACHER_INFERENCE_URL%/v1/chat/completions}" "$teacher_api_key" 240 2
  fi
  return 0
}
