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


nanohorizon_craftax_service_ready() {
  local port="$1"
  python3 - <<'PY' "$port"
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
  if nanohorizon_craftax_service_ready "$local_port"; then
    return 0
  fi

  local artifact_dir="${NANOHORIZON_CRAFTAX_TUNNEL_ARTIFACT_DIR:-$root/.out/craftax_tunnel}"
  mkdir -p "$artifact_dir"
  local container_log="$artifact_dir/craftax_core_http_shim.log"
  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"

  (
    cd "$root"
    NANOHORIZON_CRAFTAX_BIND_HOST="127.0.0.1" \
    NANOHORIZON_CRAFTAX_BIND_PORT="$local_port" \
    PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$uv_bin" run python -m nanohorizon.craftax_core.http_shim
  ) >"$container_log" 2>&1 &
  NANOHORIZON_LOCAL_CRAFTAX_PID=$!
  export NANOHORIZON_LOCAL_CRAFTAX_PID

  for _ in $(seq 1 120); do
    if nanohorizon_craftax_service_ready "$local_port"; then
      return 0
    fi
    if ! kill -0 "$NANOHORIZON_LOCAL_CRAFTAX_PID" >/dev/null 2>&1; then
      tail -n 80 "$container_log" >&2 || true
      return 1
    fi
    sleep 1
  done

  tail -n 80 "$container_log" >&2 || true
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
    kill "$NANOHORIZON_MODAL_TEACHER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NANOHORIZON_CRAFTAX_TUNNEL_PID:-}" ]]; then
    kill "$NANOHORIZON_CRAFTAX_TUNNEL_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NANOHORIZON_LOCAL_CRAFTAX_PID:-}" ]]; then
    kill "$NANOHORIZON_LOCAL_CRAFTAX_PID" >/dev/null 2>&1 || true
  fi
}


nanohorizon_wait_for_openai_compat_endpoint() {
  local base_url="$1"
  local api_key="$2"
  local startup_attempts="${3:-180}"
  local sleep_seconds="${4:-2}"

  local models_url="${base_url%/}/v1/models"
  local warmup_url="${base_url%/}/v1/chat/completions"

  for _ in $(seq 1 "$startup_attempts"); do
    if curl --max-time 10 -sf -H "Authorization: Bearer $api_key" "$models_url" >/dev/null 2>&1; then
      python3 - <<'PY' "$warmup_url" "$api_key" >/dev/null 2>&1 || true
import json
import sys
import urllib.request

url = sys.argv[1]
api_key = sys.argv[2]
payload = {
    "model": "Qwen/Qwen3.5-9B",
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
      return 0
    fi
    sleep "$sleep_seconds"
  done

  return 1
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

  local uv_bin="${UV_BIN:-/opt/homebrew/bin/uv}"
  local teacher_model="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
  local teacher_api_key="${NANOHORIZON_TEACHER_API_KEY:-dummy-local-key}"
  local teacher_app_name="${NANOHORIZON_MODAL_TEACHER_APP_NAME:-nanohorizon-craftax-teacher}"
  local code_version="${NANOHORIZON_CODE_VERSION:-$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

  if ! env \
    NANOHORIZON_MODAL_TEACHER_APP_NAME="$teacher_app_name" \
    NANOHORIZON_TEACHER_MODEL="$teacher_model" \
    NANOHORIZON_TEACHER_API_KEY="$teacher_api_key" \
    NANOHORIZON_CODE_VERSION="$code_version" \
    "$uv_bin" run --group modal modal deploy src/nanohorizon/shared/modal_teacher.py \
    >"$teacher_stdout" 2>"$teacher_stderr"; then
    cat "$teacher_stderr" >&2 || true
    cat "$teacher_stdout" >&2 || true
    return 1
  fi

  local line
  line="$(python3 - <<'PY' "$teacher_stdout"
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8') if path.exists() else ''
collapsed = ''.join(text.split())
match = re.search(r'https://[A-Za-z0-9./:-]+modal\.run', collapsed)
if match:
    print(match.group(0))
PY
)"
  if [[ -z "$line" ]]; then
    cat "$teacher_stderr" >&2 || true
    cat "$teacher_stdout" >&2 || true
    return 1
  fi

  local probe_attempts="${NANOHORIZON_TEACHER_ENDPOINT_PROBE_ATTEMPTS:-${NANOHORIZON_TEACHER_STARTUP_ATTEMPTS:-240}}"
  if ! nanohorizon_wait_for_openai_compat_endpoint \
    "${line%/}" \
    "$teacher_api_key" \
    "$probe_attempts" \
    "${NANOHORIZON_TEACHER_STARTUP_SLEEP_SECONDS:-2}"; then
    echo "teacher endpoint failed readiness probe: ${line%/}" >&2
    cat "$teacher_stderr" >&2 || true
    cat "$teacher_stdout" >&2 || true
    return 1
  fi

  export NANOHORIZON_TEACHER_INFERENCE_URL="${line%/}/v1/chat/completions"
  export NANOHORIZON_TEACHER_API_KEY="${NANOHORIZON_TEACHER_API_KEY:-dummy-local-key}"
  return 0
}
