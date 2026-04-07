#!/usr/bin/env bash
set -euo pipefail

resolve_runpod_completion_webhook() {
  if [[ -n "${NANOHORIZON_RUNPOD_COMPLETION_WEBHOOK_URL:-}" ]]; then
    printf '%s\n' "$NANOHORIZON_RUNPOD_COMPLETION_WEBHOOK_URL"
    return 0
  fi

  python3 - <<'PY'
import json
import urllib.request

request = urllib.request.Request(
    "https://webhook.site/token",
    data=b"",
    method="POST",
    headers={"Accept": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
token = str(payload.get("uuid") or "").strip()
if not token:
    raise SystemExit("failed to create webhook.site token")
print(f"https://webhook.site/{token}")
PY
}
