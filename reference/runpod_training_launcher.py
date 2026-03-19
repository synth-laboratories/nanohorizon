#!/usr/bin/env python3
"""Launch low-cost RunPod training jobs for SMR-friendly remote execution.

This utility intentionally stays small and stdlib-only:
- build a cheap-GPU RunPod Pod payload
- inject a bootstrap script via dockerStartCmd
- clone a git repo on the pod
- run setup + training commands
- write a machine-readable job manifest into /workspace
- optionally stop the pod after the training command finishes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_VOLUME_GB = 100
DEFAULT_VOLUME_MOUNT = "/workspace"
DEFAULT_NAME_PREFIX = "smr-train"
DEFAULT_WAIT_TIMEOUT_SECONDS = 600.0
DEFAULT_OUTPUT_SERVE_PORT = 8000
DEFAULT_STATUS_POLL_SECONDS = 10.0
DEFAULT_PRESET = "generic"
DEFAULT_HTTP_MAX_ATTEMPTS = 6
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_PROXY_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
}


def _build_ssl_context() -> ssl.SSLContext:
    """Prefer certifi when available so hosted worker shells have a CA bundle."""
    candidates: list[str] = []
    env_cafile = str(os.getenv("SSL_CERT_FILE") or "").strip()
    if env_cafile:
        candidates.append(env_cafile)
    try:
        import certifi  # type: ignore

        certifi_where = str(certifi.where() or "").strip()
        if certifi_where:
            candidates.append(certifi_where)
    except Exception:
        pass
    verify_paths = ssl.get_default_verify_paths()
    for candidate in (
        str(verify_paths.cafile or "").strip(),
        str(verify_paths.openssl_cafile or "").strip(),
        "/etc/ssl/cert.pem",
    ):
        if candidate:
            candidates.append(candidate)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.exists(candidate):
            continue
        seen.add(candidate)
        try:
            return ssl.create_default_context(cafile=candidate)
        except OSError:
            continue

    removed_env: dict[str, str] = {}
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = os.environ.pop(key, None)
        if value is not None:
            removed_env[key] = value
    try:
        return ssl.create_default_context()
    finally:
        os.environ.update(removed_env)

# Cheap-ish defaults guided by current RunPod docs:
# - A4000/A4500/RTX 4000 are the most cost-effective 16 GB options
# - L4/A5000/3090 are cost-effective 24 GB options
# - A6000/A40 are cost-effective 48 GB options
GPU_PROFILES: dict[str, list[str]] = {
    "small16": [
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA RTX 4000 Ada Generation",
    ],
    "mid24": [
        "NVIDIA L4",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090",
    ],
    "large48": [
        "NVIDIA RTX A6000",
        "NVIDIA A40",
        "NVIDIA RTX 6000 Ada Generation",
    ],
}

RUNPOD_PRESETS: dict[str, dict[str, Any]] = {
    "generic": {
        "description": "Default generic PyTorch image path.",
        "image_name": DEFAULT_IMAGE,
    },
    "parameter-golf-official": {
        "description": "OpenAI Parameter Golf official RunPod template.",
        "template_id": "y5cejece4j",
    },
}


@dataclass
class LaunchPlan:
    payload: dict[str, Any]
    bootstrap_script: str


class RunpodClient:
    def __init__(self, *, api_key: str, api_base: str = DEFAULT_API_BASE) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY is required")
        self.api_base = api_base.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.api_base}{path}"
        if query:
            encoded = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None},
                doseq=True,
            )
            if encoded:
                url = f"{url}?{encoded}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            raw = _urlopen_read_text(
                request,
                timeout=120.0,
                retry_http_codes={429, 500, 502, 503, 504},
            )
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only against real API
            detail = (exc.reason or "").strip()
            raise RuntimeError(f"RunPod API {method} {url} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - exercised only against real API
            raise RuntimeError(f"RunPod API {method} {url} failed: {exc}") from exc
        if not raw:
            return {}
        decoded = json.loads(raw)
        if isinstance(decoded, (dict, list)):
            return decoded
        raise RuntimeError(f"expected JSON object or list from RunPod API, got: {type(decoded).__name__}")

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", payload=payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def list_pods(self, *, name: str | None = None, desired_status: str | None = None) -> Any:
        return self._request(
            "GET",
            "/pods",
            query={
                "name": name,
                "desiredStatus": desired_status,
            },
        )

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _shell_join(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _quote(value: str) -> str:
    return shlex.quote(value)


def _proxy_artifact_url(*, pod_id: str, port: int, relative_path: str) -> str:
    clean_path = relative_path.lstrip("/")
    return f"https://{pod_id}-{port}.proxy.runpod.net/{clean_path}"


def _env_pairs(values: list[str]) -> dict[str, str]:
    env_map: dict[str, str] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise RuntimeError(f"expected KEY=VALUE env pair, got: {text!r}")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeError(f"missing env key in pair: {text!r}")
        env_map[key] = value
    return env_map


def _gpu_ids(args: argparse.Namespace) -> list[str]:
    if args.gpu_type_id:
        return [str(item).strip() for item in args.gpu_type_id if str(item).strip()]
    profile = str(args.gpu_profile or "mid24").strip()
    if profile not in GPU_PROFILES:
        raise RuntimeError(
            f"unknown gpu profile {profile!r}; expected one of {', '.join(sorted(GPU_PROFILES))}"
        )
    return list(GPU_PROFILES[profile])


def _selected_preset(args: argparse.Namespace) -> dict[str, Any]:
    name = str(getattr(args, "preset", DEFAULT_PRESET) or DEFAULT_PRESET).strip()
    if name not in RUNPOD_PRESETS:
        raise RuntimeError(
            f"unknown preset {name!r}; expected one of {', '.join(sorted(RUNPOD_PRESETS))}"
        )
    return RUNPOD_PRESETS[name]


def _selected_template_id(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "template_id", "") or "").strip()
    if explicit:
        return explicit
    preset = _selected_preset(args)
    return str(preset.get("template_id") or "").strip()


def _selected_image_name(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "image_name", "") or "").strip()
    if explicit:
        return explicit
    preset = _selected_preset(args)
    return str(preset.get("image_name") or DEFAULT_IMAGE).strip()


def build_bootstrap_script(args: argparse.Namespace) -> str:
    repo_dir = args.repo_dir.strip() or "repo"
    output_dir = args.output_dir.strip() or "/workspace/smr-runpod"
    setup_cmd = str(args.setup_cmd or "").strip()
    train_cmd = str(args.train_cmd or "").strip()
    if not train_cmd:
        raise RuntimeError("--train-cmd is required")
    git_repo = str(args.git_repo or "").strip()
    git_ref = str(args.git_ref or "").strip()
    uv_install = "1" if args.install_uv else "0"
    auto_stop = "1" if args.auto_stop else "0"
    keepalive = "1" if args.keepalive_after else "0"
    serve_port = int(args.serve_output_port or 0)
    completion_webhook_url = str(args.completion_webhook_url or "").strip()
    manifest_path = f"{output_dir.rstrip('/')}/job_status.json"
    log_path = f"{output_dir.rstrip('/')}/training.log"
    http_log_path = f"{output_dir.rstrip('/')}/http_server.log"
    setup_section = setup_cmd if setup_cmd else "echo 'no setup command configured'"
    lines = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        f"export SMR_RUNPOD_OUTPUT_DIR={_quote(output_dir)}",
        f"export SMR_RUNPOD_LOG_PATH={_quote(log_path)}",
        f"export SMR_RUNPOD_MANIFEST_PATH={_quote(manifest_path)}",
        f"export SMR_RUNPOD_HTTP_LOG_PATH={_quote(http_log_path)}",
        f"mkdir -p {_quote(output_dir)}",
        "FINALIZED=0",
        "write_manifest() {",
        "python3 - <<'PY'",
        "import json, os",
        "payload = {",
        f"  'git_repo': {git_repo!r},",
        f"  'git_ref': {git_ref!r},",
        f"  'repo_dir': {repo_dir!r},",
        f"  'setup_cmd': {setup_cmd!r},",
        f"  'train_cmd': {train_cmd!r},",
        "  'state': os.environ.get('SMR_RUNPOD_STATE', 'unknown'),",
        "  'completed': os.environ.get('SMR_RUNPOD_COMPLETED', '0') == '1',",
        "  'status': int(os.environ.get('SMR_RUNPOD_EXIT_CODE', '-1')),",
        "  'started_at': os.environ.get('START_TS'),",
        "  'ended_at': os.environ.get('END_TS'),",
        "  'log_path': os.environ.get('SMR_RUNPOD_LOG_PATH'),",
        "  'http_log_path': os.environ.get('SMR_RUNPOD_HTTP_LOG_PATH'),",
        "  'runpod_pod_id': os.environ.get('RUNPOD_POD_ID'),",
        "}",
        "with open(os.environ['SMR_RUNPOD_MANIFEST_PATH'], 'w', encoding='utf-8') as handle:",
        "    json.dump(payload, handle, indent=2, sort_keys=True)",
        "    handle.write('\\n')",
        "PY",
        "}",
        "post_completion() {",
        f"  if [ -z {_quote(completion_webhook_url)} ]; then",
        "    return 0",
        "  fi",
        f"  CALLBACK_URL={_quote(completion_webhook_url)}",
        "  CURL_ARGS=(-fsS --retry 5 --retry-delay 2 --retry-all-errors --connect-timeout 20 --max-time 300 -X POST \"$CALLBACK_URL\"",
        "    -F \"state=${SMR_RUNPOD_STATE:-unknown}\"",
        "    -F \"completed=${SMR_RUNPOD_COMPLETED:-0}\"",
        "    -F \"status=${SMR_RUNPOD_EXIT_CODE:-1}\"",
        "    -F \"pod_id=${RUNPOD_POD_ID:-}\"",
        "    -F \"job_status=@${SMR_RUNPOD_MANIFEST_PATH};type=application/json\")",
        "  if [ -f \"$SMR_RUNPOD_LOG_PATH\" ]; then",
        "    CURL_ARGS+=(-F \"training_log=@${SMR_RUNPOD_LOG_PATH};type=text/plain\")",
        "  fi",
        "  if [ -f \"$SMR_RUNPOD_HTTP_LOG_PATH\" ]; then",
        "    CURL_ARGS+=(-F \"http_log=@${SMR_RUNPOD_HTTP_LOG_PATH};type=text/plain\")",
        "  fi",
        "  if [ -f final_model.pt ]; then",
        "    CURL_ARGS+=(-F \"final_model=@final_model.pt;type=application/octet-stream\")",
        "  fi",
        "  if [ -f final_model.int8.ptz ]; then",
        "    CURL_ARGS+=(-F \"final_model_int8=@final_model.int8.ptz;type=application/octet-stream\")",
        "  fi",
        "  if [ -f train_gpt.py ]; then",
        "    CURL_ARGS+=(-F \"trainer_snapshot=@train_gpt.py;type=text/x-python\")",
        "  fi",
        "  curl \"${CURL_ARGS[@]}\" >/dev/null || true",
        "}",
        "cleanup_on_exit() {",
        "  EXIT_CODE=$?",
        "  if [ \"$FINALIZED\" != \"1\" ]; then",
        "    PREV_STATE=${SMR_RUNPOD_STATE:-unknown}",
        "    PREV_EXIT_CODE=${SMR_RUNPOD_EXIT_CODE:-$EXIT_CODE}",
        "    if [ \"$PREV_EXIT_CODE\" = \"0\" ] || [ \"$PREV_STATE\" = \"succeeded\" ]; then",
        "      export SMR_RUNPOD_STATE=succeeded",
        "    else",
        "      export SMR_RUNPOD_STATE=failed",
        "    fi",
        "    export SMR_RUNPOD_COMPLETED=1",
        "    export SMR_RUNPOD_EXIT_CODE=$PREV_EXIT_CODE",
        "    export END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "    write_manifest",
        "    post_completion",
        "  fi",
        "  exit $EXIT_CODE",
        "}",
        "trap cleanup_on_exit EXIT",
        "export SMR_RUNPOD_STATE=bootstrapping",
        "export SMR_RUNPOD_COMPLETED=0",
        "export SMR_RUNPOD_EXIT_CODE=-1",
        "write_manifest",
        f"if [ {serve_port} -gt 0 ]; then",
        f"  cd {_quote(output_dir)}",
        f"  python3 -m http.server {serve_port} --bind 0.0.0.0 > {_quote(http_log_path)} 2>&1 &",
        "  cd - >/dev/null",
        "fi",
        "NEED_APT=0",
        "if ! command -v git >/dev/null 2>&1; then NEED_APT=1; fi",
        "if ! command -v curl >/dev/null 2>&1; then NEED_APT=1; fi",
        "if [ ! -f /etc/ssl/certs/ca-certificates.crt ]; then NEED_APT=1; fi",
        "if [ \"$NEED_APT\" = \"1\" ] && command -v apt-get >/dev/null 2>&1; then",
        "  apt-get update",
        "  apt-get install -y git ca-certificates curl",
        "  update-ca-certificates || true",
        "fi",
        "if ! command -v curl >/dev/null 2>&1; then echo 'curl missing from image' >&2; exit 2; fi",
        "if ! command -v python3 >/dev/null 2>&1; then echo 'python3 missing from image' >&2; exit 2; fi",
    ]
    if git_repo:
        lines.extend(
            [
                "if ! command -v git >/dev/null 2>&1; then echo 'git missing from image' >&2; exit 2; fi",
                f"if [ ! -d {_quote(repo_dir)} ]; then git clone {_quote(git_repo)} {_quote(repo_dir)}; fi",
                f"cd {_quote(repo_dir)}",
            ]
        )
    else:
        lines.extend(
            [
                f"mkdir -p {_quote(repo_dir)}",
                f"cd {_quote(repo_dir)}",
            ]
        )
    if git_repo and git_ref:
        lines.extend(
            [
                f"git fetch --all --tags --prune || true",
                f"git checkout {_quote(git_ref)}",
            ]
        )
    lines.extend(
        [
            "python3 -m pip install --upgrade pip",
            f"if [ {uv_install} = 1 ]; then python3 -m pip install --upgrade uv; fi",
            "START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
            "export START_TS",
            "export SMR_RUNPOD_STATE=running",
            "write_manifest",
            "{",
            f"  {setup_section}",
            f"  {train_cmd}",
            f"}} 2>&1 | tee {_quote(log_path)}",
            "STATUS=${PIPESTATUS[0]}",
            "export STATUS",
            "END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
            "export END_TS",
            "export SMR_RUNPOD_EXIT_CODE=${STATUS}",
            "if [ ${STATUS} -eq 0 ]; then export SMR_RUNPOD_STATE=succeeded; else export SMR_RUNPOD_STATE=failed; fi",
            "export SMR_RUNPOD_COMPLETED=1",
            "write_manifest",
            "post_completion",
            "FINALIZED=1",
            "trap - EXIT",
            "if [ ${STATUS} -eq 0 ] && [ "
            + auto_stop
            + " = 1 ] && command -v runpodctl >/dev/null 2>&1 && [ -n \"${RUNPOD_POD_ID:-}\" ]; then",
            "  (sleep 20; runpodctl stop pod \"$RUNPOD_POD_ID\" || true) >/dev/null 2>&1 &",
            "fi",
            "if [ "
            + keepalive
            + " = 1 ]; then",
            "  echo 'training finished; keepalive requested' | tee -a \"$SMR_RUNPOD_LOG_PATH\"",
            "  tail -f /dev/null",
            "fi",
            "exit ${STATUS}",
        ]
    )
    return _shell_join(lines)


def build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    gpu_ids = _gpu_ids(args)
    env = _env_pairs(args.env)
    bootstrap_script = build_bootstrap_script(args)
    template_id = _selected_template_id(args)
    image_name = _selected_image_name(args)
    ports = list(args.port or ["22/tcp"])
    serve_port = int(args.serve_output_port or 0)
    if serve_port > 0:
        http_port = f"{serve_port}/http"
        if http_port not in ports:
            ports.append(http_port)
    payload = {
        "name": args.name.strip() or f"{DEFAULT_NAME_PREFIX}-{_now_stamp()}",
        "gpuTypeIds": gpu_ids,
        "gpuTypePriority": "custom",
        "gpuCount": args.gpu_count,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": args.volume_mount_path,
        "interruptible": args.interruptible,
        "supportPublicIp": args.support_public_ip,
        "ports": ports,
        "minVCPUPerGPU": args.min_vcpu_per_gpu,
        "minRAMPerGPU": args.min_ram_per_gpu,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [bootstrap_script],
        "env": env,
    }
    if template_id:
        payload["templateId"] = template_id
    else:
        payload["imageName"] = image_name
    if args.network_volume_id:
        payload["networkVolumeId"] = args.network_volume_id.strip()
    if args.data_center_id:
        payload["dataCenterIds"] = [item.strip() for item in args.data_center_id if item.strip()]
        payload["dataCenterPriority"] = "custom"
    if args.country_code:
        payload["countryCodes"] = [item.strip() for item in args.country_code if item.strip()]
    return LaunchPlan(payload=payload, bootstrap_script=bootstrap_script)


def _extract_pod_id(payload: dict[str, Any]) -> str:
    for key in ("id", "podId"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    raise RuntimeError(f"unable to determine pod id from response: {payload}")


def _pod_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "desiredStatus": payload.get("desiredStatus"),
        "imageName": payload.get("imageName") or payload.get("image"),
        "templateId": payload.get("templateId"),
        "costPerHr": payload.get("costPerHr"),
        "gpuCount": payload.get("gpuCount"),
        "machineId": payload.get("machineId"),
        "publicIp": payload.get("publicIp"),
        "ports": payload.get("ports"),
        "interruptible": payload.get("interruptible"),
    }


def _redact_raw_pod(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    env_payload = sanitized.get("env")
    if isinstance(env_payload, dict):
        sanitized["env"] = {key: "<redacted>" for key in env_payload}
    elif isinstance(env_payload, list):
        sanitized["env"] = ["<redacted>" for _ in env_payload]
    return sanitized


def _wait_for_desired_status(
    client: RunpodClient,
    *,
    pod_id: str,
    desired_status: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        last_payload = client.get_pod(pod_id)
        current = str(last_payload.get("desiredStatus") or "").strip().upper()
        if current == desired_status.upper():
            return last_payload
        time.sleep(5.0)
    raise TimeoutError(
        f"timed out waiting for pod {pod_id} to reach {desired_status}; last={_pod_summary(last_payload)}"
    )


def _sleep_for_retry(attempt: int) -> None:
    delay = DEFAULT_HTTP_RETRY_BACKOFF_SECONDS * max(1.0, float(attempt))
    time.sleep(delay)


def _urlopen_read_bytes(
    request: urllib.request.Request,
    *,
    timeout: float,
    retry_http_codes: set[int] | None = None,
) -> bytes:
    retriable_http_codes = set(retry_http_codes or set())
    ssl_context = _build_ssl_context()
    for attempt in range(1, DEFAULT_HTTP_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in retriable_http_codes and attempt < DEFAULT_HTTP_MAX_ATTEMPTS:
                _sleep_for_retry(attempt)
                continue
            raise urllib.error.HTTPError(
                exc.url,
                exc.code,
                detail or exc.reason,
                exc.headers,
                None,
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < DEFAULT_HTTP_MAX_ATTEMPTS:
                _sleep_for_retry(attempt)
                continue
            raise
    raise urllib.error.URLError("request exhausted without response")


def _urlopen_read_text(
    request: urllib.request.Request,
    *,
    timeout: float,
    retry_http_codes: set[int] | None = None,
) -> str:
    return _urlopen_read_bytes(
        request,
        timeout=timeout,
        retry_http_codes=retry_http_codes,
    ).decode("utf-8", errors="replace")


def _fetch_json(url: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers=dict(DEFAULT_PROXY_HEADERS))
    try:
        raw = _urlopen_read_text(
            request,
            timeout=30.0,
            retry_http_codes={502, 503, 504},
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 502, 503}:
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"artifact fetch failed for {url}: {exc.code} {detail}") from exc
    except RuntimeError:
        return None
    if not raw.strip():
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def _fetch_text(url: str) -> str | None:
    headers = dict(DEFAULT_PROXY_HEADERS)
    headers["Accept"] = "text/plain"
    request = urllib.request.Request(url, headers=headers)
    try:
        return _urlopen_read_text(
            request,
            timeout=30.0,
            retry_http_codes={502, 503, 504},
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 502, 503}:
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"artifact fetch failed for {url}: {exc.code} {detail}") from exc
    except RuntimeError:
        return None


def _fetch_webhook_requests(token_id: str, *, per_page: int = 20) -> list[dict[str, Any]]:
    url = f"https://webhook.site/token/{token_id}/requests?sorting=newest&per_page={per_page}"
    request = urllib.request.Request(url, headers=dict(DEFAULT_PROXY_HEADERS))
    try:
        raw = _urlopen_read_text(
            request,
            timeout=30.0,
            retry_http_codes={429, 500, 502, 503, 504},
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 429}:
            return []
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"webhook fetch failed for {url}: {exc.code} {detail}") from exc
    except RuntimeError:
        return []
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    requests = payload.get("data") or []
    if not isinstance(requests, list):
        raise RuntimeError(f"expected list payload from {url}")
    return [item for item in requests if isinstance(item, dict)]


def _extract_webhook_token(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc != "webhook.site" or not parts:
        raise RuntimeError(f"expected webhook.site URL, got: {url}")
    return parts[0]


def _webhook_request_pod_id(payload: dict[str, Any]) -> str:
    request_payload = payload.get("request") or {}
    if not isinstance(request_payload, dict):
        return ""
    return str(request_payload.get("pod_id") or "").strip()


def _select_webhook_request(requests: list[dict[str, Any]], *, pod_id: str) -> dict[str, Any] | None:
    for payload in requests:
        files = payload.get("files") or {}
        if not isinstance(files, dict) or "job_status" not in files:
            continue
        current_pod_id = _webhook_request_pod_id(payload)
        if current_pod_id and current_pod_id == pod_id:
            return payload
    return None


def _webhook_download_url(*, token_id: str, request_id: str, file_id: str) -> str:
    return f"https://webhook.site/token/{token_id}/request/{request_id}/download/{file_id}"


def _download_webhook_file(*, token_id: str, request_id: str, file_id: str) -> bytes:
    url = _webhook_download_url(token_id=token_id, request_id=request_id, file_id=file_id)
    request = urllib.request.Request(url, headers=dict(DEFAULT_PROXY_HEADERS))
    return _urlopen_read_bytes(
        request,
        timeout=120.0,
        retry_http_codes={429, 500, 502, 503, 504},
    )


def _decode_json_bytes(payload: bytes) -> dict[str, Any]:
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("expected JSON object from webhook artifact")
    return decoded


def _summarize_webhook_request(payload: dict[str, Any]) -> dict[str, Any]:
    request_payload = payload.get("request") or {}
    files = payload.get("files") or {}
    result: dict[str, Any] = {
        "uuid": payload.get("uuid"),
        "created_at": payload.get("created_at"),
        "pod_id": _webhook_request_pod_id(payload),
        "files": {},
    }
    if isinstance(request_payload, dict):
        result["state"] = request_payload.get("state")
        result["status"] = request_payload.get("status")
        result["completed"] = request_payload.get("completed")
    if isinstance(files, dict):
        result["files"] = {
            name: {
                "id": meta.get("id"),
                "filename": meta.get("filename"),
                "size": meta.get("size"),
                "content_type": meta.get("content_type"),
            }
            for name, meta in files.items()
            if isinstance(meta, dict)
        }
    return result


_KV_TOKEN_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):([^\s]+)")
_STEP_PATTERN = re.compile(r"step:(\d+)/(\d+)")
_RUN_LOG_PATTERN = re.compile(r"^logs/([^.]+)\.txt$")
_MEMORY_PATTERN = re.compile(r"peak memory allocated:\s*(\d+)\s*MiB reserved:\s*(\d+)\s*MiB")


def _parse_number(text: str) -> int | float | str:
    value = str(text).strip()
    if not value:
        return value
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_metric_tokens(line: str) -> dict[str, int | float | str]:
    return {key: _parse_number(value) for key, value in _KV_TOKEN_PATTERN.findall(line)}


def _parse_step_line(line: str) -> dict[str, Any] | None:
    match = _STEP_PATTERN.search(line)
    if not match:
        return None
    metrics: dict[str, Any] = _parse_metric_tokens(line)
    metrics["step"] = int(match.group(1))
    metrics["steps_total"] = int(match.group(2))
    return metrics


def _build_training_result_summary(
    *,
    log_text: str,
    manifest: dict[str, Any] | None,
    files: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_train: dict[str, Any] | None = None
    latest_val: dict[str, Any] | None = None
    final_roundtrip: dict[str, Any] | None = None
    final_roundtrip_exact: dict[str, Any] | None = None
    run_id = ""
    stopping_early = ""
    peak_memory: dict[str, int] | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        run_match = _RUN_LOG_PATTERN.match(line)
        if run_match:
            run_id = run_match.group(1)
            continue
        if line.startswith("step:"):
            metrics = _parse_step_line(line)
            if not metrics:
                continue
            if "train_loss" in metrics:
                latest_train = dict(metrics)
            if "val_loss" in metrics or "val_bpb" in metrics:
                latest_val = dict(metrics)
            continue
        if line.startswith("final_int8_zlib_roundtrip_exact"):
            final_roundtrip_exact = _parse_metric_tokens(line)
            continue
        if line.startswith("final_int8_zlib_roundtrip"):
            final_roundtrip = _parse_metric_tokens(line)
            continue
        if line.startswith("stopping_early:"):
            stopping_early = line
            continue
        memory_match = _MEMORY_PATTERN.search(line)
        if memory_match:
            peak_memory = {
                "allocated_mib": int(memory_match.group(1)),
                "reserved_mib": int(memory_match.group(2)),
            }
    files_map = files if isinstance(files, dict) else {}
    trainer_meta = files_map.get("trainer_snapshot") or {}
    final_model_meta = files_map.get("final_model_int8") or files_map.get("final_model") or {}
    bytes_code = trainer_meta.get("size") if isinstance(trainer_meta, dict) else None
    compressed_model_bytes = final_model_meta.get("size") if isinstance(final_model_meta, dict) else None
    bytes_total = None
    if isinstance(bytes_code, int) and isinstance(compressed_model_bytes, int):
        bytes_total = int(bytes_code) + int(compressed_model_bytes)
    final_metrics = final_roundtrip_exact or final_roundtrip or latest_val or {}
    summary: dict[str, Any] = {
        "run_id": run_id or None,
        "date": (manifest or {}).get("ended_at"),
        "state": (manifest or {}).get("state"),
        "status": (manifest or {}).get("status"),
        "git_repo": (manifest or {}).get("git_repo"),
        "git_ref": (manifest or {}).get("git_ref"),
        "val_loss": final_metrics.get("val_loss"),
        "val_bpb": final_metrics.get("val_bpb"),
        "train_loss_last": (latest_train or {}).get("train_loss"),
        "train_step_last": (latest_train or {}).get("step"),
        "val_step_last": (latest_val or {}).get("step"),
        "bytes_code": bytes_code,
        "compressed_model_bytes": compressed_model_bytes,
        "bytes_total": bytes_total,
        "stopping_early": stopping_early or None,
        "peak_memory": peak_memory,
        "source": (
            "final_int8_zlib_roundtrip_exact"
            if final_roundtrip_exact
            else "final_int8_zlib_roundtrip"
            if final_roundtrip
            else "latest_step_metrics"
            if latest_val
            else "none"
        ),
    }
    submission_like = {
        "name": run_id or (manifest or {}).get("runpod_pod_id") or "runpod-training",
        "date": summary["date"],
        "val_loss": summary["val_loss"],
        "val_bpb": summary["val_bpb"],
        "bytes_total": summary["bytes_total"],
        "bytes_code": summary["bytes_code"],
    }
    summary["submission_like"] = submission_like
    return summary


def _wait_for_remote_completion(
    client: RunpodClient,
    *,
    pod_id: str,
    serve_port: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    manifest_url = _proxy_artifact_url(pod_id=pod_id, port=serve_port, relative_path="job_status.json")
    log_url = _proxy_artifact_url(pod_id=pod_id, port=serve_port, relative_path="training.log")
    deadline = time.time() + timeout_seconds
    last_manifest: dict[str, Any] | None = None
    while time.time() < deadline:
        manifest = _fetch_json(manifest_url)
        if manifest:
            last_manifest = manifest
            if manifest.get("completed") is True:
                log_text = _fetch_text(log_url) or ""
                result_summary = _build_training_result_summary(
                    log_text=log_text,
                    manifest=manifest,
                    files=None,
                )
                return {
                    "manifest": manifest,
                    "manifest_url": manifest_url,
                    "log_url": log_url,
                    "result_summary": result_summary,
                    "log_tail": "\n".join(log_text.splitlines()[-40:]),
                }
        pod_payload = client.get_pod(pod_id)
        desired_status = str(pod_payload.get("desiredStatus") or "").strip().upper()
        if desired_status and desired_status != "RUNNING":
            raise RuntimeError(
                "pod exited before remote completion was observable: "
                f"pod_id={pod_id} desiredStatus={desired_status} "
                f"lastStatusChange={pod_payload.get('lastStatusChange')!r} "
                f"last_manifest={last_manifest}"
            )
        time.sleep(max(1.0, poll_seconds))
    raise TimeoutError(f"timed out waiting for remote completion on pod {pod_id}; last_manifest={last_manifest}")


def _wait_for_webhook_completion(
    client: RunpodClient,
    *,
    webhook_url: str,
    pod_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    token_id = _extract_webhook_token(webhook_url)
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        requests = _fetch_webhook_requests(token_id)
        payload = _select_webhook_request(requests, pod_id=pod_id)
        if payload:
            last_payload = _summarize_webhook_request(payload)
            files = payload.get("files") or {}
            request_id = str(payload.get("uuid") or "").strip()
            if not request_id or not isinstance(files, dict):
                time.sleep(max(1.0, poll_seconds))
                continue
            downloaded_files: dict[str, Any] = {}
            manifest: dict[str, Any] | None = None
            log_text = ""
            for name, meta in files.items():
                if not isinstance(meta, dict):
                    continue
                file_id = str(meta.get("id") or "").strip()
                if not file_id:
                    continue
                downloaded_files[name] = {
                    "download_url": _webhook_download_url(
                        token_id=token_id,
                        request_id=request_id,
                        file_id=file_id,
                    ),
                    "filename": meta.get("filename"),
                    "size": meta.get("size"),
                    "content_type": meta.get("content_type"),
                }
                if name == "job_status":
                    manifest = _decode_json_bytes(
                        _download_webhook_file(token_id=token_id, request_id=request_id, file_id=file_id)
                    )
                elif name == "training_log":
                    log_text = _download_webhook_file(
                        token_id=token_id,
                        request_id=request_id,
                        file_id=file_id,
                    ).decode("utf-8", errors="replace")
            if manifest is not None:
                result_summary = _build_training_result_summary(
                    log_text=log_text,
                    manifest=manifest,
                    files=downloaded_files,
                )
                return {
                    "request": last_payload,
                    "manifest": manifest,
                    "result_summary": result_summary,
                    "log_tail": "\n".join(log_text.splitlines()[-40:]),
                    "files": downloaded_files,
                }
        else:
            last_payload = {
                "pod_id": pod_id,
                "requests_seen": len(requests),
                "latest_request_uuid": requests[0].get("uuid") if requests else None,
            }
        pod_payload = client.get_pod(pod_id)
        desired_status = str(pod_payload.get("desiredStatus") or "").strip().upper()
        if desired_status and desired_status != "RUNNING":
            raise RuntimeError(
                "pod exited before webhook completion was observed: "
                f"pod_id={pod_id} desiredStatus={desired_status} "
                f"lastStatusChange={pod_payload.get('lastStatusChange')!r} "
                f"last={last_payload}"
            )
        time.sleep(max(1.0, poll_seconds))
    raise TimeoutError(f"timed out waiting for webhook completion for token {token_id}; last={last_payload}")


def _cleanup_pod(client: RunpodClient, *, pod_id: str, action: str) -> dict[str, Any] | None:
    if action == "none":
        return None
    if action == "stop":
        return client.stop_pod(pod_id)
    if action == "terminate":
        return client.terminate_pod(pod_id)
    raise RuntimeError(f"unsupported cleanup action: {action}")


def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_launch_plan(args)
    print(
        json.dumps(
            {
                "payload": plan.payload,
                "bootstrap_script": plan.bootstrap_script,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    plan = build_launch_plan(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "payload": plan.payload,
                    "bootstrap_script": plan.bootstrap_script,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    client = RunpodClient(api_key=_resolve_api_key(args), api_base=args.api_base)
    pod_id = ""
    completion_result: dict[str, Any] | None = None
    cleanup_result: dict[str, Any] | None = None
    created = client.create_pod(plan.payload)
    pod_id = _extract_pod_id(created)
    final_payload = created
    try:
        if args.wait_until_running or args.wait_for_completion:
            final_payload = _wait_for_desired_status(
                client,
                pod_id=pod_id,
                desired_status="RUNNING",
                timeout_seconds=args.wait_timeout_seconds,
            )
        if args.wait_for_completion:
            if str(args.completion_webhook_url or "").strip():
                completion_result = _wait_for_webhook_completion(
                    client,
                    webhook_url=str(args.completion_webhook_url),
                    pod_id=pod_id,
                    timeout_seconds=args.wait_timeout_seconds,
                    poll_seconds=args.status_poll_seconds,
                )
            else:
                completion_result = _wait_for_remote_completion(
                    client,
                    pod_id=pod_id,
                    serve_port=int(args.serve_output_port),
                    timeout_seconds=args.wait_timeout_seconds,
                    poll_seconds=args.status_poll_seconds,
                )
            cleanup_result = _cleanup_pod(client, pod_id=pod_id, action=args.cleanup_action)
    except Exception:
        if pod_id and cleanup_result is None and args.cleanup_action != "none":
            try:
                cleanup_result = _cleanup_pod(client, pod_id=pod_id, action=args.cleanup_action)
            except Exception:
                cleanup_result = {"cleanup_error": True, "pod_id": pod_id}
        raise
    print(
        json.dumps(
            {
                "pod": _pod_summary(final_payload),
                "payload": plan.payload,
                "completion": completion_result,
                "cleanup": cleanup_result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = RunpodClient(api_key=_resolve_api_key(args), api_base=args.api_base)
    payload = client.get_pod(args.pod_id)
    print(json.dumps({"pod": _pod_summary(payload), "raw": _redact_raw_pod(payload)}, indent=2, sort_keys=True))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    client = RunpodClient(api_key=_resolve_api_key(args), api_base=args.api_base)
    payload = client.list_pods(name=args.name or None, desired_status=args.desired_status or None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    client = RunpodClient(api_key=_resolve_api_key(args), api_base=args.api_base)
    payload = client.stop_pod(args.pod_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    client = RunpodClient(api_key=_resolve_api_key(args), api_base=args.api_base)
    payload = client.terminate_pod(args.pod_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_api_key(args: argparse.Namespace) -> str:
    explicit = str(args.api_key or "").strip()
    if explicit:
        return explicit
    env_value = str(os.environ.get("RUNPOD_API_KEY") or "").strip()
    if env_value:
        return env_value
    raise RuntimeError("missing RunPod API key; set RUNPOD_API_KEY or pass --api-key")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch remote training on RunPod for SMR workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_launch_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--api-base", default=DEFAULT_API_BASE)
        target.add_argument("--api-key", default="")
        target.add_argument("--name", default="")
        target.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(RUNPOD_PRESETS))
        target.add_argument("--template-id", default="")
        target.add_argument("--image-name", default="")
        target.add_argument("--gpu-profile", default="mid24", choices=sorted(GPU_PROFILES))
        target.add_argument("--gpu-type-id", action="append", default=[])
        target.add_argument("--gpu-count", type=int, default=1)
        target.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
        target.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
        target.add_argument("--volume-mount-path", default=DEFAULT_VOLUME_MOUNT)
        target.add_argument("--network-volume-id", default="")
        target.add_argument("--interruptible", action=argparse.BooleanOptionalAction, default=True)
        target.add_argument("--support-public-ip", action=argparse.BooleanOptionalAction, default=True)
        target.add_argument("--min-vcpu-per-gpu", type=int, default=4)
        target.add_argument("--min-ram-per-gpu", type=int, default=16)
        target.add_argument("--data-center-id", action="append", default=[])
        target.add_argument("--country-code", action="append", default=[])
        target.add_argument("--port", action="append", default=[])
        target.add_argument("--env", action="append", default=[])
        target.add_argument("--git-repo", default="")
        target.add_argument("--git-ref", default="")
        target.add_argument("--repo-dir", default="repo")
        target.add_argument("--output-dir", default="/workspace/smr-runpod")
        target.add_argument("--setup-cmd", default="")
        target.add_argument("--train-cmd", default="")
        target.add_argument("--install-uv", action="store_true")
        target.add_argument("--auto-stop", action="store_true")
        target.add_argument("--keepalive-after", action="store_true")
        target.add_argument("--serve-output-port", type=int, default=DEFAULT_OUTPUT_SERVE_PORT)
        target.add_argument("--completion-webhook-url", default="")

    plan_parser = subparsers.add_parser("plan", help="Render the RunPod payload and bootstrap script.")
    add_common_launch_args(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)

    launch_parser = subparsers.add_parser("launch", help="Create a RunPod pod and start the training bootstrap.")
    add_common_launch_args(launch_parser)
    launch_parser.add_argument("--wait-until-running", action="store_true")
    launch_parser.add_argument("--wait-for-completion", action="store_true")
    launch_parser.add_argument("--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS)
    launch_parser.add_argument("--status-poll-seconds", type=float, default=DEFAULT_STATUS_POLL_SECONDS)
    launch_parser.add_argument("--cleanup-action", choices=("none", "stop", "terminate"), default="stop")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.set_defaults(func=cmd_launch)

    status_parser = subparsers.add_parser("status", help="Fetch one pod by id.")
    status_parser.add_argument("pod_id")
    status_parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    status_parser.add_argument("--api-key", default="")
    status_parser.set_defaults(func=cmd_status)

    list_parser = subparsers.add_parser("list", help="List pods.")
    list_parser.add_argument("--name", default="")
    list_parser.add_argument("--desired-status", default="")
    list_parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    list_parser.add_argument("--api-key", default="")
    list_parser.set_defaults(func=cmd_list)

    stop_parser = subparsers.add_parser("stop", help="Stop a pod.")
    stop_parser.add_argument("pod_id")
    stop_parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    stop_parser.add_argument("--api-key", default="")
    stop_parser.set_defaults(func=cmd_stop)

    terminate_parser = subparsers.add_parser("terminate", help="Terminate a pod.")
    terminate_parser.add_argument("pod_id")
    terminate_parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    terminate_parser.add_argument("--api-key", default="")
    terminate_parser.set_defaults(func=cmd_terminate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
