from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import shutil
import signal
import sys
import time
from contextlib import suppress
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_PROD_BACKEND_URL = "https://api.usesynth.ai"
DEFAULT_DEV_BACKEND_URL = "https://api-dev.usesynth.ai"


def _load_synth_tunnel_dependencies() -> tuple[Any, Any, Any]:
    client_module = importlib.import_module("synth_ai.client")
    return client_module.SynthTunnel, client_module.NgrokTunnel, client_module


_TRYCLOUDFLARE_URL_RE = re.compile(r"https://[A-Za-z0-9.-]+\.trycloudflare\.com")


class _QuickTunnelHandle:
    def __init__(self, url: str, proc: asyncio.subprocess.Process) -> None:
        self.url = url
        self.worker_token = None
        self._proc = proc

    async def close_async(self) -> None:
        if self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()


async def _open_cloudflare_quick_tunnel(*, local_port: int, wait_seconds: float) -> _QuickTunnelHandle:
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise RuntimeError("cloudflared not found in PATH")

    proc = await asyncio.create_subprocess_exec(
        cloudflared,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{local_port}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    deadline = time.monotonic() + wait_seconds
    captured_lines: list[str] = []
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
        except TimeoutError:
            break
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        captured_lines.append(line)
        match = _TRYCLOUDFLARE_URL_RE.search(line)
        if match:
            return _QuickTunnelHandle(match.group(0), proc)

    if proc.returncode is None:
        proc.terminate()
        await proc.wait()
    tail = "\n".join(captured_lines[-20:])
    raise RuntimeError(f"timed out waiting for quick tunnel URL\n{tail}".strip())


def _wait_for_local_health(*, health_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            request = Request(health_url, method="GET")
            with urlopen(request, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = RuntimeError(f"health returned HTTP {response.status}")
        except URLError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"local health check failed for {health_url}: {last_error!r}")


def _resolve_backend_url(raw_value: str) -> str:
    raw = (raw_value or "").strip()
    if not raw:
        raw = (
            os.environ.get("SYNTH_BACKEND_URL")
            or os.environ.get("SYNTH_BACKEND_URL_OVERRIDE")
            or os.environ.get("PROD_SYNTH_BACKEND_URL")
            or ""
        ).strip()
    lowered = raw.lower()
    if lowered in {"", "prod", "production", "main"}:
        return DEFAULT_PROD_BACKEND_URL
    if lowered in {"dev", "development", "staging", "railway"}:
        return DEFAULT_DEV_BACKEND_URL
    if lowered in {"local", "localhost"}:
        return "http://localhost:8000"
    return raw


def _build_payload(*, tunnel: Any, backend: str, local_port: int) -> dict[str, Any]:
    return {
        "backend": backend,
        "container_url": tunnel.url,
        "container_worker_token": getattr(tunnel, "worker_token", None),
        "local_port": local_port,
    }


def _print_exports(payload: dict[str, Any]) -> None:
    print(f'export NANOHORIZON_CRAFTAX_CONTAINER_URL="{payload["container_url"]}"', flush=True)
    print(f'export NANOHORIZON_CRAFTAX_CONTAINER_URL="{payload["container_url"]}"', flush=True)
    worker_token = payload.get("container_worker_token") or ""
    print(f'export NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="{worker_token}"', flush=True)
    print(f'export NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="{worker_token}"', flush=True)


def _write_env_file(*, path: str, payload: dict[str, Any]) -> None:
    lines = [
        f'NANOHORIZON_CRAFTAX_CONTAINER_URL="{payload["container_url"]}"',
        f'NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="{payload.get("container_worker_token") or ""}"',
        f'NANOHORIZON_CRAFTAX_CONTAINER_URL="{payload["container_url"]}"',
        f'NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN="{payload.get("container_worker_token") or ""}"',
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_json_file(*, path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


async def _hold_open() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    await stop_event.wait()


async def _open_tunnel(args: argparse.Namespace) -> dict[str, Any]:
    health_url = f"{args.local_base_url.rstrip('/')}/health"
    _wait_for_local_health(health_url=health_url, timeout_seconds=float(args.health_timeout))
    backend_url = _resolve_backend_url(args.backend_url)
    os.environ["SYNTH_BACKEND_URL"] = backend_url

    if args.backend == "synthtunnel":
        SynthTunnel, _, _ = _load_synth_tunnel_dependencies()
        tunnel = await SynthTunnel(api_key=args.api_key or None, base_url=backend_url).open_async(
            local_port=int(args.local_port),
            verify_dns=not bool(args.skip_verify_dns),
            progress=True,
            requested_ttl_seconds=int(args.requested_ttl_seconds) if args.requested_ttl_seconds else None,
        )
    elif args.backend == "ngrok_managed":
        _, NgrokTunnel, _ = _load_synth_tunnel_dependencies()
        if not (args.managed_ngrok_url or "").strip():
            raise ValueError("--managed-ngrok-url is required for backend=ngrok_managed")
        tunnel = await NgrokTunnel(api_key=args.api_key or None, base_url=backend_url).open_async(
            local_port=int(args.local_port),
            managed_ngrok_url=args.managed_ngrok_url,
            verify_dns=not bool(args.skip_verify_dns),
            progress=True,
        )
    else:
        tunnel = await _open_cloudflare_quick_tunnel(
            local_port=int(args.local_port),
            wait_seconds=float(args.quick_tunnel_wait_seconds),
        )

    payload = _build_payload(tunnel=tunnel, backend=args.backend, local_port=int(args.local_port))
    payload["backend_url"] = backend_url

    if args.env_file:
        _write_env_file(path=args.env_file, payload=payload)
    if args.json_file:
        _write_json_file(path=args.json_file, payload=payload)

    if args.print_exports:
        _print_exports(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    try:
        if args.hold:
            print("Tunnel open. Press Ctrl-C to close.", file=sys.stderr)
            await _hold_open()
    finally:
        await tunnel.close_async()

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a public tunnel for the local NanoHorizon Craftax runtime")
    parser.add_argument("--backend", choices=["synthtunnel", "ngrok_managed", "cloudflare_quick"], default="synthtunnel")
    parser.add_argument("--local-port", type=int, default=8903)
    parser.add_argument("--local-base-url", default="http://127.0.0.1:8903")
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--managed-ngrok-url", default="")
    parser.add_argument("--quick-tunnel-wait-seconds", type=float, default=10.0)
    parser.add_argument("--requested-ttl-seconds", type=int, default=0)
    parser.add_argument("--skip-verify-dns", action="store_true")
    parser.add_argument("--print-exports", action="store_true")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--json-file", default="")
    parser.add_argument("--hold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(_open_tunnel(args))


if __name__ == "__main__":
    main()
