#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from typing import Any

from nanohorizon.craftax_core.rollout import run_rollout_request


def _load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    input_path = os.getenv("SYNTH_ROLLOUT_INPUT_PATH") or "/tmp/rollout.json"
    output_path = os.getenv("SYNTH_ROLLOUT_OUTPUT_PATH") or "/tmp/result.json"
    request = _load_json(input_path)
    media = request.get("media")
    if not isinstance(media, dict):
        media = {
            "capture_video": True,
            "fps": 6,
            "tile_size": 16,
            "write_mp4": False,
            "output_dir": "/tmp/craftax_rollout_media",
        }
        request["media"] = media
    result = run_rollout_request(request)
    media_payload = result.get("media")
    if isinstance(media_payload, dict):
        gif_path = Path(str(media_payload.get("gif_path") or ""))
        if gif_path.exists():
            data_url = "data:image/gif;base64," + base64.b64encode(gif_path.read_bytes()).decode("ascii")
            media_payload["gif_url"] = data_url
            media_payload["data_url"] = data_url
            media_payload["content_type"] = "image/gif"
            media_payload["file_size_bytes"] = gif_path.stat().st_size
    reward_info = result.get("reward_info") if isinstance(result.get("reward_info"), dict) else {}
    metrics = dict(reward_info.get("outcome_objectives") or {}) if isinstance(reward_info, dict) else {}
    score = float(metrics.get("unique_achievements") or metrics.get("reward") or reward_info.get("outcome_reward") or 0.0)
    _write_json(
        output_path,
        {
            "success": not bool(result.get("error")),
            "score": score,
            "metrics": {
                "outcome_reward": score,
                "details": reward_info.get("details") if isinstance(reward_info, dict) else {},
            },
            "rollout": result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
