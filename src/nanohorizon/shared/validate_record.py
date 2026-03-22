from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_FILES = (
    "metadata.json",
    "metrics.json",
    "system_info.json",
    "command.txt",
    "run_config.yaml",
)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: uv run python -m nanohorizon.shared.validate_record <record_dir>",
            file=sys.stderr,
        )
        return 2

    record_dir = Path(sys.argv[1]).expanduser().resolve()
    if not record_dir.is_dir():
        print(f"record directory not found: {record_dir}", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED_FILES if not (record_dir / name).exists()]
    if missing:
        print(
            json.dumps({"ok": False, "missing": missing, "record_dir": str(record_dir)}, indent=2)
        )
        return 1

    warnings: list[str] = []
    metrics_path = record_dir / "metrics.json"
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "record_dir": str(record_dir),
                    "error": f"failed to parse metrics.json: {type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        return 1
    if not isinstance(metrics_payload, dict):
        print(
            json.dumps(
                {
                    "ok": False,
                    "record_dir": str(record_dir),
                    "error": "metrics.json must decode to an object",
                },
                indent=2,
            )
        )
        return 1

    achievement_fields = (
        "submission_achievement_frequencies",
        "achievement_frequencies",
        "primary_achievement_frequencies",
        "final_achievement_frequencies",
        "finetuned_achievement_frequencies",
    )
    if not any(isinstance(metrics_payload.get(name), dict) for name in achievement_fields):
        warnings.append(
            "metrics.json is missing achievement frequencies; new submissions should include the 22-achievement frequency table."
        )

    print(json.dumps({"ok": True, "record_dir": str(record_dir), "warnings": warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
