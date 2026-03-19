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
        print("usage: python3 tools/validate_record.py <record_dir>", file=sys.stderr)
        return 2

    record_dir = Path(sys.argv[1]).expanduser().resolve()
    if not record_dir.is_dir():
        print(f"record directory not found: {record_dir}", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED_FILES if not (record_dir / name).exists()]
    if missing:
        print(json.dumps({"ok": False, "missing": missing, "record_dir": str(record_dir)}, indent=2))
        return 1

    print(json.dumps({"ok": True, "record_dir": str(record_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
