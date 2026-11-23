"""Validate an existing ZIP without extracting outside the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_tools.archive_validation import preflight_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = preflight_archive(args.archive)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

