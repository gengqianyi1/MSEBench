from __future__ import annotations

import argparse
from pathlib import Path

from .validation import validate_repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m msebench_harness.cli")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_repository(Path(args.path))
        print(report.to_json())
        return 0 if report.ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
