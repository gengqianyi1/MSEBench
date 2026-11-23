#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    source = (root / "templates/task/run_legacy_grader.py").read_bytes()
    for task in sorted((root / "tasks").glob("task-*")):
        destination = task / "tests/run_legacy_grader.py"
        destination.write_bytes(source)
        print(destination.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
