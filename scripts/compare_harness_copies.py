#!/usr/bin/env python3
"""Compare per-family harness copies against the authoritative src/msebench_harness.

Usage:
  python3 scripts/compare_harness_copies.py            # report only
  python3 scripts/compare_harness_copies.py --dup-only # print only the byte-identical copies

Authoritative source: src/msebench_harness/ (8 files: __init__ budget cli gateway grader protocol state validation)
Copies:
  frameworks/01-03/common/matbench_harness/
  frameworks/07-09/framework/matbench_harness/
  frameworks/10-13/harbor_agent_harness/chembench_harness/
  frameworks/14-17/chembench_harness/
  frameworks/19-20/chembench_harness/
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/msebench_harness"
COPIES = [
    ROOT / "frameworks/01-03/common/matbench_harness",
    ROOT / "frameworks/07-09/framework/matbench_harness",
    ROOT / "frameworks/10-13/harbor_agent_harness/chembench_harness",
    ROOT / "frameworks/14-17/chembench_harness",
    ROOT / "frameworks/19-20/chembench_harness",
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dup-only", action="store_true",
                        help="print only copies that are byte-identical to src")
    args = parser.parse_args()
    src_files = {p.name: md5(p) for p in sorted(SRC.glob("*.py"))}
    if not src_files:
        print(f"missing authoritative source: {SRC}", file=sys.stderr)
        return 2
    print(f"authoritative: {SRC.relative_to(ROOT)} ({len(src_files)} files)")
    for name, digest in src_files.items():
        print(f"  {digest[:12]}  {name}")
    print()
    for copy in COPIES:
        rel = copy.relative_to(ROOT)
        if not copy.is_dir():
            print(f"{rel}: (absent)")
            continue
        cf = {p.name: md5(p) for p in sorted(copy.glob("*.py"))}
        same = [k for k in cf if k in src_files and cf[k] == src_files[k]]
        diff = [k for k in cf if k in src_files and cf[k] != src_files[k]]
        only = [k for k in cf if k not in src_files]
        miss = [k for k in src_files if k not in cf]
        verdict = "DEDUP-CANDIDATE" if not diff and not only else "DIVERGED"
        if args.dup_only and verdict != "DEDUP-CANDIDATE":
            continue
        print(f"{verdict}  {rel}  ({len(cf)} files): identical={len(same)} differ={diff} extra={only} missing={miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
