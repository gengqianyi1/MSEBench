#!/usr/bin/env python3
"""Offline gate tests for the integrated MatBench repo.

Run: python3 -m pytest tests/ -q      (or: python3 tests/test_contracts.py)
See docs/RUN_GUIDE.md for Docker and Harbor checks.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_task_contracts() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_tasks.py")],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_task_dir_contracts_resolve() -> None:
    for t in ("04", "05", "06", "18"):
        spec = json.loads((ROOT / "tasks" / f"task{t}" / "task.json").read_text())
        target = (ROOT / "tasks" / f"task{t}" / spec["task_dir"]).resolve()
        assert target.is_dir(), f"task{t}: task_dir missing -> {target}"
        assert (target / "instruction.md").is_file(), f"task{t}: no instruction.md under task_dir"


def test_engine_layout() -> None:
    for rel in (
        "src/msebench/orchestrator.py",
        "src/msebench/agent/unified.py",
        "src/msebench/protocols/workspace_sidecar.py",
        "src/msebench/agents/sidecar_oracle.py",
        "src/msebench/templates/sidecar/workspace_sidecar_server.py",
        "src/msebench/schemas/harness-contract.schema.json",
        "src/msebench_harness/protocol.py",
    ):
        assert (ROOT / rel).is_file(), f"engine file missing: {rel}"
    for t in ("04", "05", "06", "18"):
        pkg = ROOT / "tasks" / f"task{t}"
        for gone in ("matbench_harness", "schemas", "agents", "templates"):
            assert not (pkg / gone).exists(), f"task{t}: redundant copy still present: {gone}"


def test_image_locks() -> None:
    manifest = json.loads((ROOT / "images/manifest.json").read_text())
    for lock_path in sorted((ROOT / "images").glob("*/images.lock.json")):
        lock = json.loads(lock_path.read_text())
        assert lock["schema"] == "matbench.images.lock/1.0"
        for tar_name in lock["images"]:
            assert (lock_path.parent / tar_name).is_file(), f"missing tar {lock_path.parent/tar_name}"
    assert manifest["schema"] == "matbench.images.manifest/1.0"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
