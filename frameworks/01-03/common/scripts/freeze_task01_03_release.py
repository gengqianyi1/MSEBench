#!/usr/bin/env python3
"""Create the signed Task 01-03 release freeze manifest."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "author_delivery_task01_03_20260905_nearfull"
TASKS = {
    "task01": ROOT / "tasks/task-01-jarvis-2d-exfoliation-candidate",
    "task02": ROOT / "tasks/task-02-oligothiophene-smd-sapt",
    "task03": ROOT / "tasks/task-03-five-ring-xtb-dipro-marcus",
}
IMAGES = {
    "task01": "chembench/std-task01-verifier:2.1.0",
    "task02": "chembench/std-task02-verifier:2.1.2",
    "task03": "chembench/std-task03-verifier:2.1.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def image_identity(tag: str) -> dict[str, str]:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{json .}}"], text=True
    )
    inspected = json.loads(raw)
    digests = inspected.get("RepoDigests") or []
    return {"tag": tag, "image_id": inspected["Id"], "repo_digest": digests[0] if digests else ""}


def main() -> None:
    approvals = {}
    for key in TASKS:
        manifest = json.loads((DELIVERY / key / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("scientific_approval") != "APPROVED" or manifest.get("production_release_approval") != "APPROVED":
            raise RuntimeError(f"{key} is not author approved")
        approvals[key] = manifest["approval_record"]
    payload = {
        "schema_version": "chembench-task01-03-release-freeze/1.0",
        "status": "FROZEN_AUTHOR_APPROVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signatory": "Shuangyan Tian",
        "signed_date": "2026-09-05",
        "approvals": approvals,
        "images": {key: image_identity(tag) for key, tag in IMAGES.items()},
        "task_files": {key: inventory(path) for key, path in TASKS.items()},
        "oracle_bundle_files": {key: inventory(DELIVERY / key) for key in TASKS},
    }
    output = DELIVERY / "release_freeze_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "task_file_counts": {k: len(v) for k, v in payload["task_files"].items()},
                      "oracle_file_counts": {k: len(v) for k, v in payload["oracle_bundle_files"].items()}}, indent=2))


if __name__ == "__main__":
    main()
