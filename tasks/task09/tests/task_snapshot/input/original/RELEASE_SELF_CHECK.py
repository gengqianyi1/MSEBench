#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {item["path"]: item for item in manifest["files"]}
actual = {
    path.relative_to(root).as_posix(): path
    for path in root.rglob("*")
    if path.is_file() and path.name != "MANIFEST.json"
}
if set(expected) != set(actual):
    raise SystemExit("MANIFEST_MEMBER_SET_MISMATCH")
for name, path in sorted(actual.items()):
    data = path.read_bytes()
    item = expected[name]
    if len(data) != item["size_bytes"]:
        raise SystemExit(f"SIZE_MISMATCH:{name}")
    if hashlib.sha256(data).hexdigest().upper() != item["sha256"]:
        raise SystemExit(f"SHA256_MISMATCH:{name}")
print(f"PASS: {len(actual)} manifest-bound files")
