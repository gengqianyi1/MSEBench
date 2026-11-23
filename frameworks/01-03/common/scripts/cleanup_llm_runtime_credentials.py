#!/usr/bin/env python3
"""Remove only launcher-created credential files from named formal runs."""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "formal_runs"
removed = []
for path in root.glob("deepseek-v4-flash-signed-20260905*/.runtime.env"):
    resolved = path.resolve()
    if root.resolve() not in resolved.parents or resolved.name != ".runtime.env":
        raise RuntimeError(f"unsafe credential cleanup target: {resolved}")
    resolved.unlink()
    removed.append(str(resolved))
print({"removed_count": len(removed), "remaining": len(list(root.glob("deepseek-v4-flash-signed-20260905*/.runtime.env")))})
