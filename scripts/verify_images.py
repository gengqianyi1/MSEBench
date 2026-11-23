"""Verify the complete organizer Docker archive against its lock files."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "images"


def verify(root: Path) -> list[str]:
    locks = {}
    errors = []
    for lock in sorted(root.glob("*/images.lock.json")):
        records = json.loads(lock.read_text(encoding="utf-8"))["images"]
        for name, record in records.items():
            if Path(name).name != name or not name.endswith(".tar"):
                errors.append(f"Invalid archive name in {lock.relative_to(root)}: {name}")
                continue
            locks[lock.parent / name] = record

    actual = set(root.rglob("*.tar"))
    for path in sorted(locks.keys() - actual):
        errors.append(f"Missing archive: {path.relative_to(root)}")
    for path in sorted(actual - locks.keys()):
        errors.append(f"Archive has no lock entry: {path.relative_to(root)}")
    if not locks:
        errors.append("No image lock entries found")

    for path in sorted(locks.keys() & actual):
        name = path.relative_to(root)
        record = locks[path]
        digest = record.get("tar_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"Missing or invalid SHA256: {name}")
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"Archive is not a regular file inside images/: {name}")
            continue
        if "bytes" in record and path.stat().st_size != record["bytes"]:
            errors.append(f"Size mismatch: {name}")
            continue
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                errors.append(f"SHA256 mismatch: {name}")
    return errors


def main() -> int:
    try:
        errors = verify(ROOT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {sum(1 for _ in ROOT.rglob('*.tar'))} locked image archives verified by SHA256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
