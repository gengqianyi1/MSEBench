"""Write or verify the immutable public case manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmark_case"
FROZEN = ("case_spec.json", "reference.xyz", "reference_structure.json")


def payload() -> dict:
    files = []
    for name in FROZEN:
        path = CASE / name
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "1.0", "algorithm": "SHA-256", "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = CASE / "case_manifest.json"
    expected = payload()
    if args.verify:
        actual = json.loads(path.read_text(encoding="utf-8"))
        print(f"case_manifest_valid={actual == expected}")
        return 0 if actual == expected else 2
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(expected))
    print(f"case_manifest_sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

