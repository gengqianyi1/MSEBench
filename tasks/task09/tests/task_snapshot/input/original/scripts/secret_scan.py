"""Scan the project/release without ever loading credential values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_tools.scanners import scan_forbidden_names, scan_secrets


ROOT = Path(__file__).resolve().parents[1]
SKIP = {"audit_tmp", ".test_tmp", ".pytest_cache", "staging", "runs", "__pycache__", ".git"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    files = [
        path for path in root.rglob("*")
        if path.is_file() and not any(part in SKIP for part in path.relative_to(root).parts)
    ]
    secrets = scan_secrets(root, files)
    forbidden = scan_forbidden_names(root, files)
    result = {
        "status": "PASS" if not secrets and not forbidden and not (root / ".env").exists() else "FAIL",
        "file_count": len(files),
        "secret_finding_count": len(secrets),
        "forbidden_finding_count": len(forbidden),
        "real_env_present": (root / ".env").exists(),
        "secret_findings": secrets,
        "forbidden_findings": forbidden,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

