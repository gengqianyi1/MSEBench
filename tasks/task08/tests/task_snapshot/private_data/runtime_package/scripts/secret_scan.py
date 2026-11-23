from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package_tools.policy import collect_allowlisted_files
from package_tools.scanners import scan_runtime_absolute_paths, scan_secrets


def _source_files() -> list[Path]:
    forbidden_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "staging",
        "temporary_validation",
    }
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part.casefold() in forbidden_parts for part in path.relative_to(ROOT).parts)
        and path.suffix.casefold() not in {".pyc", ".pyo", ".zip"}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan benchmark artifacts without printing secrets")
    parser.add_argument("--scope", choices=["source", "package"], default="source")
    args = parser.parse_args(argv)
    files = _source_files() if args.scope == "source" else collect_allowlisted_files(ROOT)
    secrets = scan_secrets(ROOT, files)
    runtime_paths, expected_test_literals = scan_runtime_absolute_paths(ROOT, files)
    report = {
        "schema_version": "1.0",
        "passed": not secrets and not runtime_paths,
        "scope": args.scope,
        "file_count": len(files),
        "secret_finding_count": len(secrets),
        "runtime_absolute_path_finding_count": len(runtime_paths),
        "expected_security_test_path_literal_count": len(expected_test_literals),
        "secret_findings": secrets,
        "runtime_absolute_path_findings": runtime_paths,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

