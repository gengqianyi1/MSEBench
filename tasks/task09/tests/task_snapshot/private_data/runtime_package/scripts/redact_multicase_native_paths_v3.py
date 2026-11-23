"""Preserve raw logs outside release scope and redact local executable paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_multicase_suite_v3 import artifact_manifest, canonical_bytes, sha256


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "organizer_private" / "cases"
RAW_ROOT = ROOT.parent / "validation" / "native_raw_logs"
WINDOWS_XTB_PATH_RE = re.compile(
    rb"(?i)(?:[A-Z]:\\(?:[^\\\r\n\"]+\\)*xtb\.exe)"
)


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def main() -> int:
    count = 0
    for case_root in sorted(path for path in CASES.iterdir() if path.is_dir()):
        for attempt in sorted((case_root / "native_attempts").iterdir()):
            stdout_path = attempt / "stdout.log"
            result_path = attempt / "result.json"
            raw = stdout_path.read_bytes()
            raw_hash = hashlib.sha256(raw).hexdigest().upper()
            raw_target = RAW_ROOT / case_root.name / attempt.name / "stdout.raw.log"
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            if raw_target.exists() and raw_target.read_bytes() != raw:
                raise RuntimeError(f"different raw evidence already exists: {raw_target}")
            raw_target.write_bytes(raw)
            projected, replacement_count = WINDOWS_XTB_PATH_RE.subn(b"xtb", raw)
            if replacement_count == 0:
                raise RuntimeError(f"expected native program path absent: {stdout_path}")
            stdout_path.write_bytes(projected)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("stdout_sha256") != raw_hash:
                raise RuntimeError(f"pre-redaction hash mismatch: {result_path}")
            result["stdout_raw_sha256"] = raw_hash
            result["stdout_sha256"] = sha256(stdout_path)
            result["stdout_projection"] = "native-program-call-path-redaction/1.0"
            write_json(result_path, result)
            write_json(attempt / "attempt_manifest.json", artifact_manifest(attempt))
            count += 1
        provenance_path = case_root / "oracle_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["native_log_projection"] = "native-program-call-path-redaction/1.0"
        provenance["raw_logs_retained_outside_release_scope"] = True
        provenance["attempt_result_sha256"] = {
            attempt.name: sha256(attempt / "result.json")
            for attempt in sorted((case_root / "native_attempts").iterdir())
        }
        write_json(provenance_path, provenance)
        write_json(case_root / "case_manifest.json", artifact_manifest(case_root))
    print(json.dumps({"redacted_stdout_logs": count, "raw_evidence_root": str(RAW_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
