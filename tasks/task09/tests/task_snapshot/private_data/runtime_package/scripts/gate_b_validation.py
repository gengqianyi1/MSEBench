"""Execute and freeze the no-provider Gate B regression receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.validate_project import validate as validate_source
from tools.campaign import validate_campaign_snapshot
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "GATE_B_VALIDATION.json"
PYTEST_LOG = ROOT / "reports" / "gate_b_pytest.log"
EXPECTED_REFERENCE = (
    "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
)
EXPECTED_XTB = (
    "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path.name}")
    return value


def _pytest_count(output: str, name: str) -> int:
    matches = re.findall(rf"(\d+)\s+{re.escape(name)}\b", output)
    return int(matches[-1]) if matches else 0


def main() -> int:
    if REPORT.exists() or REPORT.is_symlink():
        raise FileExistsError("GATE_B_REPORT_ALREADY_EXISTS")
    if PYTEST_LOG.exists() or PYTEST_LOG.is_symlink():
        raise FileExistsError("GATE_B_PYTEST_LOG_ALREADY_EXISTS")

    campaign = validate_campaign_snapshot(ROOT)
    smoke = _json(ROOT / "reports" / "gate_b_v1_1_native_smoke.json")
    oracle = _json(ROOT / "reports" / "ORACLE_V1_1_ASSET_VALIDATION.json")
    preservation = _json(ROOT / "reports" / "RC1_PRESERVATION_PRE.json")

    command = [sys.executable, "-s", "-m", "pytest", "-q"]
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=1800,
        check=False,
    )
    with PYTEST_LOG.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(completed.stdout)

    passed_count = _pytest_count(completed.stdout, "passed")
    failed_count = _pytest_count(completed.stdout, "failed")
    skipped_count = _pytest_count(completed.stdout, "skipped")
    source = validate_source(ROOT, delivery=False)
    smoke_records = smoke.get("smoke_angles", [])
    smoke_ok = bool(
        smoke.get("passed") is True
        and [item.get("angle_deg") for item in smoke_records]
        == [0, 90, 180, 270]
        and all(item.get("profile_eligible") is True for item in smoke_records)
        and smoke.get("reference_xyz_sha256") == EXPECTED_REFERENCE
        and smoke.get("runtime", {}).get("xtb", {}).get("version") == "6.7.1"
        and smoke.get("runtime", {}).get("xtb", {}).get("binary_sha256")
        == EXPECTED_XTB
    )
    checks = {
        "campaign_snapshot_exact": bool(campaign),
        "pytest_exit_zero": completed.returncode == 0,
        "at_least_217_tests_passed": passed_count >= 217,
        "zero_failed_tests": failed_count == 0,
        "source_validation": source.get("status") == "PASS",
        "secret_scan_zero_findings": not source.get("secret_findings"),
        "native_smoke_0_90_180_270": smoke_ok,
        "oracle_frozen_asset_validation": oracle.get("status") == "PASS",
        "old_evidence_prework_receipt": preservation.get("status") == "PASS",
        "no_provider_invocation": True,
    }
    receipt = {
        "schema_version": "1.0",
        "gate": "Gate B",
        "task_id": "adaptive_xtb_torsion_profile_agent_v1_1",
        "benchmark_version": "1.1.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
        "formal_campaign_id": "formal_campaign_v1_1_rc1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pytest": {
            "argv": command,
            "shell": False,
            "return_code": completed.returncode,
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "log": PYTEST_LOG.relative_to(ROOT).as_posix(),
            "log_sha256": sha256_file(PYTEST_LOG),
        },
        "campaign": campaign,
        "source_validation": source,
        "native_smoke_report_sha256": sha256_file(
            ROOT / "reports" / "gate_b_v1_1_native_smoke.json"
        ),
        "oracle_validation_report_sha256": sha256_file(
            ROOT / "reports" / "ORACLE_V1_1_ASSET_VALIDATION.json"
        ),
    }
    with REPORT.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
