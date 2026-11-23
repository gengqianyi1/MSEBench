"""Subprocess-isolated bridge from the legacy scientific evaluator to Harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from harness.canonical_replay import canonical_report_hash, canonicalize_report


def grade(context):
    bundle = Path(__file__).resolve().parent
    src = bundle / "src"
    explain_path = context.run_dir / "explain_report.json"
    harness_manifest_path = context.run_dir / "grader_harness_manifest.json"
    harness_manifest_path.write_text(
        json.dumps(context.run_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            sys.executable,
            str(src / "evaluate.py"),
            "--submission-dir",
            str(context.submission_dir),
            "--harness-manifest",
            str(harness_manifest_path),
            "--explain-json",
            str(explain_path),
        ],
        cwd=str(bundle),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    (context.run_dir / "grader_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (context.run_dir / "grader_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not explain_path.is_file():
        raise RuntimeError(f"scientific evaluator failed with exit code {completed.returncode}")
    report = json.loads(explain_path.read_text(encoding="utf-8"))
    canonical = canonicalize_report(report)
    (context.run_dir / "canonical_explain_report.json").write_text(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    replay_path = context.run_dir / "explain_report_replay.json"
    replay = subprocess.run(
        [sys.executable, str(src / "evaluate.py"), "--submission-dir", str(context.submission_dir), "--harness-manifest", str(harness_manifest_path), "--explain-json", str(replay_path)],
        cwd=str(bundle), env=env, capture_output=True, text=True, timeout=1800, check=False,
    )
    (context.run_dir / "grader_replay_stdout.log").write_text(replay.stdout, encoding="utf-8")
    (context.run_dir / "grader_replay_stderr.log").write_text(replay.stderr, encoding="utf-8")
    if replay.returncode != 0 or not replay_path.is_file():
        raise RuntimeError(f"canonical replay evaluator failed with exit code {replay.returncode}")
    replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
    first_hash = canonical_report_hash(report)
    replay_hash = canonical_report_hash(replay_report)
    replay_record = {"schema_version": "1.0.0", "matched": first_hash == replay_hash, "first_canonical_sha256": first_hash, "replay_canonical_sha256": replay_hash}
    (context.run_dir / "canonical_replay.json").write_text(json.dumps(replay_record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not replay_record["matched"]:
        raise RuntimeError("canonical replay mismatch")
    sections = {
        str(section.get("section_id")): float(section.get("earned_points", 0.0))
        for section in report.get("sections", [])
        if isinstance(section, dict)
    }
    gates = [
        str(gate.get("gate_id"))
        for gate in report.get("hard_gates", [])
        if isinstance(gate, dict) and gate.get("gate_id")
    ]
    contract_errors = report.get("contract_errors")
    if not isinstance(contract_errors, list):
        contract_errors = []
    legacy_evidence = report.get("legacy_evidence")
    if not isinstance(legacy_evidence, dict):
        legacy_evidence = {}
    return {
        "score": float(report.get("total_score", 0.0)),
        "subscores": sections,
        "sections": report.get("sections", []),
        "hard_gates": report.get("hard_gates", []),
        "failure_codes": [f"HARD_GATE:{gate}" for gate in gates],
        "evidence": {
            "explain_report": "explain_report.json",
            "canonical_explain_report": "canonical_explain_report.json",
            "canonical_sha256": first_hash,
            "hard_gates": gates,
        },
        "contract_errors": contract_errors,
        "legacy_evidence": legacy_evidence,
    }
