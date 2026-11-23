#!/usr/bin/env python3
"""Docker-hosted security probes for Task 09 organizer evidence replay."""
from __future__ import annotations
import importlib.util,json,os,shutil,tempfile
from pathlib import Path

WORK_BASE = Path("/workspace")
EVIDENCE_BASE = Path("/organizer_evidence")

def load_grader():
    path = Path("/tests/protocol_grader.py")
    spec = importlib.util.spec_from_file_location("task09_security_grader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def grade(work, evidence, logs):
    old = {k: os.environ.get(k) for k in ("CHEMBENCH_WORKSPACE", "CHEMBENCH_LOGS", "CHEMBENCH_ORGANIZER_EVIDENCE")}
    os.environ.update({
        "CHEMBENCH_WORKSPACE": str(work),
        "CHEMBENCH_LOGS": str(logs),
        "CHEMBENCH_ORGANIZER_EVIDENCE": str(evidence),
    })
    try:
        load_grader().main()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return json.loads((logs / "grader-result.json").read_text())

def copy_case_artifacts(src_work, dst_work):
    shutil.copytree(src_work, dst_work)

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        honest_work = root / "honest_work"
        honest_evidence = root / "honest_evidence"
        honest_logs = root / "honest_logs"
        copy_case_artifacts(WORK_BASE, honest_work)
        shutil.copytree(EVIDENCE_BASE, honest_evidence)
        honest = grade(honest_work, honest_evidence, honest_logs)
        assert honest["score"] > 0 and not honest["failure_codes"], honest

        forged_work = root / "forged_work"
        forged_evidence = root / "forged_evidence"
        forged_logs = root / "forged_logs"
        copy_case_artifacts(WORK_BASE, forged_work)
        shutil.copytree(EVIDENCE_BASE, forged_evidence)
        artifact = forged_work / "cases" / "case-1" / "case_result.json"
        value = json.loads(artifact.read_text())
        value["submission"]["evidence_id"] = "../oracle/2_fluorobiphenyl_central_torsion_v1"
        artifact.write_text(json.dumps(value))
        result = grade(forged_work, forged_evidence, forged_logs)
        assert result["score"] < honest["score"] or result["failure_codes"], result

        tamper_work = root / "tamper_work"
        tamper_evidence = root / "tamper_evidence"
        tamper_logs = root / "tamper_logs"
        copy_case_artifacts(WORK_BASE, tamper_work)
        shutil.copytree(EVIDENCE_BASE, tamper_evidence)
        case_dir = next(tamper_evidence.rglob("final_answer.json"))
        value = json.loads(case_dir.read_text())
        value["predicted_energies_kcal_mol"] = [0.0] * 72
        case_dir.write_text(json.dumps(value))
        result = grade(tamper_work, tamper_evidence, tamper_logs)
        assert result["score"] < honest["score"] or result["failure_codes"], result

        self_report_work = root / "self_report_work"
        self_report_evidence = root / "self_report_evidence"
        self_report_logs = root / "self_report_logs"
        copy_case_artifacts(WORK_BASE, self_report_work)
        shutil.copytree(EVIDENCE_BASE, self_report_evidence)
        artifact = self_report_work / "cases" / "case-1" / "case_result.json"
        value = json.loads(artifact.read_text())
        value["candidate"]["rationale"] = "FAKE INJECTED RESULT"
        value["submission"]["ok"] = True
        artifact.write_text(json.dumps(value))
        result = grade(self_report_work, self_report_evidence, self_report_logs)
        assert result["score"] == honest["score"] and not result["failure_codes"], result

    print(json.dumps({
        "status": "PASS",
        "task": 9,
        "evidence_forgery_blocked": True,
        "path_traversal_blocked": True,
        "evidence_tamper_blocked": True,
        "workspace_self_report_ignored": True,
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
