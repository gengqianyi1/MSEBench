#!/usr/bin/env python3
"""Offline scorer/evidence parity suite for author Tasks 10--13."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


TASKS = {
    "10": "task-10-matbench-bandgap-pipeline",
    "11": "task-11-lithium-electrolyte-design",
    "12": "task-12-pybamm-fast-charge",
    "13": "task-13-retrosynthesis-route",
}
ORACLE_12 = {
    "cc_c_rate": 2.0,
    "cc_until_voltage_V": 4.1,
    "cv_voltage_V": 4.1,
    "cv_until_current_C": 0.15,
    "negative_electrode_thickness_um": 100,
    "positive_electrode_thickness_um": 90,
    "electrode_porosity": 0.35,
    "particle_radius_um": 4.5,
}
ORACLE_13 = [
    {"reaction":"suzuki_coupling","reactants":["COC(=O)c1cc(Br)cc(Br)c1","Cc1ccc(B(O)O)cc1"],"product":"COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1"},
    {"reaction":"suzuki_coupling","reactants":["COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1","OB(O)c1ccccc1"],"product":"COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1"},
    {"reaction":"ester_hydrolysis","reactants":["COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1"],"product":"Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1"},
    {"reaction":"fischer_esterification","reactants":["Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1","OCCBr"],"product":"Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1"},
    {"reaction":"sn2_n_alkylation","reactants":["NCc1ccccc1","Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1"],"product":"Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1"},
    {"reaction":"n_acylation","reactants":["Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1","CC(=O)Cl"],"product":"CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1"},
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task10_cases(snapshot: Path):
    rows = list(csv.DictReader((snapshot / "private_data" / "labels.csv").open(encoding="utf-8-sig")))
    exact = [(row["id"], row["gap_expt_eV"]) for row in rows]
    return {
        "oracle_exact_private_labels": exact,
        "invalid_missing_id": exact[:-1],
        "invalid_duplicate_id": exact[:-1] + [exact[0]],
        "invalid_unknown_id": exact + [("unknown-id", "0.0")],
    }


def materialize(number: str, case: str, workspace: Path, snapshot: Path) -> None:
    if number == "10":
        rows = task10_cases(snapshot)[case]
        with (workspace / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["id", "prediction_eV"]); writer.writerows(rows)
        write_json(workspace / "pipeline.json", {"estimator": "private-parity-fixture", "seed": 0})
        write_json(workspace / "final_answer.json", {"status": "fixture"})
    elif number == "11":
        payloads = {
            "oracle_known_valid": {"smiles": "COCCOC(=O)C(F)OC"},
            "invalid_parse": {"smiles": "not-smiles"},
            "invalid_reserved": {"smiles": "CC(OC)(OC)C(OC)OC"},
            "invalid_three_oxygen_neighbors": {"smiles": "COC(OC)(OC)OC"},
        }
        write_json(workspace / "final_answer.json", payloads[case])
    elif number == "12":
        payloads = {
            "oracle_public_envelope": {"candidate": ORACLE_12},
            "invalid_missing_fields": {"candidate": {"cc_c_rate": 2.0}},
            "invalid_out_of_range": {"candidate": {**ORACLE_12, "cc_c_rate": 9.0}},
            "invalid_non_finite": {"candidate": {**ORACLE_12, "cc_c_rate": float("nan")}},
        }
        write_json(workspace / "final_answer.json", payloads[case])
    else:
        payloads = {
            "oracle_six_step_route": {"route": ORACLE_13},
            "invalid_empty_route": {"route": []},
            "invalid_unknown_reaction": {"route": [
                {"reaction": "magic", "reactants": ["CCO"], "product": "CCO"},
                {"reaction": "magic", "reactants": ["CCO"], "product": "CCO"},
            ]},
            "invalid_wrong_final": {"route": ORACLE_13[:-1]},
        }
        write_json(workspace / "final_answer.json", payloads[case])


def case_names(number: str):
    return {
        "10": ("oracle_exact_private_labels", "invalid_missing_id", "invalid_duplicate_id", "invalid_unknown_id"),
        "11": ("oracle_known_valid", "invalid_parse", "invalid_reserved", "invalid_three_oxygen_neighbors"),
        "12": ("oracle_public_envelope", "invalid_missing_fields", "invalid_out_of_range", "invalid_non_finite"),
        "13": ("oracle_six_step_route", "invalid_empty_route", "invalid_unknown_reaction", "invalid_wrong_final"),
    }[number]


def direct_grade(snapshot: Path, workspace: Path, run_dir: Path):
    grader = load_module("legacy_" + snapshot.name.replace("-", "_"), snapshot / "private_grader" / "grader.py")
    context = SimpleNamespace(
        task_dir=snapshot,
        submission_dir=workspace,
        private_data_dir=snapshot / "private_data",
        run_dir=run_dir,
    )
    return grader.grade(context)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "schema_version": "chembench-author-parity/1.0",
        "scope": "offline_submission_and_scorer_parity_not_live_model_trajectory_parity",
        "raw_score_tolerance": 1e-9,
        "tasks": {},
    }
    for number, dirname in TASKS.items():
        task = root / "tasks" / dirname
        snapshot = task / "tests" / "task_snapshot"
        adapter_module = load_module("adapter_" + number, task / "tests" / "private_suite_adapter.py")
        adapter = adapter_module.create_adapter(snapshot)
        task_rows = []
        protocol_grader_end_to_end = False
        for case in case_names(number):
            with tempfile.TemporaryDirectory(prefix=f"chembench-{number}-") as temporary:
                temp = Path(temporary); workspace = temp / "workspace"; workspace.mkdir()
                materialize(number, case, workspace, snapshot)
                legacy = direct_grade(snapshot, workspace, temp / "legacy-run")
                evidence = temp / "organizer-evidence"
                manifest = adapter.capture_evidence(workspace, evidence)
                replay = adapter.grade_evidence(evidence, temp / "replay-run")
                score_delta = abs(float(legacy["score"]) - float(replay["score"]))
                parity = score_delta <= 1e-9 and legacy.get("failure_codes") == replay.get("failure_codes") and legacy == replay
                task_rows.append({
                    "case": case,
                    "legacy_score": legacy["score"],
                    "replay_score": replay["score"],
                    "score_delta": score_delta,
                    "failure_codes": replay.get("failure_codes", []),
                    "evidence_id": manifest["evidence_id"],
                    "exact_result_parity": parity,
                })
                if not parity:
                    raise AssertionError(f"Task {number} parity failed for {case}")
                if case == case_names(number)[0]:
                    protocol_run = temp / "protocol-run"
                    protocol_logs = temp / "protocol-logs"
                    environment = {
                        **os.environ,
                        "CHEMBENCH_WORKSPACE": str(workspace),
                        "CHEMBENCH_LOGS": str(protocol_logs),
                        "CHEMBENCH_RUN": str(protocol_run),
                        "CHEMBENCH_TASK_SNAPSHOT": str(snapshot),
                        "PYBAMM_DISABLE_TELEMETRY": "true",
                    }
                    completed = subprocess.run(
                        [sys.executable, str(task / "tests" / "protocol_grader.py")],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )
                    if completed.returncode != 0:
                        raise AssertionError(
                            f"Task {number} protocol grader failed: {completed.stderr[-2000:]}"
                        )
                    reward_row = json.loads((protocol_logs / "reward.json").read_text(encoding="utf-8"))
                    grader_result = json.loads(
                        (protocol_logs / "grader-result.json").read_text(encoding="utf-8")
                    )
                    protocol_grader_end_to_end = (
                        set(reward_row) == {"reward"}
                        and abs(float(reward_row["reward"]) - float(replay["score"]) / 100.0) <= 1e-9
                        and abs(float(grader_result["raw_score"]) - float(replay["score"])) <= 1e-9
                        and grader_result["status"] == "scored"
                    )
                    if not protocol_grader_end_to_end:
                        raise AssertionError(f"Task {number} protocol grader reward mismatch")
                artifact = evidence / "submission" / manifest["artifacts"][0]["path"]
                artifact.write_bytes(artifact.read_bytes() + b"\n")
                try:
                    adapter.grade_evidence(evidence, temp / "tamper-run")
                except ValueError as exc:
                    if str(exc) != "evidence_digest_mismatch":
                        raise
                else:
                    raise AssertionError("tampered organizer evidence was accepted")
        report["tasks"][number] = {
            "cases": task_rows,
            "oracle_score": task_rows[0]["replay_score"],
            "invalid_case_count": 3,
            "all_exact_result_parity": all(row["exact_result_parity"] for row in task_rows),
            "tamper_detection": "passed",
            "protocol_grader_end_to_end": protocol_grader_end_to_end,
        }
    if args.output:
        output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True); write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
