"""Validate the protocol-2.0 source tree or a clean delivery tree.

The source mode is intentionally usable before the formal campaign is run.
Delivery mode additionally requires the all-PASS provider preflight, all four
frozen trajectories, exact model identities, both root commitments, and an
exact MANIFEST.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import yaml

from evaluator.evaluate_v2 import EvaluationError, evaluate_run
from framework.provider_retry import DEFAULT_PROVIDER_RETRY_POLICY
from framework.schemas import (
    ACTIVE_PREFLIGHT_REPORT,
    BENCHMARK_VERSION,
    CASE_ID,
    FORMAL_CAMPAIGN_ID,
    FORMAL_MAX_PROVIDER_TURNS,
    FORMAL_MODELS,
    FORMAL_VALIDATION_REPORTS,
    PROTOCOL,
    TASK_ID,
)
from framework.trajectory import DEFAULT_PROTOCOL
from package_tools.commitments import (
    CommitmentError,
    validate_execution_commitment,
    validate_release_commitment,
)
from package_tools.manifest import validate_manifest
from package_tools.policy import PACKAGE_GATE_CHECKS, required_delivery_paths
from package_tools.scanners import (
    scan_forbidden_names,
    scan_runtime_absolute_paths,
    scan_secrets,
)
from tools.science import sha256_file, verify_case_manifest, xtb_version_record
from tools.tool_registry import load_tool_contracts


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA256 = (
    "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
)
XTB_SHA256 = (
    "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
)
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".test_tmp",
    "__pycache__",
    "audit_tmp",
    "runs",
    "runs_failed",
    "staging",
}


def _strict_json(path: Path) -> Any:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
    )
    if duplicates:
        raise ValueError(f"duplicate JSON keys in {path.name}")
    return value


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in SKIP_PARTS for part in path.relative_to(root).parts
        )
    )


def _evaluator_independent(root: Path) -> bool:
    """The independent evaluator may import only stdlib/NumPy/evaluator code."""

    for path in (root / "evaluator").glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            return False
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(
                name == "framework"
                or name.startswith("framework.")
                or name == "tools"
                or name.startswith("tools.")
                for name in names
            ):
                return False
    return True


def _load_commitment(root: Path, name: str) -> dict[str, Any]:
    value = _strict_json(root / name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not an object")
    return value


def _commitment_checks(root: Path, *, delivery: bool) -> tuple[bool, bool]:
    execution_path = root / "execution_commitment.json"
    release_path = root / "release_commitment.json"
    execution_ok = not delivery and not execution_path.exists()
    release_ok = not delivery and not release_path.exists()
    try:
        if execution_path.is_file():
            validate_execution_commitment(
                root, _load_commitment(root, "execution_commitment.json")
            )
            execution_ok = True
    except (CommitmentError, OSError, UnicodeError, ValueError, TypeError):
        execution_ok = False
    try:
        if release_path.is_file():
            validate_release_commitment(
                root, _load_commitment(root, "release_commitment.json")
            )
            release_ok = True
    except (CommitmentError, OSError, UnicodeError, ValueError, TypeError):
        release_ok = False
    return execution_ok, release_ok


def _preflight_check(root: Path, *, require_pass: bool) -> bool:
    path = root / ACTIVE_PREFLIGHT_REPORT
    if not path.is_file():
        return not require_pass
    try:
        receipt = _strict_json(path)
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    if not isinstance(receipt, dict):
        return False
    models = receipt.get("models")
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("benchmark_version") != BENCHMARK_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("native_xtb_calls") != 0
        or receipt.get("preflight_provider_turns_per_model") != 1
        or receipt.get("raw_provider_responses_persisted") is not False
        or receipt.get("formal_campaign_allowed")
        != (receipt.get("status") == "PASS")
        or not isinstance(models, dict)
        or set(models) != set(FORMAL_MODELS)
    ):
        return False
    for label, expected_model in FORMAL_MODELS.items():
        item = models.get(label)
        if (
            not isinstance(item, dict)
            or item.get("requested_model") != expected_model
            or item.get("native_xtb_calls") != 0
            or item.get("formal_trajectory") is not False
            or item.get("fallback_used") is not False
            or item.get("model_override_used") is not False
            or item.get("raw_response_persisted") is not False
        ):
            return False
        if require_pass and (
            item.get("status") != "PASS"
            or item.get("actual_model") != expected_model
            or item.get("exact_identity") is not True
            or item.get("structured_json_valid") is not True
        ):
            return False
    if require_pass:
        return (
            receipt.get("status") == "PASS"
            and receipt.get("formal_campaign_allowed") is True
        )
    return receipt.get("status") in {"PASS", "FAIL", "BLOCKED"}


def _evaluate_frozen_run(
    root: Path, label: str, expected_model: str
) -> dict[str, Any]:
    run_root = root / "trajectories" / label
    required = {
        "trajectory.jsonl",
        "final_answer.json",
        "evaluation.json",
        "run_manifest.json",
        "usage_report.json",
    }
    if not run_root.is_dir() or any(
        not (run_root / name).is_file() for name in required
    ):
        return {
            "present": False,
            "replay_identical": False,
            "terminal": False,
            "exact_identity": False,
        }
    try:
        first = evaluate_run(run_root, project_root=root, write=False)
        second = evaluate_run(run_root, project_root=root, write=False)
        stored = (run_root / "evaluation.json").read_bytes()
    except (EvaluationError, OSError, UnicodeError, ValueError, TypeError):
        return {
            "present": True,
            "replay_identical": False,
            "terminal": False,
            "exact_identity": False,
        }
    replay_identical = (
        first == second and _compact_json(first) == stored
    )
    if label == "baseline":
        actual = first.get("reconstructed", {}).get("actual_model_identity")
        exact = actual == expected_model
        legal_failure = False
    else:
        gates = first.get("eligibility", {}).get("gates", {})
        actual = first.get(
            "actual_model_identity",
            first.get("reconstructed", {}).get("actual_model_identity"),
        )
        exact = (
            actual == expected_model
            and (
                gates.get("exact_model_identity") is True
                or first.get("exact_model_identity") is True
            )
        )
        legal_failure = bool(
            first.get("run_status") in {"FAIL", "BLOCKED"}
            and first.get("legal_failure_envelope") is True
            and first.get("execution_freeze_binding", {}).get("passed") is True
        )
    return {
        "present": True,
        "replay_identical": replay_identical,
        "terminal": first.get("run_status") in {"PASS", "FAIL"}
        or legal_failure,
        "actual_model_identity": actual,
        "exact_identity": exact,
        "legal_failure_envelope": legal_failure,
        "run_status": first.get("run_status"),
        "eligible": first.get("eligible"),
        "failure_reason": (
            first.get("blocker") or first.get("failure_audit_reason")
        ),
        "total_score": first.get("total_score"),
    }


def _release_registry_matches(
    release: Any,
    formal: dict[str, dict[str, Any]],
    root: Path = ROOT,
) -> bool:
    if not isinstance(release, dict):
        return False
    models = release.get("models")
    package_gate = release.get("gates", {}).get("package", {})
    package_checks = (
        package_gate.get("checks", {})
        if isinstance(package_gate, dict)
        else {}
    )
    if (
        release.get("schema_version") != "2.0"
        or release.get("task_id") != TASK_ID
        or release.get("benchmark_version") != BENCHMARK_VERSION
        or release.get("protocol") != PROTOCOL
        or release.get("formal_campaign_id") != FORMAL_CAMPAIGN_ID
        or release.get("formal_campaign_execution_status") != "PASS"
        or release.get("benchmark_framework_status") != "PASS"
        or release.get("status") != "PASS"
        or release.get("release_ready") is not True
        or release.get("final_zip_created") is not True
        or release.get("package_submission_invoked") is not True
        or release.get("blocker_report") is not None
        or not isinstance(models, dict)
        or set(models) != set(FORMAL_MODELS)
        or not isinstance(package_gate, dict)
        or package_gate.get("status") != "PASS"
        or package_gate.get("passed") is not True
        or not isinstance(package_checks, dict)
        or any(package_checks.get(name) is not True for name in PACKAGE_GATE_CHECKS)
    ):
        return False
    for label, requested_model in FORMAL_MODELS.items():
        result = formal.get(label)
        item = models.get(label)
        if not isinstance(result, dict) or not isinstance(item, dict):
            return False
        legal_failure = result.get("legal_failure_envelope") is True
        score = item.get("score")
        expected_score = result.get("total_score")
        if (
            item.get("requested_model") != requested_model
            or item.get("validation_report")
            != FORMAL_VALIDATION_REPORTS[label]
            or not (root / FORMAL_VALIDATION_REPORTS[label]).is_file()
            or item.get("formal_trajectory_started") is not True
            or item.get("terminal") is not True
            or item.get("status") != result.get("run_status")
            or item.get("eligible") is not result.get("eligible")
            or item.get("exact_identity") is not result.get("exact_identity")
            or item.get("legal_failure_envelope") is not legal_failure
            or item.get("actual_model")
            != result.get("actual_model_identity")
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isinstance(expected_score, (int, float))
            or isinstance(expected_score, bool)
            or not math.isclose(
                float(score), float(expected_score), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            return False
        if legal_failure:
            blocker = item.get("model_blocker_report")
            if (
                result.get("run_status") not in {"FAIL", "BLOCKED"}
                or item.get("safe_status") != result.get("failure_reason")
                or not isinstance(blocker, str)
                or not blocker.startswith("reports/")
                or not (root / blocker).is_file()
            ):
                return False
        elif item.get("model_blocker_report") is not None:
            return False
    return True


def validate(root: Path, *, delivery: bool = False) -> dict[str, Any]:
    root = root.resolve(strict=True)
    required = {
        "README.md",
        "TASK.md",
        "configs/task.yaml",
        "configs/tool_contracts.json",
        "benchmark_case/reference.xyz",
        "benchmark_case/reference_structure.json",
        "benchmark_case/case_spec.json",
        "benchmark_case/case_manifest.json",
        "tools/science.py",
        "tools/sampling_state.py",
        "tools/auto_profile.py",
        "tools/tool_registry.py",
        "framework/trajectory.py",
        "framework/canonical_state.py",
        "framework/agent_runner.py",
        "framework/execution_freeze.py",
        "framework/trusted_final.py",
        "evaluator/evaluate_v2.py",
        "evaluator/profile_recompute_v2.py",
        "evaluator/scoring_v2.py",
        "package_tools/archive_validation.py",
        "package_tools/commitments.py",
        "package_tools/provenance.py",
        "scripts/provider_preflight.py",
        "scripts/run_model.py",
        "scripts/replay_trajectories.py",
        "scripts/package_submission.py",
    }
    if delivery:
        required.update(required_delivery_paths())
        required.add("MANIFEST.json")

    files = _files(root)
    relative = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(required - relative)
    compile_errors: list[str] = []
    for path in files:
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (OSError, UnicodeError, SyntaxError):
                compile_errors.append(path.relative_to(root).as_posix())

    secrets = scan_secrets(root, files)
    forbidden = scan_forbidden_names(root, files)
    absolute_paths, expected_path_literals = scan_runtime_absolute_paths(
        root, files
    )
    manifest = (
        validate_manifest(root)
        if delivery
        else {"passed": True, "errors": []}
    )
    execution_ok, release_commitment_ok = _commitment_checks(
        root, delivery=delivery
    )

    try:
        xtb_raw = xtb_version_record()
        xtb_ok = (
            xtb_raw.get("version") == "6.7.1"
            and xtb_raw.get("binary_sha256") == XTB_SHA256
            and xtb_raw.get("version_exit_code") == 0
        )
        xtb = {
            "executable_role": "allowlisted_xtb",
            "version": xtb_raw.get("version"),
            "binary_sha256": xtb_raw.get("binary_sha256"),
            "version_exit_code": xtb_raw.get("version_exit_code"),
        }
    except Exception:
        xtb, xtb_ok = {}, False

    try:
        tool_names = [item["name"] for item in load_tool_contracts()]
    except (OSError, UnicodeError, ValueError, TypeError):
        tool_names = []
    try:
        case = _strict_json(root / "benchmark_case" / "case_spec.json")
    except (OSError, UnicodeError, ValueError, TypeError):
        case = {}
    try:
        task_config = yaml.safe_load(
            (root / "configs" / "task.yaml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError, TypeError):
        task_config = {}
    expected_task_config = {
        "id": TASK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "formal_campaign_id": FORMAL_CAMPAIGN_ID,
    }
    identity_ok = bool(
        isinstance(case, dict)
        and isinstance(task_config, dict)
        and task_config.get("task") == expected_task_config
        and case.get("task_id") == TASK_ID
        and case.get("benchmark_version") == BENCHMARK_VERSION
        and case.get("case_id") == CASE_ID
        and case.get("protocol") == PROTOCOL
        and DEFAULT_PROTOCOL
        == {
            "name": "adaptive_torsion_sampling_tool_loop",
            "version": "2.0",
        }
    )
    retry_policy_ok = bool(
        DEFAULT_PROVIDER_RETRY_POLICY.maximum_total_attempts == 4
        and math.isclose(
            DEFAULT_PROVIDER_RETRY_POLICY.maximum_cumulative_delay_seconds,
            60.0,
        )
        and FORMAL_MAX_PROVIDER_TURNS == 5
    )

    try:
        smoke = _strict_json(root / "reports" / "gate1_scientific_smoke.json")
        oracle = _strict_json(
            root / "reports" / "oracle_independent_validation.json"
        )
        scientific_evidence_ok = bool(
            isinstance(smoke, dict)
            and smoke.get("passed") is True
            and [
                item.get("angle_deg") for item in smoke.get("smoke_angles", [])
            ]
            == [0, 90, 180, 270]
            and all(
                item.get("profile_eligible") is True
                for item in smoke.get("smoke_angles", [])
            )
            and isinstance(oracle, dict)
            and oracle.get("status") == "PASS"
            and oracle.get("checks", {}).get("point_count") == 72
            and oracle.get("checks", {}).get("success_count") == 72
            and oracle.get("checks", {}).get("failure_count") == 0
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        scientific_evidence_ok = False

    baseline = _evaluate_frozen_run(
        root, "baseline", "deterministic-baseline/2.0"
    )
    formal = {
        label: _evaluate_frozen_run(root, label, model)
        for label, model in FORMAL_MODELS.items()
    } if delivery else {}
    all_formal_ok = bool(
        not delivery
        or all(
            item["present"]
            and item["replay_identical"]
            and item["terminal"]
            and (
                item["exact_identity"]
                or item.get("legal_failure_envelope") is True
            )
            for item in formal.values()
        )
    )
    try:
        release = _strict_json(root / "configs" / "release.json")
    except (OSError, UnicodeError, ValueError, TypeError):
        release = {}
    release_ready = bool(
        not delivery or _release_registry_matches(release, formal, root)
    )

    checks = {
        "required_files": not missing,
        "python_compile": not compile_errors,
        "case_manifest": verify_case_manifest(root / "benchmark_case"),
        "reference_hash": (
            sha256_file(root / "benchmark_case" / "reference.xyz")
            == REFERENCE_SHA256
        ),
        "exact_two_public_actions": tool_names
        == ["run_scan_batch", "inspect_failed_attempt"],
        "evaluator_independent": _evaluator_independent(root),
        "v2_identity": identity_ok,
        "provider_retry_policy": retry_policy_ok,
        "provider_preflight": _preflight_check(
            root, require_pass=delivery
        ),
        "scientific_evidence": scientific_evidence_ok,
        "baseline_live_replay": (
            baseline["present"]
            and baseline["replay_identical"]
            and baseline["terminal"]
            and baseline["exact_identity"]
            and baseline.get("eligible") is True
        ),
        "formal_live_replay_and_exact_identity": all_formal_ok,
        "execution_commitment": execution_ok,
        "release_commitment": release_commitment_ok,
        "release_ready": release_ready,
        "xtb_6_7_1": xtb_ok,
        "no_root_env": not (root / ".env").exists(),
        "no_secret_findings": not secrets,
        "no_runtime_absolute_paths": not absolute_paths,
        "no_forbidden_files": not forbidden,
        "manifest_exact": manifest["passed"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "2.0",
        "benchmark_version": BENCHMARK_VERSION,
        "protocol": PROTOCOL,
        "mode": "delivery" if delivery else "source",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "missing": missing,
        "compile_errors": compile_errors,
        "secret_findings": secrets,
        "runtime_absolute_path_findings": absolute_paths,
        "expected_security_test_path_literals": expected_path_literals,
        "forbidden_findings": forbidden,
        "manifest": manifest,
        "baseline": baseline,
        "formal": formal,
        "xtb": xtb,
        "file_count": len(files),
    }


def _resolve_output(root: Path, output: Path) -> Path:
    candidate = output if output.is_absolute() else root / output
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("validation output must remain inside the project") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    result = validate(root, delivery=args.delivery)
    text = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output:
        target = _resolve_output(root, args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
