"""Execute release gates without invoking a model API or altering run evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package_tools.policy import collect_allowlisted_files
from package_tools.scanners import scan_runtime_absolute_paths, scan_secrets
from scripts.preflight_xtb import collect_preflight
from scripts.runtime_support import (
    RuntimeIntegrityError,
    load_json,
    pre_model_source_fingerprint,
    safe_environment,
    sha256_file,
    verify_case_template,
)
from scripts.validate_config import validate_configuration


FORMAL_MODELS = {
    "kimi": "Pro/moonshotai/Kimi-K2.6",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "glm": "zai-org/GLM-5.2",
}

# The first matching condition wins.  Availability and frozen-fixture failures
# precede downstream tests because those tests may fail as a consequence.  A
# concrete identity mismatch is more specific than a generic trajectory defect.
BLOCKER_PRIORITY = (
    "BLOCKED_CONFIG_CONFLICT",
    "BLOCKED_XTB_UNAVAILABLE",
    "BLOCKED_INITIAL_FIXTURE",
    "BLOCKED_TEST_FAILURE",
    "BLOCKED_BASELINE_FAILURE",
    "BLOCKED_SECRET_FINDING",
    "BLOCKED_PACKAGE_VALIDATION",
    "BLOCKED_MODEL_IDENTITY",
    "BLOCKED_MODEL_TRAJECTORIES",
)


def _run(command: list[str], *, timeout: int = 1200) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=safe_environment(),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    combined = completed.stdout + "\n" + completed.stderr

    def count(label: str) -> int:
        matches = re.findall(rf"(?:^|\s)(\d+)\s+{label}\b", combined)
        return int(matches[-1]) if matches else 0

    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "passed_count": count("passed"),
        "failed_count": count("failed"),
        "skipped_count": count("skipped"),
        "error_count": count("error(?:s)?"),
    }


def _test_gates() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for group in ("unit", "integration", "security", "e2e"):
        path = ROOT / "tests" / group
        if not path.is_dir() or not any(path.glob("test_*.py")):
            results[group] = {
                "passed": False,
                "exit_code": None,
                "passed_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "error_count": 1,
                "reason": "TEST_GROUP_MISSING",
            }
            continue
        results[group] = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"tests/{group}",
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        )
    return {
        "passed": all(item["passed"] for item in results.values()),
        "groups": results,
        "total_passed": sum(item["passed_count"] for item in results.values()),
        "total_skipped": sum(item["skipped_count"] for item in results.values()),
        "total_failed": sum(item["failed_count"] for item in results.values()),
        "total_errors": sum(item["error_count"] for item in results.values()),
    }


def _native_gate() -> dict[str, Any]:
    preflight = collect_preflight(ROOT / "configs" / "task.yaml")
    manifest = verify_case_template()
    regeneration_path = (
        ROOT
        / "runs_failed"
        / "initial_failure_regeneration_check"
        / "regeneration_check.json"
    )
    hessian_path = ROOT / "runs_failed" / "native_hessian_smoke" / "execution_summary.json"
    legacy_path = ROOT / "benchmark_case" / "legacy" / "artifact_manifest.json"
    regeneration = load_json(regeneration_path) if regeneration_path.is_file() else {}
    hessian = load_json(hessian_path) if hessian_path.is_file() else {}
    legacy = load_json(legacy_path)
    legacy_receipt = legacy.get("native_run_receipt", {}) if isinstance(legacy, dict) else {}
    passed = bool(
        preflight.get("passed")
        and manifest.get("source_type") == "NATIVE_XTB"
        and manifest.get("initial_failure", {}).get("verified_not_converged") is True
        and regeneration.get("passed") is True
        and regeneration.get("optimization_converged") is False
        and int(regeneration.get("complete_trajectory_frames", 0)) >= 1
        and legacy_receipt.get("optimization_converged") is True
        and hessian.get("hessian_completed") is True
        and hessian.get("normal_termination") is True
    )
    return {
        "passed": passed,
        "xtb": preflight.get("xtb", {}),
        "initial_failure_native": manifest.get("source_type") == "NATIVE_XTB",
        "initial_failure_cycle_limit": manifest.get("initial_failure", {}).get(
            "optimization_cycle_limit"
        ),
        "initial_failure_regenerated": regeneration.get("passed") is True,
        "successful_optimization_verified": legacy_receipt.get(
            "optimization_converged"
        )
        is True,
        "hessian_verified": hessian.get("hessian_completed") is True,
    }


def _package_scope_scan() -> dict[str, Any]:
    files = collect_allowlisted_files(ROOT)
    secrets = scan_secrets(ROOT, files)
    absolute, expected = scan_runtime_absolute_paths(ROOT, files)
    return {
        "passed": not secrets and not absolute,
        "file_count": len(files),
        "secret_finding_count": len(secrets),
        "runtime_absolute_path_finding_count": len(absolute),
        "expected_security_test_path_literal_count": len(expected),
    }


def _baseline_gate() -> dict[str, Any]:
    replay = _run(
        [sys.executable, "baselines/baseline_agent.py", "--mode", "replay"],
        timeout=120,
    )
    replay_fixture = ROOT / "baselines" / "replay_fixture"
    source = replay_fixture if (replay_fixture / "evaluation.json").is_file() else ROOT / "baselines" / "live"
    evaluation_path = source / "evaluation.json"
    evaluation = load_json(evaluation_path) if evaluation_path.is_file() else {}
    return {
        "passed": bool(
            replay["passed"]
            and evaluation.get("eligibility", {}).get("eligible")
            and evaluation.get("official_score") == 100
        ),
        "replay_exit_code": replay["exit_code"],
        "evaluation_byte_identical": replay["passed"],
        "evaluator_eligible": bool(evaluation.get("eligibility", {}).get("eligible")),
        "official_score": evaluation.get("official_score", 0),
    }


def _formal_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    replay = _run([sys.executable, "scripts/replay_trajectories.py"], timeout=300)
    models: dict[str, Any] = {}
    passed = replay["passed"]
    identity_mismatch_detected = False
    missing_or_ineligible_detected = not replay["passed"]
    for logical, expected in FORMAL_MODELS.items():
        directory = ROOT / "trajectories" / logical
        metadata_path = directory / "metadata.json"
        evaluation_path = directory / "evaluation.json"
        trajectory_path = directory / "trajectory.jsonl"
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        evaluation = load_json(evaluation_path) if evaluation_path.is_file() else {}
        exact = bool(
            metadata.get("logical_model") == logical
            and metadata.get("requested_model") == expected
            and metadata.get("actual_model") == expected
            and metadata.get("execution_mode") == "REAL_API"
            and metadata.get("native_mode") == "LIVE_XTB"
        )
        eligible = bool(evaluation.get("eligibility", {}).get("eligible"))
        model_passed = bool(exact and eligible and trajectory_path.is_file())
        identity_mismatch = metadata_path.is_file() and not exact
        identity_mismatch_detected = identity_mismatch_detected or identity_mismatch
        missing_or_ineligible_detected = missing_or_ineligible_detected or not bool(
            metadata_path.is_file()
            and evaluation_path.is_file()
            and trajectory_path.is_file()
            and eligible
        )
        passed = passed and model_passed
        counts = metadata.get("counts", {}) if isinstance(metadata, dict) else {}
        trusted = evaluation.get("trusted_evidence", {}) if isinstance(evaluation, dict) else {}
        artifacts = trusted.get("artifacts", {}) if isinstance(trusted, dict) else {}
        models[logical] = {
            "logical_model": logical,
            "requested_model": expected,
            "actual_model": metadata.get("actual_model"),
            "provider": metadata.get("provider"),
            "execution_mode": metadata.get("execution_mode"),
            "native_mode": metadata.get("native_mode"),
            "status": "PASS" if model_passed else "FAIL",
            "exact_identity": exact,
            "identity_mismatch": identity_mismatch,
            "trajectory_present": trajectory_path.is_file(),
            "evaluation_present": evaluation_path.is_file(),
            "evaluator_eligible": eligible,
            "tool_calls": counts.get("tool_calls", 0),
            "xtb_attempts": counts.get("native_executions", 0),
            "optimization_completed": artifacts.get("optimization_converged", False),
            "hessian_completed": artifacts.get("hessian_completed", False),
            "official_score": evaluation.get("official_score", 0),
            "trajectory_sha256": (
                sha256_file(trajectory_path) if trajectory_path.is_file() else None
            ),
            "replay_byte_identical": replay["passed"],
        }
    return {
        "passed": passed,
        "three_model_replay_exit_code": replay["exit_code"],
        "identity_mismatch_detected": identity_mismatch_detected,
        "missing_or_ineligible_detected": missing_or_ineligible_detected,
    }, models


def _release_status(
    stage: str,
    *,
    config: dict[str, Any],
    tests: dict[str, Any],
    native: dict[str, Any],
    baseline: dict[str, Any],
    scan: dict[str, Any],
    formal: dict[str, Any],
) -> str:
    """Map evidence to one deterministic release state."""

    if not config.get("passed"):
        return "BLOCKED_CONFIG_CONFLICT"
    if native.get("xtb", {}).get("status") != "AVAILABLE":
        return "BLOCKED_XTB_UNAVAILABLE"
    if not native.get("passed"):
        return "BLOCKED_INITIAL_FIXTURE"
    if not tests.get("passed"):
        return "BLOCKED_TEST_FAILURE"
    if not baseline.get("passed"):
        return "BLOCKED_BASELINE_FAILURE"
    if int(scan.get("secret_finding_count", 0)):
        return "BLOCKED_SECRET_FINDING"
    if not scan.get("passed"):
        return "BLOCKED_PACKAGE_VALIDATION"
    if stage == "pre-model":
        return "PRE_MODEL_GATES_PASS"
    if formal.get("identity_mismatch_detected"):
        return "BLOCKED_MODEL_IDENTITY"
    if not formal.get("passed"):
        return "BLOCKED_MODEL_TRAJECTORIES"
    return "READY_FOR_PACKAGE"


def _exception_status(exc: BaseException) -> str:
    marker = f"{type(exc).__name__}:{exc}".upper()
    if "XTB" in marker and any(word in marker for word in ("MISSING", "UNAVAILABLE", "NOT FOUND")):
        return "BLOCKED_XTB_UNAVAILABLE"
    if any(word in marker for word in ("CASE_TEMPLATE", "CASE MANIFEST", "FIXTURE")):
        return "BLOCKED_INITIAL_FIXTURE"
    if "SECRET" in marker:
        return "BLOCKED_SECRET_FINDING"
    if any(word in marker for word in ("PACKAGE", "MANIFEST", "ABSOLUTE_PATH")):
        return "BLOCKED_PACKAGE_VALIDATION"
    return "BLOCKED_TEST_FAILURE"


def _write_release(value: dict[str, Any]) -> None:
    path = ROOT / "configs" / "release.json"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeIntegrityError("release temporary file already exists")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def validate(stage: str, *, write_release: bool) -> dict[str, Any]:
    config = validate_configuration()
    tests = _test_gates()
    native = _native_gate()
    baseline = _baseline_gate()
    scan = _package_scope_scan()
    formal: dict[str, Any] = {"passed": False, "not_run": True}
    models = {
        logical: {
            "logical_model": logical,
            "requested_model": model,
            "status": "PENDING",
        }
        for logical, model in FORMAL_MODELS.items()
    }
    pre_model_passed = bool(
        config["passed"] and tests["passed"] and native["passed"] and baseline["passed"] and scan["passed"]
    )
    if stage == "final":
        formal, models = _formal_gate()
    status = _release_status(
        stage,
        config=config,
        tests=tests,
        native=native,
        baseline=baseline,
        scan=scan,
        formal=formal,
    )
    release = {
        "schema_version": "1.0",
        "task_id": "xtb_failure_diagnosis_restart_agent_v1",
        "status": status,
        "pre_model_gates_passed": pre_model_passed,
        "pre_model_source_sha256": pre_model_source_fingerprint(ROOT),
        "python_version": platform.python_version(),
        "gates": {
            "configuration": config,
            "tests": tests,
            "native_xtb": native,
            "baseline": baseline,
            "security_and_path_scan": scan,
            "formal_models": formal,
            "package": {
                "status": "NOT_RUN",
                "passed": False,
            },
        },
        "models": models,
    }
    if write_release:
        _write_release(release)
    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark release gates")
    parser.add_argument("--stage", choices=["pre-model", "final"], default="final")
    parser.add_argument("--write-release", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate(args.stage, write_release=args.write_release)
    except (RuntimeIntegrityError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps(
                {"status": _exception_status(exc), "error_type": type(exc).__name__},
                sort_keys=True,
            )
        )
        return 2
    public = {
        "status": result["status"],
        "pre_model_gates_passed": result["pre_model_gates_passed"],
        "test_totals": {
            key: result["gates"]["tests"][key]
            for key in ("total_passed", "total_skipped", "total_failed", "total_errors")
        },
        "native_xtb_passed": result["gates"]["native_xtb"]["passed"],
        "baseline_passed": result["gates"]["baseline"]["passed"],
        "formal_models_passed": result["gates"]["formal_models"]["passed"],
        "scan_passed": result["gates"]["security_and_path_scan"]["passed"],
    }
    print(json.dumps(public, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if result["status"] in {"PRE_MODEL_GATES_PASS", "READY_FOR_PACKAGE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
