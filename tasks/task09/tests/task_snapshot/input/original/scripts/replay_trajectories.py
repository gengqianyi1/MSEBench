"""Deterministically replay frozen v2 trajectories without provider or xTB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluator.evaluate_v2 import EvaluationError, evaluate_run
from framework.schemas import ACTIVE_REPLAY_REPORT, FORMAL_MODELS
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "baseline": "deterministic-baseline/2.0",
    **FORMAL_MODELS,
}


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_commitment(run_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(run_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            (item for item in run_root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(run_root).as_posix(),
        )
    ]


def replay(label: str, run_root: Path, *, project_root: Path = ROOT) -> dict:
    """Recompute one evaluation twice and prove that replay changed no bytes."""

    expected_model = EXPECTED_MODELS[label]
    required = {
        "trajectory.jsonl",
        "final_answer.json",
        "evaluation.json",
        "run_manifest.json",
        "usage_report.json",
    }
    missing = sorted(
        name for name in required if not (run_root / name).is_file()
    )
    if missing:
        return {
            "schema_version": "2.0",
            "label": label,
            "status": "BLOCKED",
            "reason": "TRAJECTORY_INCOMPLETE",
            "missing": missing,
            "provider_called": False,
            "native_xtb_called": False,
        }

    before = _file_commitment(run_root)
    stored_bytes = (run_root / "evaluation.json").read_bytes()
    final_before = sha256_file(run_root / "final_answer.json")
    try:
        first = evaluate_run(
            run_root, project_root=project_root, write=False
        )
        second = evaluate_run(
            run_root, project_root=project_root, write=False
        )
    except (EvaluationError, OSError, UnicodeError, ValueError, TypeError) as exc:
        return {
            "schema_version": "2.0",
            "label": label,
            "status": "FAIL",
            "reason": type(exc).__name__,
            "provider_called": False,
            "native_xtb_called": False,
        }
    after = _file_commitment(run_root)
    actual_model = first.get(
        "actual_model_identity",
        first.get("reconstructed", {}).get("actual_model_identity"),
    )
    legal_failure = bool(
        first.get("run_status") in {"FAIL", "BLOCKED"}
        and first.get("legal_failure_envelope") is True
        and first.get("execution_freeze_binding", {}).get("passed") is True
    )
    if label == "baseline":
        exact_identity = actual_model == expected_model
    elif legal_failure:
        exact_identity = False
    else:
        exact_identity = bool(
            actual_model == expected_model
            and first.get("eligibility", {})
            .get("gates", {})
            .get("exact_model_identity")
            is True
        )
    checks = {
        "evaluation_byte_identical": _compact_json(first) == stored_bytes,
        "double_recompute_identical": first == second,
        "trajectory_terminal": (
            first.get("run_status") in {"PASS", "FAIL"} or legal_failure
        ),
        "identity_or_legal_failure_envelope": (
            exact_identity or legal_failure
        ),
        "final_answer_unchanged": (
            sha256_file(run_root / "final_answer.json") == final_before
        ),
        "all_run_files_unchanged": before == after,
        "provider_not_called": True,
        "native_xtb_not_called": True,
    }
    artifacts = [
        item for item in before if item["path"].startswith("artifacts/")
    ]
    result = {
        "schema_version": "2.0",
        "label": label,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_status": first.get("run_status"),
        "eligible": first.get("eligible"),
        "total_score": first.get("total_score"),
        "metrics": first.get("metrics", {}),
        "actual_model_identity": actual_model,
        "legal_failure_envelope": legal_failure,
        "trajectory_sha256": sha256_file(run_root / "trajectory.jsonl"),
        "final_answer_sha256": final_before,
        "evaluation_sha256": sha256_file(
            run_root / "evaluation.json"
        ),
        "artifact_count": len(artifacts),
        "artifact_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(artifacts)
        ).hexdigest().upper(),
        "provider_called": False,
        "native_xtb_called": False,
    }
    return result


def _resolve_output(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("replay output must remain inside the project") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels",
        nargs="*",
        default=["baseline", "kimi", "deepseek", "glm"],
        choices=sorted(EXPECTED_MODELS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ACTIVE_REPLAY_REPORT,
    )
    args = parser.parse_args()
    results = [
        replay(
            label,
            ROOT / "trajectories" / label,
            project_root=ROOT,
        )
        for label in args.labels
    ]
    payload = {
        "schema_version": "2.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "mode": "OFFLINE_REPLAY_NO_PROVIDER_NO_XTB",
        "status": (
            "PASS"
            if results and all(item["status"] == "PASS" for item in results)
            else (
                "BLOCKED"
                if any(item["status"] == "BLOCKED" for item in results)
                else "FAIL"
            )
        ),
        "results": results,
    }
    output = _resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError("replay output partial file already exists")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
