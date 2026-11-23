#!/usr/bin/env python3
"""Independently validate and summarize the frozen post-recharge v3 campaign."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate_multicase_run_v3 import (  # noqa: E402
    _manifest_valid,
    canonical_bytes,
    evaluate_case,
)
from evaluator.evaluate_suite import aggregate_suite  # noqa: E402
from evaluator.profile_recompute_v2 import recompute_automatic_profile  # noqa: E402
from evaluator.symmetry import evaluate_symmetry_aware_profile  # noqa: E402
from framework.schemas import FORMAL_MODELS  # noqa: E402


MODEL_ORDER = ("deepseek", "kimi", "glm")
EXTERNAL_BLOCKERS = {
    "NETWORK_CONNECTION_FAILED",
    "READ_TIMEOUT",
    "PROVIDER_OVERLOADED",
    "RATE_LIMITED",
}
HARTREE_TO_KCAL_MOL = 627.5094740631
REQUIRED_METADATA = {
    "benchmark_name",
    "benchmark_version",
    "task_id",
    "case_id",
    "provider",
    "endpoint",
    "requested_model_id",
    "actual_model_id",
    "temperature",
    "seed",
    "started_at",
    "ended_at",
    "duration_seconds",
    "token_usage",
    "reasoning_token_usage",
    "cache_usage",
    "provider_retries",
    "terminal_status",
    "environment_versions",
    "oracle_commitment",
    "profile_artifact_sha256",
    "artifact_hashes",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_NOT_OBJECT:{path}")
    return value


def resolve_under(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root.resolve())
    return resolved


def extrema_metrics(final_answer: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, Any]:
    curve = final_answer.get("predicted_energies_kcal_mol")
    if not isinstance(curve, list) or len(curve) != 72:
        return {
            "minimum_precision": None,
            "minimum_recall": None,
            "minimum_f1": None,
            "maximum_precision": None,
            "maximum_recall": None,
            "maximum_f1": None,
        }
    details = evaluate_symmetry_aware_profile(
        list(range(0, 360, 5)),
        curve,
        list(range(0, 360, 5)),
        oracle["relative_energies_kcal_mol"],
    )["local_extrema"]
    return {
        "minimum_precision": details["minima"]["precision"],
        "minimum_recall": details["minima"]["recall"],
        "minimum_f1": details["minima"]["f1"],
        "maximum_precision": details["maxima"]["precision"],
        "maximum_recall": details["maxima"]["recall"],
        "maximum_f1": details["maxima"]["f1"],
    }


def validate_hidden_oracles(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for case in cases:
        if case.get("visibility") != "hidden":
            continue
        case_id = str(case["case_id"])
        case_root = ROOT / str(case["case_root"])
        profile = load_json(case_root / "oracle_profile.json")
        provenance = load_json(case_root / "oracle_provenance.json")
        expected_angles = list(range(0, 360, 5))
        checks = {
            "profile_commitment": sha256(case_root / "oracle_profile.json")
            == str(case["oracle_sha256"]).upper(),
            "provenance_commitment": sha256(case_root / "oracle_provenance.json")
            == str(case["provenance_sha256"]).upper(),
            "profile_self_commitment": provenance.get("oracle_profile_sha256")
            == str(case["oracle_sha256"]).upper(),
            "grid_72": profile.get("grid_angles_deg") == expected_angles,
            "energy_count_72": len(profile.get("total_energies_hartree", [])) == 72,
            "provenance_count_72": provenance.get("native_attempt_count") == 72,
            "all_angles_eligible": provenance.get("all_angles_eligible") is True,
            "xtb_6_7_1": provenance.get("xtb_identity", {}).get("version") == "6.7.1",
        }
        attempts = case_root / "native_attempts"
        point_checks: list[bool] = []
        for index, angle in enumerate(expected_angles):
            attempt = attempts / f"angle_{angle:03d}"
            try:
                result = load_json(attempt / "result.json")
                valid = bool(
                    _manifest_valid(attempt)
                    and result.get("angle_deg") == angle
                    and result.get("profile_eligible") is True
                    and result.get("xtb_identity", {}).get("version") == "6.7.1"
                    and math.isclose(
                        float(result["total_energy_hartree"]),
                        float(profile["total_energies_hartree"][index]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and sha256(attempt / "result.json")
                    == provenance["attempt_result_sha256"][f"angle_{angle:03d}"]
                )
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                valid = False
            point_checks.append(valid)
        case_success = sum(point_checks)
        success_count += case_success
        failure_count += 72 - case_success
        checks["all_72_native_points_valid"] = all(point_checks)
        rows.append(
            {
                "case_id": case_id,
                "success_count": case_success,
                "failure_count": 72 - case_success,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "expected_point_count": 216,
        "success_count": success_count,
        "failure_count": failure_count,
        "cases": rows,
        "passed": success_count == 216 and failure_count == 0 and all(row["passed"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = resolve_under(ROOT, args.campaign)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.relative_to((ROOT / "reports").resolve())
    if output.exists() or output.is_symlink():
        raise FileExistsError("POST_RECHARGE_VALIDATION_OUTPUT_EXISTS")

    suite_path = ROOT / "organizer_private" / "suite.json"
    suite = load_json(suite_path)
    cases = [dict(item) for item in suite["cases"]]
    case_ids = [str(case["case_id"]) for case in cases]
    lock = load_json(campaign / "campaign_lock.json")
    lock_checks = {
        relative: sha256(ROOT / relative) == expected
        for relative, expected in lock["locked_file_sha256"].items()
    }
    lock_valid = bool(
        lock.get("models") == {label: FORMAL_MODELS[label] for label in MODEL_ORDER}
        and lock.get("case_oracle_commitments")
        == {str(case["case_id"]): str(case["oracle_sha256"]).upper() for case in cases}
        and all(lock_checks.values())
    )

    campaign_summary = load_json(campaign / "campaign_summary.json")
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    all_replays = True
    all_metadata = True
    all_artifacts = True
    all_native = True
    for label in MODEL_ORDER:
        evaluations: list[dict[str, Any]] = []
        scores: list[float] = []
        for case in cases:
            case_id = str(case["case_id"])
            case_root = campaign / label / case_id
            metadata = load_json(case_root / "metadata.json")
            status = load_json(case_root / "status.json")
            stored = load_json(case_root / "evaluation.json")
            final_answer = load_json(case_root / "final_answer.json")
            replay_one = evaluate_case(case_root, case, project_root=ROOT)
            replay_two = evaluate_case(case_root, case, project_root=ROOT)
            replay_match = bool(
                canonical_bytes(replay_one) == canonical_bytes(replay_two)
                and canonical_bytes(replay_one) == canonical_bytes(stored)
            )
            all_replays = all_replays and replay_match

            environment = metadata.get("environment_versions", {})
            metadata_valid = bool(
                REQUIRED_METADATA.issubset(metadata)
                and metadata.get("benchmark_name") == "Adaptive xTB Torsion Profile Agent Benchmark"
                and metadata.get("benchmark_version") == "3.0.0"
                and metadata.get("task_id") == "adaptive_xtb_torsion_profile_agent_v3"
                and metadata.get("case_id") == case_id
                and metadata.get("provider") == "siliconflow"
                and metadata.get("endpoint") == "https://api.siliconflow.cn/v1"
                and metadata.get("requested_model_id") == FORMAL_MODELS[label]
                and (
                    metadata.get("actual_model_id") in {None, FORMAL_MODELS[label]}
                )
                and metadata.get("temperature") == 0.0
                and metadata.get("seed") is None
                and datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))
                <= datetime.fromisoformat(metadata["ended_at"].replace("Z", "+00:00"))
                and isinstance(metadata.get("duration_seconds"), (int, float))
                and isinstance(metadata.get("token_usage"), Mapping)
                and isinstance(metadata.get("reasoning_token_usage"), Mapping)
                and isinstance(metadata.get("cache_usage"), Mapping)
                and isinstance(metadata.get("provider_retries"), int)
                and metadata.get("terminal_status") in {"COMPLETE", "BLOCKED", "FAILED"}
                and environment.get("python")
                and environment.get("rdkit")
                and environment.get("xtb") == "6.7.1"
                and isinstance(environment.get("os"), Mapping)
                and metadata.get("oracle_commitment", {}).get("profile_sha256")
                == str(case["oracle_sha256"]).upper()
            )
            all_metadata = all_metadata and metadata_valid

            artifact_valid = True
            for key, expected in metadata["artifact_hashes"].items():
                name = {
                    "trajectory_jsonl_sha256": "trajectory.jsonl",
                    "final_answer_json_sha256": "final_answer.json",
                    "dialogue_md_sha256": "dialogue.md",
                    "case_spec_json_sha256": f"artifacts/{case_id}/case_spec.json",
                    "reference_xyz_sha256": f"artifacts/{case_id}/reference.xyz",
                    "reference_structure_json_sha256": f"artifacts/{case_id}/reference_structure.json",
                }[key]
                artifact_valid = artifact_valid and sha256(case_root / name) == expected
            curve = final_answer.get("predicted_energies_kcal_mol")
            selected_for_profile = final_answer.get("selected_angles_deg", [])
            profile_curve = curve
            if (
                not isinstance(profile_curve, list)
                and isinstance(selected_for_profile, list)
                and selected_for_profile
            ):
                native_root_for_profile = (
                    case_root / "artifacts" / case_id / "native_attempts"
                )
                energies = [
                    float(
                        load_json(
                            native_root_for_profile
                            / f"angle_{int(angle):03d}"
                            / "result.json"
                        )["total_energy_hartree"]
                    )
                    for angle in selected_for_profile
                ]
                minimum_energy = min(energies)
                relative = [
                    (energy - minimum_energy) * HARTREE_TO_KCAL_MOL
                    for energy in energies
                ]
                profile_curve = recompute_automatic_profile(
                    [int(angle) for angle in selected_for_profile], relative
                )["predicted_energies"]
            if isinstance(profile_curve, list):
                expected_profile_hash = hashlib.sha256(
                    canonical_bytes(
                        {
                            "case_id": case_id,
                            "selected_angles_deg": selected_for_profile,
                            "predicted_energies_kcal_mol": [
                                float(value) for value in profile_curve
                            ],
                        }
                    )
                ).hexdigest().upper()
                artifact_valid = artifact_valid and metadata.get("profile_artifact_sha256") == expected_profile_hash
            else:
                artifact_valid = artifact_valid and metadata.get("profile_artifact_sha256") is None
            all_artifacts = all_artifacts and artifact_valid

            native_root = case_root / "artifacts" / case_id / "native_attempts"
            attempt_dirs = sorted(path for path in native_root.glob("angle_*") if path.is_dir())
            native_success = 0
            native_failure = 0
            for attempt in attempt_dirs:
                try:
                    result = load_json(attempt / "result.json")
                    valid = bool(
                        _manifest_valid(attempt)
                        and result.get("profile_eligible") is True
                        and result.get("xtb_identity", {}).get("version") == "6.7.1"
                    )
                except (OSError, ValueError, TypeError):
                    valid = False
                native_success += int(valid)
                native_failure += int(not valid)
            native_valid = bool(
                native_success == metadata.get("native_xtb_attempts")
                and native_failure == 0
            )
            all_native = all_native and native_valid

            failure_category = (status.get("failure") or {}).get("category")
            legal_terminal = bool(
                status.get("run_status") == "PASS"
                or (
                    status.get("run_status") == "BLOCKED"
                    and failure_category in EXTERNAL_BLOCKERS
                )
            )
            oracle = load_json(ROOT / str(case["oracle_path"]))
            extrema = extrema_metrics(final_answer, oracle)
            metrics = dict(stored.get("metrics", {}))
            metrics.update(extrema)
            score = stored.get("official_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                scores.append(float(score))
            rows.append(
                {
                    "logical_model": label,
                    "case_id": case_id,
                    "provider_status": stored.get("provider_status"),
                    "run_status": stored.get("status"),
                    "eligible": stored.get("eligible"),
                    "official_score": stored.get("official_score"),
                    "scientific_score": stored.get("scientific_score"),
                    "process_score": stored.get("process_score"),
                    "selected_angles_deg": stored.get("selected_angles_deg"),
                    "unique_valid_angles": stored.get("unique_valid_angles"),
                    "xtb_success_points": native_success,
                    "xtb_failed_points": native_failure,
                    "feedback_rounds": stored.get("feedback_rounds"),
                    "metrics": metrics,
                    "tool_calls": native_success + native_failure,
                    "model_calls": metadata.get("provider_turns"),
                    "provider_retries": metadata.get("provider_retries"),
                    "wall_time_seconds": metadata.get("duration_seconds"),
                    "requested_model_id": metadata.get("requested_model_id"),
                    "actual_model_id": metadata.get("actual_model_id"),
                    "failure_category": failure_category,
                    "metadata_valid": metadata_valid,
                    "artifact_hashes_valid": artifact_valid,
                    "native_manifests_valid": native_valid,
                    "stored_evaluation_matches_replay": replay_match,
                    "legal_terminal": legal_terminal,
                }
            )
            evaluations.append(replay_one)

        replay_suite = aggregate_suite(evaluations, case_ids)
        stored_suite = load_json(campaign / label / "suite_evaluation.json")
        suite_match = canonical_bytes(replay_suite) == canonical_bytes(stored_suite)
        all_replays = all_replays and suite_match
        aggregates[label] = {
            "case_statuses": {row["case_id"]: row["run_status"] for row in rows if row["logical_model"] == label},
            "case_scores": {row["case_id"]: row["official_score"] for row in rows if row["logical_model"] == label},
            "eligible_case_count": stored_suite["coverage"]["scored"],
            "pass_count": stored_suite["coverage"]["pass"],
            "fail_count": stored_suite["coverage"]["fail"],
            "blocked_count": stored_suite["coverage"]["blocked"],
            "scored_case_mean": statistics.fmean(scores) if scores else None,
            "macro_average_excluding_blocked": stored_suite["macro_official_score_excluding_blocked"],
            "population_standard_deviation": statistics.pstdev(scores) if scores else None,
            "minimum_score": min(scores) if scores else None,
            "maximum_score": max(scores) if scores else None,
            "ranking_eligible": stored_suite["ranking_eligible"],
            "stored_suite_matches_replay": suite_match,
        }

    oracle_validation = validate_hidden_oracles(cases)
    no_balance_blockers = all(row["failure_category"] != "INSUFFICIENT_BALANCE" for row in rows)
    legal_terminals = all(row["legal_terminal"] for row in rows)
    summary_valid = bool(
        campaign_summary.get("run_count") == 12
        and campaign_summary.get("all_replays_match") is True
        and len(campaign_summary.get("rows", [])) == 12
    )
    checks = {
        "campaign_lock": lock_valid,
        "campaign_summary_12_rows": summary_valid,
        "required_metadata_all_12": all_metadata,
        "trajectory_hash_chain_and_evaluator_replay_all_12": all_replays,
        "stored_evaluations_match_replay_all_12": all_replays,
        "profile_and_artifact_hashes_all_12": all_artifacts,
        "native_attempt_manifests_all_valid": all_native,
        "four_case_oracle_commitments": all(
            sha256(ROOT / str(case["oracle_path"])) == str(case["oracle_sha256"]).upper()
            for case in cases
        ),
        "hidden_oracle_native_points_216_of_216": oracle_validation["passed"],
        "no_insufficient_balance_blockers": no_balance_blockers,
        "only_pass_or_evidenced_external_blocked": legal_terminals,
    }
    report = {
        "schema_version": "post-recharge-campaign-validation/1.0",
        "campaign": str(campaign.relative_to(ROOT)),
        "campaign_sha256": sha256(campaign / "campaign_summary.json"),
        "campaign_lock_sha256": sha256(campaign / "campaign_lock.json"),
        "models": {label: FORMAL_MODELS[label] for label in MODEL_ORDER},
        "case_count": 4,
        "run_count": 12,
        "pass_count": sum(row["run_status"] == "PASS" for row in rows),
        "fail_count": sum(row["run_status"] == "FAIL" for row in rows),
        "blocked_count": sum(row["run_status"] == "BLOCKED" for row in rows),
        "blocker_categories": sorted(
            {row["failure_category"] for row in rows if row["failure_category"]}
        ),
        "rows": rows,
        "aggregates": aggregates,
        "oracle_validation": oracle_validation,
        "lock_file_checks": lock_checks,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "delivery_eligible_with_evidenced_external_blockers": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass_count": report["pass_count"],
                "blocked_count": report["blocked_count"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
