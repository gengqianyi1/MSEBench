from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_scoring import score_candidate_library
from candidate_validator import validate_candidate_library
from prediction_validator import validate_predictions
from private_scoring import score_private
from scoring import COMPLIANCE_PENALTY_WEIGHTS


def _safe_workspace_path(workspace: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    resolved = (workspace / path).resolve()
    resolved.relative_to(workspace.resolve())
    return resolved


def _zero_property_result(reason: str) -> dict:
    return {
        "numeric_score": 0.0,
        "numeric_max_score": 50.0,
        "scoring_error": reason,
    }


def _zero_candidate_result(reason: str) -> dict:
    return {
        "candidate_score": 0.0,
        "candidate_max_score": 50.0,
        "scoring_error": reason,
    }


def _compliance_deductions(evaluation: dict) -> tuple[float, dict]:
    evidence = evaluation.get("evidence") or {}
    breakdown = {
        key: 0.0 if bool(evidence.get(key)) else weight
        for key, weight in COMPLIANCE_PENALTY_WEIGHTS.items()
    }
    return float(sum(breakdown.values())), breakdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a valid submission with private property labels and deterministic candidate-library rules."
    )
    parser.add_argument("--workspace", required=True, help="Submission workspace containing result.json and artifacts.")
    parser.add_argument("--labels", required=True, help="Evaluator-only private property labels.")
    parser.add_argument("--metadata", required=True, help="Frozen dataset metadata with measured property anchors.")
    parser.add_argument("--workflow-report", required=True, help="Public evaluator report.")
    parser.add_argument("--output")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    result = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
    artifacts = result.get("artifacts") or {}

    predictions_path = _safe_workspace_path(workspace, artifacts.get("predictions_file", "predictions.csv"))
    test_public_path = workspace / "data" / "test_public.json.gz"
    try:
        prediction_validation = validate_predictions(predictions_path, test_public_path)
    except Exception as exc:
        prediction_validation = {"passed": False, "errors": [str(exc)], "warnings": []}

    candidate_path = _safe_workspace_path(workspace, artifacts.get("candidate_library_file", "candidate_library.json.gz"))
    try:
        candidate_validation = validate_candidate_library(
            candidate_path,
            ROOT / "tasks" / "jarvis_2d_exfoliation" / "candidate_record_schema.json",
            required_count=1000,
        )
    except Exception as exc:
        candidate_validation = {"passed": False, "errors": [str(exc)], "warnings": []}

    report = json.loads(Path(args.workflow_report).read_text(encoding="utf-8"))
    evaluation = report.get("evaluation", report)
    if evaluation.get("integrity_violation"):
        raise SystemExit(json.dumps({
            "score": 0,
            "max_score": 100,
            "hard_gate_passed": False,
            "score_eligible": False,
            "integrity_violation": True,
            "workflow_evaluation": evaluation,
        }, indent=2))

    labels = pd.read_json(args.labels, orient="records", lines=True, compression="infer")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    if prediction_validation["passed"]:
        try:
            property_result = score_private(pd.read_csv(predictions_path), labels, metadata)
        except Exception as exc:
            property_result = _zero_property_result(str(exc))
    else:
        property_result = _zero_property_result(
            "; ".join(prediction_validation.get("errors", [])) or "prediction validation failed"
        )

    try:
        candidate_result = score_candidate_library(
            candidate_path=candidate_path,
            train_path=workspace / "data" / "train.json.gz",
            constraints_path=workspace / "data" / "candidate_generation_constraints.json",
            workspace=workspace,
            generator_entrypoint=_safe_workspace_path(
                workspace, artifacts.get("generator_entrypoint", "generator/generate.py")
            ),
            generator_config=_safe_workspace_path(
                workspace, artifacts.get("generator_config", "generator/config.json")
            ),
            generation_manifest=_safe_workspace_path(
                workspace, artifacts.get("generation_manifest", "generator/generation_manifest.json")
            ),
            selected_candidates=result.get("selected_generated_candidates") or [],
            chgnet_tool_attempts=int(report.get("chgnet_tool_attempts", 0)),
            chgnet_score_output_dir=Path(args.output).resolve().parent / "private_chgnet_evaluation"
            if args.output
            else workspace.parent / "private_chgnet_evaluation",
        )
    except Exception as exc:
        candidate_result = _zero_candidate_result(str(exc))

    if not candidate_validation.get("passed", False):
        constraints = json.loads(
            (workspace / "data" / "candidate_generation_constraints.json").read_text(
                encoding="utf-8"
            )
        )
        schema_cap = float(constraints.get("schema_failure_candidate_score_cap", 35.0))
        original = float(candidate_result["candidate_score"])
        candidate_result["candidate_score"] = min(original, schema_cap)
        candidate_result["candidate_schema_score_cap"] = schema_cap
        candidate_result["candidate_schema_score_cap_applied"] = original > schema_cap

    compliance_penalty, compliance_breakdown = _compliance_deductions(evaluation)
    raw_score = float(property_result["numeric_score"]) + float(candidate_result["candidate_score"])
    final_score = max(0.0, raw_score - compliance_penalty)

    final = {
        "score": round(final_score, 6),
        "max_score": property_result["numeric_max_score"] + candidate_result["candidate_max_score"],
        "raw_module_score": round(raw_score, 6),
        "hard_gate_passed": True,
        "score_eligible": True,
        "property_prediction": property_result,
        "candidate_library": candidate_result,
        "prediction_validation": prediction_validation,
        "candidate_validation": candidate_validation,
        "compliance_penalty": compliance_penalty,
        "compliance_penalty_breakdown": compliance_breakdown,
        "scoring_split": {
            "property_prediction_points": property_result["numeric_max_score"],
            "candidate_library_points": candidate_result["candidate_max_score"],
        },
    }
    text = json.dumps(final, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
