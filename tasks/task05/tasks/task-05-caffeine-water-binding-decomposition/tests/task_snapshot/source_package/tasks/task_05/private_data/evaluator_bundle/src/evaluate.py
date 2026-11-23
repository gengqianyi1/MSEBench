"""Deterministic evaluator for task_05_binding_decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.io_utils import atomic_write_json, load_yaml
from evaluator.core import (
    EvaluationError,
    ScoreBuilder,
    load_json_optional,
    validate_final_answer,
)
from evaluator.goal_driven import score_task2_goal_driven as score_task2


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "task_05_binding_decomposition"
REFERENCE_DIR = ROOT / "evaluation" / "reference"
REQUIRED_OUTPUT_SECTIONS = {'protocol.yaml': ('eligibility_and_protocol',), 'optimized_structures/': ('candidate_coverage_and_clustering', 'minima_and_fragment_provenance'), 'pose_clusters.json': ('candidate_coverage_and_clustering',), 'raw_energies.csv': ('minima_and_fragment_provenance', 'energy_decomposition'), 'interaction_energies.csv': ('energy_decomposition',), 'deformation_energies.csv': ('energy_decomposition',), 'binding_energies.csv': ('energy_decomposition',), 'ranking.json': ('ranking_and_robustness',), 'recovery_log.json': ('verified_feedback_and_recovery',), 'report.md': ('ranking_and_robustness',), 'scientific_interpretation.json': ('quantitative_interpretation',)}


def _present(submission_dir: Path, name: str) -> bool:
    path = submission_dir / name.rstrip("/")
    return path.is_dir() and any(path.iterdir()) if name.endswith("/") else path.is_file()


def score_submission(submission_dir: str | Path, harness_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    submission_dir = Path(submission_dir).resolve()
    task_config = load_yaml(TASK_DIR / "config" / "task.yaml", default={}) or {}
    scoring_config = load_yaml(TASK_DIR / "config" / "scoring.yaml", default={}) or {}
    if not isinstance(task_config, dict) or not isinstance(scoring_config, dict):
        raise EvaluationError("task/scoring YAML must contain objects")
    builder = ScoreBuilder(scoring_config)
    missing_root = [name for name in ("run_manifest.json",) if not (submission_dir / name).is_file()]
    if missing_root:
        builder.hard_gate(
            "mandatory_submission_root_required",
            reason="missing mandatory file(s): " + ", ".join(missing_root),
            evidence=missing_root,
            total_zero=True,
        )
        report = builder.finalize("task_05", metadata={"missing_mandatory_files": missing_root})
        return _finalize("task_05", "task_05_binding_decomposition", report)
    final = load_json_optional(submission_dir / "final_answer.json")
    final_errors = validate_final_answer(TASK_DIR, final)
    if final_errors:
        builder.hard_gate(
            "valid_final_answer_required",
            reason="final_answer.json is missing, malformed, or violates the public schema",
            evidence=final_errors,
            total_zero=True,
        )
        report = builder.finalize("task_05", metadata={"final_answer_errors": final_errors})
        return _finalize("task_05", "task_05_binding_decomposition", report)
    metadata = score_task2(TASK_DIR, submission_dir, REFERENCE_DIR, builder, harness_manifest=harness_manifest)
    missing = [str(name) for name in task_config.get("required_outputs", []) if not _present(submission_dir, str(name))]
    if missing:
        builder.notes.append("missing model-authored output(s), scored only where independently verifiable: " + ", ".join(missing))
    report = builder.finalize("task_05", metadata=metadata)
    return _finalize("task_05", "task_05_binding_decomposition", report)


def _finalize(short_id: str, long_id: str, report: dict[str, Any]) -> dict[str, Any]:
    """Raw-scientific finalize (spec 4.1/4.3): the four official score fields
    are numerically identical and any historical calibrated value would only
    live under legacy_evidence.legacy_calibrated_score."""
    report["scientific_problem_id"] = long_id
    report["contract_errors"] = sorted(set(report.get("metadata", {}).get("contract_errors", [])))
    total = float(report.get("total_score", 0.0))
    report["score"] = total
    report["task_local_score"] = total
    report["scientific_basis_score"] = total
    report["comparison_score"] = total
    report.setdefault("legacy_evidence", {})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--harness-manifest", type=Path)
    parser.add_argument("--explain-json", type=Path)
    args = parser.parse_args()
    harness_manifest = json.loads(args.harness_manifest.read_text(encoding="utf-8")) if args.harness_manifest else None
    report = score_submission(args.submission_dir, harness_manifest=harness_manifest)
    if args.explain_json:
        atomic_write_json(args.explain_json, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
