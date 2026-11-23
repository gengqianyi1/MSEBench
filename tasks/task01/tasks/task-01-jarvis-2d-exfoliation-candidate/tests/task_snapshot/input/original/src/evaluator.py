from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pickletools
import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

try:
    from .prediction_validator import validate_predictions
    from .candidate_validator import read_jsonl, validate_candidate_library
    from .scoring import score_workflow
except ImportError:  # script-style execution
    from prediction_validator import validate_predictions
    from candidate_validator import read_jsonl, validate_candidate_library
    from scoring import score_workflow

TARGET = "exfoliation_energy_mev_atom"


def _safe_relpath(workspace: Path, rel: Any) -> Path | None:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    path = (workspace / rel).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError:
        return None
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_errors(result: Dict[str, Any], schema_path: Path) -> List[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [f"schema: {'/'.join(str(x) for x in err.absolute_path) or '<root>'}: {err.message}" for err in sorted(validator.iter_errors(result), key=lambda e: list(e.absolute_path))]


def _load_public_data(workspace: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    data = workspace / "data"
    train = pd.read_json(data / "train.json.gz", orient="records", lines=True, compression="infer")
    val = pd.read_json(data / "validation.json.gz", orient="records", lines=True, compression="infer")
    test = pd.read_json(data / "test_public.json.gz", orient="records", lines=True, compression="infer")
    meta = json.loads((data / "dataset_metadata.json").read_text(encoding="utf-8"))
    return train, val, test, meta


def _invalid_structure_count(frames: List[pd.DataFrame]) -> int:
    try:
        from pymatgen.core import Structure
    except Exception:
        return -1
    invalid = 0
    for frame in frames:
        for value in frame.get("structure", []):
            try:
                Structure.from_dict(value)
            except Exception:
                invalid += 1
    return invalid


def _chemical_systems_and_elements(frames: List[pd.DataFrame]) -> Tuple[set[str], set[str]]:
    systems: set[str] = set()
    elements: set[str] = set()
    for frame in frames:
        if "chemical_system" in frame.columns:
            for raw in frame["chemical_system"].dropna().astype(str):
                parts = [x for x in raw.split("-") if x]
                if parts:
                    systems.add("-".join(sorted(parts)))
                    elements.update(parts)
    return systems, elements


def _audit_evidence(result: Dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, meta: Dict[str, Any], workspace: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    dataset = result.get("dataset") or {}
    audit = result.get("data_audit") or {}
    expected_counts = (len(train), len(val), len(test))
    got_counts = (dataset.get("n_train"), dataset.get("n_validation"), dataset.get("n_test"))
    if got_counts != expected_counts:
        errors.append(f"dataset counts mismatch: reported={got_counts}, expected={expected_counts}")
    if dataset.get("snapshot_id") != meta.get("snapshot_id"):
        errors.append("snapshot_id does not match dataset metadata")

    all_ids = pd.concat([
        train[["sample_id"]].assign(split="train"),
        val[["sample_id"]].assign(split="validation"),
        test[["sample_id"]].assign(split="test"),
    ], ignore_index=True)
    duplicate_count = int(all_ids["sample_id"].astype(str).duplicated(keep=False).sum())
    if audit.get("duplicate_id_count") != duplicate_count:
        errors.append(f"duplicate_id_count mismatch: {audit.get('duplicate_id_count')} != {duplicate_count}")

    missing_train = int(pd.to_numeric(train[TARGET], errors="coerce").isna().sum())
    missing_val = int(pd.to_numeric(val[TARGET], errors="coerce").isna().sum())
    if audit.get("missing_train_target_count") != missing_train:
        errors.append("missing_train_target_count mismatch")
    if audit.get("missing_validation_target_count") != missing_val:
        errors.append("missing_validation_target_count mismatch")

    labeled = pd.concat([train[TARGET], val[TARGET]], ignore_index=True).astype(float)
    expected_stats = {
        "target_min_mev_atom": float(labeled.min()),
        "target_q10_mev_atom": float(labeled.quantile(0.1)),
        "target_median_mev_atom": float(labeled.median()),
        "target_mean_mev_atom": float(labeled.mean()),
        "target_q90_mev_atom": float(labeled.quantile(0.9)),
        "target_max_mev_atom": float(labeled.max()),
        "target_std_mev_atom": float(labeled.std(ddof=1)),
    }
    for key, expected in expected_stats.items():
        try:
            reported = float(audit.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key} is not numeric")
            continue
        if not np.isclose(reported, expected, rtol=1e-6, atol=1e-6):
            errors.append(f"{key} mismatch: {reported} != {expected}")

    systems, elements = _chemical_systems_and_elements([train, val, test])
    if systems and audit.get("chemical_system_count") != len(systems):
        errors.append("chemical_system_count mismatch")
    if elements:
        reported_elements = set(audit.get("elements") or [])
        if reported_elements != elements or audit.get("element_count") != len(elements):
            errors.append("elements or element_count mismatch")

    invalid = _invalid_structure_count([train, val, test])
    if invalid >= 0 and audit.get("invalid_structure_count") != invalid:
        errors.append("invalid_structure_count mismatch")

    hashes = meta.get("public_file_sha256") or {}
    hashes_ok = bool(hashes)
    for name, expected in hashes.items():
        path = workspace / "data" / name
        if not path.exists() or _sha256(path) != expected:
            hashes_ok = False
    if dataset.get("checksums_verified") is not hashes_ok:
        errors.append("checksums_verified does not match recomputed public hashes")
    return len(errors) == 0, errors


def _top10_evidence(result: Dict[str, Any], prediction_file: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    pred = pd.read_csv(prediction_file)
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["predicted_exfoliation_energy_mev_atom"] = pd.to_numeric(pred["predicted_exfoliation_energy_mev_atom"], errors="raise")
    expected = pred.sort_values(["predicted_exfoliation_energy_mev_atom", "sample_id"], kind="mergesort").head(10).reset_index(drop=True)
    recs = result.get("recommended_candidates") or []
    if len(recs) != 10:
        return False, ["recommended_candidates must contain exactly 10 rows"]
    for i, rec in enumerate(recs):
        if rec.get("rank") != i + 1:
            errors.append(f"candidate rank mismatch at index {i}")
        if str(rec.get("sample_id")) != expected.loc[i, "sample_id"]:
            errors.append(f"candidate sample_id mismatch at rank {i + 1}")
        try:
            value = float(rec.get("predicted_exfoliation_energy_mev_atom"))
        except (TypeError, ValueError):
            errors.append(f"candidate value is not numeric at rank {i + 1}")
        else:
            if not np.isclose(value, float(expected.loc[i, "predicted_exfoliation_energy_mev_atom"]), rtol=1e-9, atol=1e-9):
                errors.append(f"candidate value mismatch at rank {i + 1}")
    return len(errors) == 0, errors


def _artifact_evidence(result: Dict[str, Any], workspace: Path) -> Tuple[bool, bool, List[str]]:
    errors: List[str] = []
    artifacts = result.get("artifacts") or {}
    resolved: Dict[str, Path] = {}
    for key in ("predictions_file", "model_file", "feature_names_file", "training_log"):
        path = _safe_relpath(workspace, artifacts.get(key))
        if path is None:
            errors.append(f"unsafe or missing artifact path: {key}")
        elif not path.is_file():
            errors.append(f"artifact not found: {artifacts.get(key)}")
        else:
            resolved[key] = path

    reproducible = False
    validation_evidence = False
    if len(resolved) == 4:
        try:
            feature_names = json.loads(resolved["feature_names_file"].read_text(encoding="utf-8"))
            if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
                raise ValueError("feature_names.json must be a string array")
            if len(feature_names) != int((result.get("model") or {}).get("n_features")):
                raise ValueError("feature name count does not match model.n_features")
            model_bytes = resolved["model_file"].read_bytes()
            if not model_bytes or len(model_bytes) > 500_000_000:
                raise ValueError("model artifact is empty or exceeds the 500 MB size limit")
            # Inspect enough of the pickle stream to reject empty/compressed
            # placeholders without executing the untrusted artifact. Valid
            # uncompressed joblib files can include numpy payload bytes that
            # pickletools may report as unknown opcodes, so do not require a
            # full-stream disassembly.
            if model_bytes.startswith(b"\x1f\x8b"):
                raise ValueError("model artifact appears gzip-compressed")
            parsed_opcodes = 0
            try:
                for opcode, _, _ in pickletools.genops(model_bytes):
                    parsed_opcodes += 1
                    if opcode.name == "STOP":
                        break
            except Exception as exc:
                if parsed_opcodes == 0:
                    raise ValueError(f"model artifact is not a readable pickle/joblib stream: {exc}") from exc
            log = json.loads(resolved["training_log"].read_text(encoding="utf-8"))
            required_log = {"package_versions", "random_seeds", "data_hashes", "validation_metrics", "selected_hyperparameters", "elapsed_seconds", "refit_policy"}
            missing = sorted(required_log - set(log)) if isinstance(log, dict) else sorted(required_log)
            if missing:
                raise ValueError(f"training_log missing keys: {missing}")
            reproducible = True
            reported = result.get("validation") or {}
            logged = log.get("validation_metrics") or {}
            metric_aliases = {
                "mae_mev_atom": [
                    "mae_mev_atom",
                    "best_model_mae_mev_atom",
                    "learned_model_mae_mev_atom",
                    "selected_model_mae_mev_atom",
                    "final_mae_mev_atom",
                ],
                "rmse_mev_atom": [
                    "rmse_mev_atom",
                    "best_model_rmse_mev_atom",
                    "learned_model_rmse_mev_atom",
                    "selected_model_rmse_mev_atom",
                    "final_rmse_mev_atom",
                ],
                "spearman": [
                    "spearman",
                    "best_model_spearman",
                    "learned_model_spearman",
                    "selected_model_spearman",
                    "final_spearman",
                ],
                "constant_baseline_mae_mev_atom": ["constant_baseline_mae_mev_atom"],
                "constant_baseline_rmse_mev_atom": ["constant_baseline_rmse_mev_atom"],
                "invariance_mae_mev_atom": ["invariance_mae_mev_atom"],
            }

            def logged_metric(key: str) -> Any:
                for alias in metric_aliases[key]:
                    if alias in logged:
                        return logged[alias]
                return None

            validation_evidence = all(
                key in reported and logged_metric(key) is not None
                and np.isclose(float(logged_metric(key)), float(reported[key]), rtol=1e-8, atol=1e-8)
                for key in metric_aliases
            )
            if not validation_evidence:
                errors.append("training_log validation metrics do not match result.json")
        except Exception as exc:
            errors.append(f"artifact parse/load failure: {exc}")
    return reproducible, validation_evidence, errors



def _candidate_evidence(result: Dict[str, Any], workspace: Path) -> Tuple[bool, bool, Dict[str, Any], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    artifacts = result.get("artifacts") or {}
    required = {
        "candidate_library_file": artifacts.get("candidate_library_file"),
        "candidate_summary_file": artifacts.get("candidate_summary_file"),
        "generator_entrypoint": artifacts.get("generator_entrypoint"),
        "generator_config": artifacts.get("generator_config"),
        "generation_manifest": artifacts.get("generation_manifest"),
    }
    resolved: Dict[str, Path] = {}
    for key, value in required.items():
        path = _safe_relpath(workspace, value)
        if path is None:
            errors.append(f"unsafe or missing candidate artifact path: {key}")
        elif not path.is_file():
            errors.append(f"candidate artifact not found: {value}")
        else:
            resolved[key] = path

    report: Dict[str, Any] = {"passed": False, "errors": ["candidate artifacts unavailable"], "warnings": []}
    integrity = False
    artifacts_ok = len(resolved) == len(required)
    if artifacts_ok:
        record_schema = Path(__file__).resolve().parents[1] / "tasks" / "jarvis_2d_exfoliation" / "candidate_record_schema.json"
        report = validate_candidate_library(resolved["candidate_library_file"], record_schema, required_count=1000)
        errors.extend(report.get("errors", []))
        warnings.extend(report.get("warnings", []))
        integrity = bool(report.get("passed"))
        try:
            records = read_jsonl(resolved["candidate_library_file"])
            by_id = {str(record.get("candidate_id")): record for record in records}
            selected = result.get("selected_generated_candidates") or []
            selected_ids = [str(item.get("candidate_id", "")) for item in selected]
            if len(selected_ids) != 5 or len(set(selected_ids)) != 5:
                errors.append("selected_generated_candidates must contain five distinct candidate IDs")
            for item in selected:
                candidate_id = str(item.get("candidate_id", ""))
                record = by_id.get(candidate_id)
                if record is None:
                    errors.append(f"selected candidate is absent from candidate library: {candidate_id}")
                    continue
                observed = float(item.get("predicted_exfoliation_energy_mev_atom"))
                expected = float(record.get("predicted_exfoliation_energy_mev_atom"))
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-8):
                    errors.append(f"selected candidate prediction mismatch: {candidate_id}")
        except Exception as exc:
            errors.append(f"selected candidate consistency check failed: {exc}")

        try:
            summary = json.loads(resolved["candidate_summary_file"].read_text(encoding="utf-8"))
            summary_schema = Path(__file__).resolve().parents[1] / "tasks" / "jarvis_2d_exfoliation" / "candidate_summary_schema.json"
            summary_errors = _schema_errors(summary, summary_schema)
            if summary_errors:
                errors.extend(f"candidate_summary {item}" for item in summary_errors)
            claimed = result.get("candidate_generation") or {}
            for key in ("submitted_count", "claimed_valid_count", "claimed_pathological_count", "claimed_unique_count", "claimed_novel_count", "random_seed", "generation_strategy_summary"):
                if summary.get(key) != claimed.get(key):
                    errors.append(f"candidate_summary.json does not match result.json for {key}")
        except Exception as exc:
            errors.append(f"candidate_summary.json is unreadable: {exc}")

        try:
            config = json.loads(resolved["generator_config"].read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("generator config must be a JSON object")
        except Exception as exc:
            artifacts_ok = False
            errors.append(f"generator/config.json is unreadable: {exc}")
        try:
            manifest = json.loads(resolved["generation_manifest"].read_text(encoding="utf-8"))
            required_manifest = {"random_seed", "input_files", "library_versions", "generation_parameters", "submitted_count", "interface_version"}
            if not isinstance(manifest, dict) or not required_manifest.issubset(manifest):
                raise ValueError(f"missing keys: {sorted(required_manifest - set(manifest or {}))}")
            if manifest.get("submitted_count") != 1000:
                raise ValueError("submitted_count must be 1000")
            if manifest.get("interface_version") != "candidate-generator-v1":
                raise ValueError("interface_version must be candidate-generator-v1")
        except Exception as exc:
            artifacts_ok = False
            errors.append(f"generation_manifest.json is invalid: {exc}")

    return integrity, artifacts_ok, report, errors, warnings

def evaluate_result(result: Dict[str, Any], workspace: Path | None = None, schema_path: Path | None = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    evidence = {
        "prediction_integrity": False,
        "schema_and_consistency": False,
        "data_audit_accuracy": False,
        "validation_evidence": False,
        "reproducibility_artifacts": False,
        "scientific_interpretation": False,
        "candidate_submission_integrity": False,
        "candidate_generator_artifacts": False,
    }

    if schema_path is None:
        schema_path = Path(__file__).resolve().parents[1] / "tasks" / "jarvis_2d_exfoliation" / "result_schema.json"
    schema_errors = _schema_errors(result, schema_path)
    errors.extend(schema_errors)

    prediction_report: Dict[str, Any] = {"passed": False, "errors": ["workspace unavailable"], "warnings": []}
    candidate_report: Dict[str, Any] = {"passed": False, "errors": ["workspace unavailable"], "warnings": []}
    top10_ok = False
    if workspace is not None:
        workspace = workspace.resolve()
        try:
            train, val, test, meta = _load_public_data(workspace)
        except Exception as exc:
            errors.append(f"could not load public data: {exc}")
        else:
            pred_path = _safe_relpath(workspace, (result.get("artifacts") or {}).get("predictions_file", "predictions.csv"))
            if pred_path is None:
                errors.append("unsafe predictions path")
            else:
                prediction_report = validate_predictions(pred_path, workspace / "data" / "test_public.json.gz")
                errors.extend(prediction_report.get("errors", []))
                warnings.extend(prediction_report.get("warnings", []))
                evidence["prediction_integrity"] = bool(prediction_report.get("passed"))
                if prediction_report.get("passed"):
                    top10_ok, top10_errors = _top10_evidence(result, pred_path)
                    errors.extend(top10_errors)
            audit_ok, audit_errors = _audit_evidence(result, train, val, test, meta, workspace)
            evidence["data_audit_accuracy"] = audit_ok
            errors.extend(audit_errors)

        reproducible, validation_ok, artifact_errors = _artifact_evidence(result, workspace)
        evidence["reproducibility_artifacts"] = reproducible
        evidence["validation_evidence"] = validation_ok
        errors.extend(artifact_errors)

        candidate_ok, generator_ok, candidate_report, candidate_errors, candidate_warnings = _candidate_evidence(result, workspace)
        evidence["candidate_submission_integrity"] = candidate_ok
        evidence["candidate_generator_artifacts"] = generator_ok
        errors.extend(candidate_errors)
        warnings.extend(candidate_warnings)

    interp = result.get("scientific_interpretation") or {}
    validation = result.get("validation") or {}
    evidence["scientific_interpretation"] = all(
        isinstance(interp.get(key), str) and len(interp.get(key).strip()) >= 20
        for key in ("vacuum_sensitivity_assessment", "applicability_domain", "limitations")
    ) and validation.get("invariance_mae_mev_atom") is not None

    evidence["schema_and_consistency"] = not schema_errors and top10_ok and evidence["candidate_submission_integrity"]
    workflow_score = score_workflow(evidence)
    score_eligible = bool(
        evidence["prediction_integrity"]
        or evidence["candidate_submission_integrity"]
    )
    integrity_fields = (
        "test_labels_not_accessed",
        "no_external_network_or_label_lookup",
        "workspace_only_file_access",
    )
    checks = result.get("checks") or {}
    integrity_violation = any(checks.get(key) is False for key in integrity_fields)
    if integrity_violation:
        score_eligible = False

    return {
        "passed": len(errors) == 0 and not integrity_violation,
        "hard_gate_passed": score_eligible,
        "score_eligible": score_eligible,
        "integrity_violation": integrity_violation,
        "errors": errors,
        "warnings": warnings,
        "prediction_validation": prediction_report,
        "candidate_validation": candidate_report,
        "evidence": evidence,
        **workflow_score,
        "note": "Public errors produce compliance deductions. Property and candidate modules are scored independently; confirmed integrity violations invalidate the run.",
    }
