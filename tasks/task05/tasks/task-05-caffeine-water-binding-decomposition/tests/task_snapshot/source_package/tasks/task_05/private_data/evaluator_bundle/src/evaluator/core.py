"""Trusted-evidence loading and deterministic score accumulation."""

from __future__ import annotations

import csv
import json
import hashlib
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import yaml

from common.io_utils import load_yaml, native_path, safe_resolve, sha256_file
from tools.xtb_parse import parse_xtb_output


class EvaluationError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    """Replace non-finite diagnostic numbers so explain reports are strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class TrustedJob:
    job_id: str
    record: dict[str, Any]
    workdir: Path
    parsed: dict[str, Any]


class ScoreBuilder:
    """Accumulate item-level scores from the frozen ``scoring.yaml``."""

    def __init__(self, scoring_config: dict[str, Any]) -> None:
        self.config = scoring_config
        self.items: dict[str, dict[str, Any]] = {}
        self.specs: dict[str, dict[str, Any]] = {}
        self.sections: dict[str, dict[str, Any]] = {}
        self.hard_gates: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.total_zero = False
        for section_id, section in (scoring_config.get("sections") or {}).items():
            if not isinstance(section, dict):
                continue
            self.sections[str(section_id)] = section
            for item in section.get("items", []) or []:
                if isinstance(item, dict) and item.get("id"):
                    spec = dict(item)
                    spec["section"] = str(section_id)
                    self.specs[str(item["id"])] = spec

    def _spec(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.specs:
            raise EvaluationError(f"unknown scoring item: {item_id}")
        return self.specs[item_id]

    def add(self, item_id: str, *, fraction: float, evidence: Any, reason: str) -> None:
        spec = self._spec(item_id)
        bounded = max(0.0, min(1.0, float(fraction))) if math.isfinite(float(fraction)) else 0.0
        maximum = float(spec.get("points", 0.0))
        self.items[item_id] = {
            "item_id": item_id,
            "section": spec["section"],
            "rule": spec.get("rule"),
            "earned_points": maximum * bounded,
            "max_points": maximum,
            "fraction": bounded,
            "evidence": evidence,
            "reason": reason,
        }

    def exact(self, item_id: str, condition: Any, *, evidence: Any, reason: str) -> None:
        self.add(item_id, fraction=1.0 if bool(condition) else 0.0, evidence=evidence, reason=reason)

    def direct(self, item_id: str, fraction: float, *, evidence: Any, reason: str) -> None:
        self.add(item_id, fraction=fraction, evidence=evidence, reason=reason)

    def linear(self, item_id: str, error: float, *, evidence: Any, reason: str) -> None:
        spec = self._spec(item_id)
        full = float(spec.get("full_credit_error", 0.0))
        zero = float(spec.get("zero_credit_error", full))
        value = float(error)
        if not math.isfinite(value):
            fraction = 0.0
        elif value <= full:
            fraction = 1.0
        elif zero <= full or value >= zero:
            fraction = 0.0
        else:
            fraction = (zero - value) / (zero - full)
        self.add(item_id, fraction=fraction, evidence=evidence, reason=reason)

    def lower_bound(self, item_id: str, value: float, *, evidence: Any, reason: str) -> None:
        spec = self._spec(item_id)
        full = float(spec.get("full_credit_value", 1.0))
        zero = float(spec.get("zero_credit_value", 0.0))
        numeric = float(value)
        if not math.isfinite(numeric):
            fraction = 0.0
        elif numeric >= full:
            fraction = 1.0
        elif full <= zero or numeric <= zero:
            fraction = 0.0
        else:
            fraction = (numeric - zero) / (full - zero)
        self.add(item_id, fraction=fraction, evidence=evidence, reason=reason)

    def hard_gate(self, gate_id: str, *, reason: str, evidence: Any = None, total_zero: bool = False, invalidates: Sequence[str] = ()) -> None:
        unknown = sorted(set(map(str, invalidates)) - set(self.specs))
        if unknown:
            raise EvaluationError(f"hard gate {gate_id} names unknown scoring items: {unknown}")
        self.hard_gates.append({"gate_id": gate_id, "reason": reason, "evidence": evidence, "scope": "total" if total_zero else ("dependent_items" if invalidates else "report_only"), "invalidated_item_ids": list(map(str, invalidates)), "total_zero": total_zero})
        self.total_zero |= bool(total_zero)

    def finalize(self, task_id: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        for item_id, spec in self.specs.items():
            if item_id not in self.items:
                self.add(item_id, fraction=0.0, evidence=None, reason="scoring item was not satisfied or could not be evaluated")
        for gate in self.hard_gates:
            for item_id in gate.get("invalidated_item_ids", []):
                item = self.items[item_id]
                item["earned_points"] = 0.0
                item["fraction"] = 0.0
                item["reason"] = f"hard gate {gate['gate_id']}: {item['reason']}"
        if self.total_zero:
            for item in self.items.values():
                item["earned_points"] = 0.0
                item["fraction"] = 0.0
                item["reason"] = f"total-score hard gate: {item['reason']}"
        section_reports: list[dict[str, Any]] = []
        for section_id, section in self.sections.items():
            entries = [self.items[item_id] for item_id, spec in self.specs.items() if spec["section"] == section_id]
            section_reports.append(
                {
                    "section_id": section_id,
                    "earned_points": round(sum(float(item["earned_points"]) for item in entries), 6),
                    "max_points": round(sum(float(item["max_points"]) for item in entries), 6),
                    "items": entries,
                }
            )
        total = sum(float(item["earned_points"]) for item in self.items.values())
        maximum = sum(float(item["max_points"]) for item in self.items.values())
        configured = float(self.config.get("total_points", maximum))
        if abs(maximum - configured) > 1e-6:
            self.notes.append(f"scoring item weights sum to {maximum}, configured total is {configured}")
        return _json_safe({
            "schema_version": "1.0",
            "task_id": task_id,
            "total_score": round(total, 2),
            "max_score": round(configured, 2),
            "hard_gates": self.hard_gates,
            "sections": section_reports,
            "notes": self.notes,
            "metadata": metadata or {},
        })


def load_json_required(path: str | Path) -> dict[str, Any]:
    target = native_path(path)
    if not target.is_file():
        raise EvaluationError(f"required JSON file is missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root must be an object: {target}")
    return value


def load_json_optional(path: str | Path) -> dict[str, Any]:
    """Load an object when present; return empty for missing/malformed submissions."""
    target = native_path(path)
    if not target.is_file():
        return {}
    try:
        return load_json_required(target)
    except EvaluationError:
        return {}


def load_json_value_optional(path: str | Path) -> Any:
    """Load any JSON root type for contracts that explicitly allow arrays.

    Most auxiliary outputs require an object and therefore continue to use
    :func:`load_json_optional`. Recovery logs are the deliberate exception:
    both a top-level event array and ``{"entries": [...]}`` are public,
    equivalent representations.
    """
    target = native_path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return {}


def load_yaml_optional(path: str | Path) -> dict[str, Any]:
    """Load an optional submission YAML object without aborting the scorer.

    Malformed model-authored auxiliary files are local scoring failures.  They
    must not be promoted to an evaluator exception/whole-task zero; downstream
    protocol and consistency checks will receive an empty object and zero only
    the sections that depend on it.
    """
    try:
        value = load_yaml(path, default={}) or {}
    except (EvaluationError, yaml.YAMLError, OSError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = native_path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_final_answer(task_dir: Path, final_answer: dict[str, Any]) -> list[str]:
    schema_path = native_path(task_dir / "final_answer.schema.json")
    if not schema_path.is_file():
        return ["final_answer.schema.json is missing"]
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
    except Exception as exc:
        return [f"invalid final-answer schema: {exc}"]
    return [f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}" for error in sorted(validator.iter_errors(final_answer), key=lambda e: list(e.absolute_path))]


def validate_json_against_schema(payload: dict[str, Any], schema_path: str | Path) -> list[str]:
    """Validate an auxiliary structured output without promoting failure to a task-wide gate."""
    target = Path(schema_path)
    io_target = native_path(target)
    if not io_target.is_file():
        return [f"schema is missing: {target.name}"]
    try:
        schema = json.loads(io_target.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
    except Exception as exc:
        return [f"invalid schema {target.name}: {exc}"]
    return [
        f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    ]


def reference_manifest(reference_dir: str | Path) -> dict[str, Any]:
    path = Path(reference_dir)
    if path.is_file():
        target = path
    else:
        target = path / "reference_manifest.json"
    return load_json_required(target)


def normalize_recovery_log(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("entries", "recoveries", "recovery_events", "attempts"):
            entries = value.get(key)
            if isinstance(entries, list):
                return [item for item in entries if isinstance(item, dict)]
        return [value] if value else []
    return []


def verify_and_parse_jobs(submission_dir: str | Path) -> tuple[dict[str, TrustedJob], list[str], dict[str, Any]]:
    root = Path(os.path.realpath(os.path.abspath(os.fspath(submission_dir))))
    manifest_path = native_path(root / "run_manifest.json")
    if not manifest_path.is_file():
        return {}, ["run_manifest.json is missing"], {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"run_manifest.json is invalid: {exc}"], {}
    if not isinstance(manifest, dict):
        return {}, ["run_manifest.json root is not an object"], {}
    jobs: dict[str, TrustedJob] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for raw in manifest.get("jobs", []) if isinstance(manifest.get("jobs"), list) else []:
        if not isinstance(raw, dict):
            errors.append("manifest contains a non-object job record")
            continue
        job_id = str(raw.get("job_id") or "")
        if not job_id or job_id in seen:
            errors.append(f"missing or duplicate job_id: {job_id!r}")
            continue
        seen.add(job_id)
        workdir_value = raw.get("workdir")
        if not isinstance(workdir_value, str):
            errors.append(f"{job_id}: workdir is missing")
            continue
        try:
            logical_workdir = safe_resolve(root, workdir_value, must_exist=True)
            workdir = native_path(logical_workdir)
        except Exception as exc:
            errors.append(f"{job_id}: unsafe/missing workdir: {exc}")
            continue
        if not workdir.is_dir():
            errors.append(f"{job_id}: workdir is not a directory")
            continue
        hashes = raw.get("files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            errors.append(f"{job_id}: files_sha256 is missing")
            continue
        # Spec 5.1: normalize registered relative POSIX paths and reject unsafe forms.
        normalized: dict[str, str] = {}
        key_ok = True
        for relative, expected in hashes.items():
            rel = str(relative)
            if not rel or "\x00" in rel:
                errors.append(f"{job_id}: invalid registered path {rel!r}")
                key_ok = False
                continue
            if "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
                errors.append(f"{job_id}: unsafe registered path {rel!r}")
                key_ok = False
                continue
            parts = [part for part in rel.split("/") if part not in ("", ".")]
            if any(part == ".." for part in parts):
                errors.append(f"{job_id}: parent-directory traversal in {rel!r}")
                key_ok = False
                continue
            norm = "/".join(parts)
            if not norm:
                errors.append(f"{job_id}: empty normalized path {rel!r}")
                key_ok = False
                continue
            if norm in normalized:
                errors.append(f"{job_id}: duplicate normalized path {norm!r}")
                key_ok = False
                continue
            normalized[norm] = str(expected)
        if not key_ok:
            continue
        fold_seen: dict[str, str] = {}
        fold_ok = True
        for key in list(normalized):
            cf = key.casefold()
            if cf in fold_seen and fold_seen[cf] != key:
                errors.append(f"{job_id}: casefold conflict {fold_seen[cf]!r} vs {key!r}")
                fold_ok = False
            fold_seen.setdefault(cf, key)
            nf = unicodedata.normalize("NFC", key)
            if nf != key:
                errors.append(f"{job_id}: unicode normalization conflict for {key!r}")
                fold_ok = False
        if not fold_ok:
            continue
        # Enumerate all ordinary files; the registered set must be exactly equal.
        actual_files: dict[str, str] = {}
        enum_ok = True
        for path in sorted(workdir.rglob("*")):
            if path.is_symlink():
                errors.append(f"{job_id}: symlink is not allowed: {path.name}")
                enum_ok = False
                break
            if path.is_file():
                actual_files[path.relative_to(workdir).as_posix()] = sha256_file(path)
        if not enum_ok:
            continue
        norm_keys = set(normalized)
        actual_keys = set(actual_files)
        missing = sorted(actual_keys - norm_keys)
        extra = sorted(norm_keys - actual_keys)
        if missing:
            errors.append(f"{job_id}: files not registered in files_sha256: {missing}")
        if extra:
            errors.append(f"{job_id}: files_sha256 lists non-existent files: {extra}")
        if missing or extra:
            continue
        mismatches = [rel for rel in sorted(norm_keys) if actual_files[rel] != normalized[rel]]
        if mismatches:
            errors.append(f"{job_id}: SHA-256 mismatch for {mismatches}")
            continue
        required = ("request.json", "command.json", "environment.json", "input.xyz", "stdout.txt", "stderr.txt")
        missing_required = sorted({name for name in required if name not in norm_keys})
        if missing_required:
            errors.append(f"{job_id}: missing required files: {missing_required}")
            continue
        # Recompute the frozen evidence tree (path+NUL+sha256+LF, sorted).
        registered_evidence = raw.get("evidence_tree_sha256")
        if not isinstance(registered_evidence, str) or not registered_evidence:
            errors.append(f"{job_id}: evidence_tree_sha256 is missing")
            continue
        digest = hashlib.sha256()
        for rel in sorted(normalized):
            digest.update(rel.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(normalized[rel].encode("utf-8"))
            digest.update(b"\n")
        if registered_evidence != digest.hexdigest():
            errors.append(f"{job_id}: evidence_tree_sha256 mismatch")
            continue
        try:
            parsed = parse_xtb_output(workdir)
        except Exception as exc:
            errors.append(f"{job_id}: parser failed: {type(exc).__name__}: {exc}")
            continue
        jobs[job_id] = TrustedJob(job_id=job_id, record=dict(raw), workdir=workdir, parsed=parsed)
    return jobs, errors, manifest


def job_valid_termination(job: TrustedJob | None) -> bool:
    return bool(job and job.record.get("status") == "completed" and int(job.record.get("exit_code", 1)) == 0 and (job.parsed.get("status") or {}).get("normal_termination"))


def job_no_significant_imaginary(job: TrustedJob | None) -> bool:
    """Return true only for a *complete, actually executed* Hessian.

    An empty frequency list is missing evidence, not evidence for a minimum.  The
    parser v2 supplies the explicit flags below; the count check is retained as
    a defensive independent validation of those flags.
    """
    vibrations = (job.parsed.get("vibrations") or {}) if job else {}
    frequencies = vibrations.get("frequencies_cm1")
    imaginary = vibrations.get("significant_imaginary_frequencies_cm1")
    performed = vibrations.get("hessian_performed") is True
    complete = vibrations.get("complete") is True
    expected = vibrations.get("expected_mode_count")
    parsed = vibrations.get("parsed_mode_count")
    count_ok = (
        isinstance(expected, int)
        and expected > 0
        and isinstance(parsed, int)
        and parsed >= expected
        and isinstance(frequencies, list)
        and len(frequencies) >= expected
    )
    return bool(
        job
        and job_valid_termination(job)
        and performed
        and complete
        and count_ok
        and isinstance(imaginary, list)
        and len(imaginary) == 0
    )


def job_energy(job: TrustedJob | None, key: str) -> float | None:
    if not job or not job_valid_termination(job):
        return None
    value = (job.parsed.get("energies") or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def manifest_execution_identity_audit(
    scientific_manifest: Mapping[str, Any],
    harness_manifest: Mapping[str, Any] | None,
    *,
    accepted_task_ids: set[str],
    expected_task_version: str,
) -> tuple[bool, dict[str, Any]]:
    """Validate immutable execution identity, using Harness checks when available."""
    environment = scientific_manifest.get("environment") if isinstance(scientific_manifest.get("environment"), Mapping) else {}
    keys = ("environment_lock_sha256", "task_config_sha256", "public_inputs_sha256", "xtb_executable_sha256")
    submitted = {key: scientific_manifest.get(key) for key in keys}
    sha_format_ok = all(
        isinstance(submitted.get(key), str)
        and len(str(submitted[key])) == 64
        and all(character in "0123456789abcdef" for character in str(submitted[key]))
        for key in keys
    )
    environment_match = all(environment.get(key) == submitted.get(key) for key in keys)
    runtime = harness_manifest.get("runtime_checks") if isinstance(harness_manifest, Mapping) else None
    expected: dict[str, Any] = {}
    if isinstance(runtime, Mapping):
        binaries = runtime.get("binaries") if isinstance(runtime.get("binaries"), list) else []
        xtb = next((item for item in binaries if isinstance(item, Mapping) and item.get("name") == "xtb"), {})
        hashes = runtime.get("hashes") if isinstance(runtime.get("hashes"), Mapping) else {}
        lock = runtime.get("environment_lock") if isinstance(runtime.get("environment_lock"), Mapping) else {}
        expected = {"environment_lock_sha256": lock.get("actual_sha256"), "task_config_sha256": hashes.get("task_config_sha256"), "public_inputs_sha256": hashes.get("public_inputs_sha256"), "xtb_executable_sha256": xtb.get("sha256")}
    runtime_match = not expected or all(expected.get(key) == submitted.get(key) for key in keys)
    harness_identity_match = not isinstance(harness_manifest, Mapping) or all(
        harness_manifest.get(key) == submitted.get(key) for key in keys
    )
    schema_identity_ok = bool(
        scientific_manifest.get("schema_name") == "xtb_submission_manifest_v2"
        and scientific_manifest.get("schema_version") == "2.1.0"
        and scientific_manifest.get("manifest_role") == "scientific_submission"
    )
    identity_ok = bool(
        schema_identity_ok
        and scientific_manifest.get("task_id") in accepted_task_ids
        and str(scientific_manifest.get("task_version") or "") == str(expected_task_version)
        and sha_format_ok
        and environment_match
        and runtime_match
        and harness_identity_match
    )
    return identity_ok, {
        "submitted_hashes": submitted,
        "submitted_environment": dict(environment),
        "trusted_runtime_environment": expected,
        "schema_identity_ok": schema_identity_ok,
        "sha256_format_ok": sha_format_ok,
        "environment_match": environment_match,
        "runtime_match": runtime_match,
        "harness_identity_match": harness_identity_match,
    }


def required_output_status(task_config: Mapping[str, Any], submission_dir: str | Path) -> dict[str, bool]:
    """Check the frozen submission contract without trusting declared content."""
    root = native_path(submission_dir)
    result: dict[str, bool] = {}
    for relative in task_config.get("required_outputs", []) or []:
        name = str(relative)
        target = root / name.rstrip("/\\")
        if name.endswith(("/", "\\")):
            result[name] = target.is_dir() and any(path.is_file() for path in target.rglob("*"))
        else:
            result[name] = target.is_file()
    return result


def is_deliberate_protocol_failure(job: TrustedJob | None) -> bool:
    """Identify invalid-CLI/wrong-method attempts that cannot earn recovery credit."""
    if not job:
        return False
    failure = job.parsed.get("failure") or {}
    category = str(failure.get("category") or "").lower()
    evidence = " ".join(map(str, failure.get("evidence_lines") or [])).lower()
    request = job.record.get("request") or {}
    args = " ".join(map(str, request.get("arguments") or request.get("args") or [])).lower()
    return bool(
        category in {"invalid_argument", "cli_error", "unsupported_method", "protocol_mismatch"}
        or re.search(r"unknown|invalid|unrecognized|unsupported\s+(?:option|argument|method)", evidence)
        or re.search(r"--gfn\s+(?:0|1|ff)\b", args)
    )


def protocol_matches(
    job: TrustedJob | None,
    *,
    version: str,
    gfn: int,
    solvent_model: str,
    solvent: str | None,
    charge: int,
    uhf: int,
) -> bool:
    if not job:
        return False
    protocol = job.parsed.get("protocol") or {}
    model = str(protocol.get("solvent_model") or "none").lower()
    actual_solvent = protocol.get("solvent")
    return bool(
        str(protocol.get("version") or "") == str(version)
        and protocol.get("gfn") == int(gfn)
        and model == str(solvent_model).lower()
        and ((solvent is None and actual_solvent in (None, "", "none")) or str(actual_solvent).lower() == str(solvent).lower())
        and protocol.get("charge") == int(charge)
        and protocol.get("uhf") == int(uhf)
    )


def mean_absolute_error(reference: Mapping[str, float], candidate: Mapping[str, float]) -> float:
    if not reference or set(reference) != set(candidate):
        return math.inf
    return sum(abs(float(reference[key]) - float(candidate[key])) for key in reference) / len(reference)


def relative_mae(reference: Mapping[str, float], candidate: Mapping[str, float], scale: float = 1.0) -> float:
    if not reference or set(reference) != set(candidate):
        return math.inf
    ref_min = min(float(value) for value in reference.values())
    cand_min = min(float(value) for value in candidate.values())
    return sum(abs((float(reference[key]) - ref_min) - (float(candidate[key]) - cand_min)) * scale for key in reference) / len(reference)


def text_contains_affirmative_claim(path: str | Path, *, patterns: Sequence[str], negations: Sequence[str] = ()) -> bool:
    target = native_path(path)
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    for negation in negations:
        lowered = lowered.replace(str(negation).lower(), " ")
    return any(re.search(pattern, lowered, re.I) for pattern in patterns)
