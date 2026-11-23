"""Artifact hashing and neutral provenance inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .file_tools import ToolInputError, WorkspaceSandbox, sha256_file
from .xyz_validator import validate_xyz


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()


def read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _manifest_entries(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, Mapping):
        return []
    raw = document.get("entries", document.get("artifacts", document.get("files", [])))
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for path, metadata in raw.items():
            if isinstance(path, str) and isinstance(metadata, Mapping):
                entries.append({"path": path, **dict(metadata)})
            elif isinstance(path, str) and isinstance(metadata, str):
                entries.append({"path": path, "sha256": metadata})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries.extend(dict(item) for item in raw if isinstance(item, Mapping))
    return entries


def _safe_manifest_relative(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("benchmark_case/"):
        normalized = normalized[len("benchmark_case/") :]
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        return None
    return pure.as_posix()


class ProvenanceInspector:
    """Check hash-bound provenance without trusting filenames."""

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        *,
        manifest_path: str = "case_manifest.json",
        expected_symbols: Sequence[str] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.manifest_path = manifest_path
        self.expected_symbols = tuple(expected_symbols) if expected_symbols else None

    def _case_entry(self, relative: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            path = self.sandbox.resolve(self.manifest_path)
        except ToolInputError:
            return None, None
        try:
            document = read_strict_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None, sha256_file(path)
        for entry in _manifest_entries(document):
            if _safe_manifest_relative(entry.get("path")) == relative:
                return entry, sha256_file(path)
        return None, sha256_file(path)

    def _case_document(self) -> Mapping[str, Any] | None:
        try:
            path = self.sandbox.resolve(self.manifest_path)
            value = read_strict_json(path)
        except (ToolInputError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None

    def _attempt_entry(self, relative: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        parts = PurePosixPath(relative).parts
        if len(parts) < 3 or parts[0] != "attempts" or not parts[1].startswith("attempt_"):
            return None, None
        attempt_root = self.sandbox.root / parts[0] / parts[1]
        manifest_path = attempt_root / "artifact_manifest.json"
        validation_path = attempt_root / "validation.json"
        if not manifest_path.is_file():
            return None, None
        try:
            manifest = read_strict_json(manifest_path)
            validation = read_strict_json(validation_path) if validation_path.is_file() else None
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None, None
        for entry in _manifest_entries(manifest):
            if _safe_manifest_relative(entry.get("path")) == relative:
                return entry, validation if isinstance(validation, dict) else None
        return None, validation if isinstance(validation, dict) else None

    def inspect(self, path: str, frame_index: int | None = None) -> dict[str, Any]:
        resolved = self.sandbox.resolve(path)
        relative = self.sandbox.relative(resolved)
        actual_hash = sha256_file(resolved)
        case_entry, case_manifest_hash = self._case_entry(relative)
        attempt_entry, validation = self._attempt_entry(relative)
        entry = attempt_entry or case_entry
        expected_hash = None
        if isinstance(entry, Mapping):
            expected_hash = entry.get("sha256", entry.get("content_sha256"))
        hash_matches = bool(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and actual_hash.casefold() == expected_hash.casefold()
        )
        source_kind = "UNMANIFESTED"
        current = False
        restart_allowed = False
        success_allowed = False
        origin = None
        if isinstance(case_entry, Mapping):
            origin = case_entry.get("origin", case_entry.get("source_kind"))
            source_kind = str(origin or "CASE_MANIFEST_ARTIFACT")
            declared_current = case_entry.get("current", case_entry.get("is_current", False)) is True
            current = bool(hash_matches and declared_current)
            restart_allowed = bool(
                hash_matches
                and case_entry.get("restart_allowed", case_entry.get("allow_restart", False)) is True
            )
            success_allowed = bool(
                current
                and case_entry.get(
                    "success_evidence_allowed", case_entry.get("allow_success_evidence", False)
                )
                is True
            )
            case_document = self._case_document()
            initial = case_document.get("initial_failure") if isinstance(case_document, Mapping) else None
            initial_id = initial.get("attempt_id") if isinstance(initial, Mapping) else None
            initial_input = (
                f"attempts/{initial_id}/input.xyz" if isinstance(initial_id, str) else None
            )
            initial_geometry = (
                f"attempts/{initial_id}/xtbopt.xyz" if isinstance(initial_id, str) else None
            )
            initial_trajectory = (
                f"attempts/{initial_id}/xtbopt.log" if isinstance(initial_id, str) else None
            )
            entries = _manifest_entries(case_document)
            initial_input_entry = next(
                (
                    item
                    for item in entries
                    if _safe_manifest_relative(item.get("path")) == initial_input
                ),
                None,
            )
            initial_input_hash = (
                initial_input_entry.get("sha256")
                if isinstance(initial_input_entry, Mapping)
                else None
            )
            if hash_matches and (
                relative == initial_input
                or (
                    relative.startswith("input/")
                    and isinstance(initial_input_hash, str)
                    and actual_hash.casefold() == initial_input_hash.casefold()
                )
            ):
                source_kind = "ORIGINAL_INPUT"
                origin = initial_input
                current = True
                restart_allowed = True
                success_allowed = False
            elif hash_matches and relative in {initial_geometry, initial_trajectory}:
                source_kind = "CURRENT_TRAJECTORY_FRAME"
                origin = initial_id
                current = True
                restart_allowed = bool(
                    isinstance(initial, Mapping)
                    and initial.get("verified_not_converged") is True
                )
                success_allowed = False
            elif relative.startswith("legacy/"):
                legacy = case_document.get("legacy") if isinstance(case_document, Mapping) else None
                source_kind = "LEGACY_ARTIFACT"
                origin = legacy.get("provenance_state") if isinstance(legacy, Mapping) else None
                current = False
                restart_allowed = bool(
                    hash_matches
                    and isinstance(legacy, Mapping)
                    and legacy.get("allowed_as_restart_source") is True
                )
                success_allowed = bool(
                    hash_matches
                    and isinstance(legacy, Mapping)
                    and legacy.get("allowed_as_current_success") is True
                )
        if isinstance(attempt_entry, Mapping):
            role = attempt_entry.get("role")
            source_kind = (
                "VALIDATED_ATTEMPT_ARTIFACT"
                if isinstance(validation, Mapping)
                else "ATTEMPT_ARTIFACT_UNVALIDATED"
            )
            parts = PurePosixPath(relative).parts
            attempt_id = parts[1] if len(parts) >= 2 else None
            origin = f"{attempt_id}:{role}"
            current = hash_matches
            restart_allowed = False
            success_allowed = False
        if isinstance(attempt_entry, Mapping) and isinstance(validation, Mapping):
            role = attempt_entry.get("role")
            validation_passed = validation.get("validation_passed") is True
            artifacts_current = validation.get("artifacts_current") is True
            current = bool(hash_matches and artifacts_current)
            restart_sources = validation.get("restart_sources")
            authorized_restart_paths = (
                {item.get("path") for item in restart_sources if isinstance(item, Mapping)}
                if isinstance(restart_sources, list)
                else set()
            )
            restart_allowed = bool(
                current
                and (
                    (validation_passed and role == "optimized_geometry")
                    or relative in authorized_restart_paths
                )
            )
            success_allowed = bool(
                current
                and validation_passed
                and role in {"optimized_geometry", "hessian", "vibrational_spectrum"}
            )
            origin = f"{validation.get('attempt_id')}:{role}"
        xyz_result: dict[str, Any] | None = None
        if (
            resolved.suffix.casefold() == ".xyz"
            or source_kind == "CURRENT_TRAJECTORY_FRAME"
            or (
                source_kind == "VALIDATED_ATTEMPT_ARTIFACT"
                and str(origin or "").rsplit(":", 1)[-1]
                in {"trajectory", "restart_geometry", "optimized_geometry"}
            )
        ):
            xyz_result = validate_xyz(
                self.sandbox,
                relative,
                expected_symbols=self.expected_symbols,
                frame_index=frame_index,
            )
            xyz_compatible = bool(
                xyz_result.get("valid") is True
                or (
                    xyz_result.get("safely_recoverable") is True
                    and xyz_result.get("selected_frame_is_complete") is True
                )
            )
            if not xyz_compatible:
                restart_allowed = False
                success_allowed = False
            elif xyz_result.get("valid") is not True:
                # A complete frame preceding a truncated tail can be restart
                # input, but the incomplete document is never success evidence.
                success_allowed = False
        return {
            "path": relative,
            "frame_index": frame_index,
            "size_bytes": resolved.stat().st_size,
            "sha256": actual_hash,
            "manifest_entry_present": entry is not None,
            "manifest_sha256": case_manifest_hash,
            "declared_sha256": expected_hash,
            "hash_matches_manifest": hash_matches,
            "source_kind": source_kind,
            "origin": origin,
            "current": current,
            "compatible": bool(
                xyz_result is None
                or xyz_result.get("valid") is True
                or xyz_result.get("safely_recoverable") is True
            ),
            "restart_allowed": restart_allowed,
            "success_evidence_allowed": success_allowed,
            "xyz_validation": xyz_result,
            "neutrality_note": "currentness and permissions require hash-bound manifest evidence",
        }


def build_artifact_manifest(
    sandbox: WorkspaceSandbox,
    attempt_id: str,
    roles: Mapping[str, str],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative, role in sorted(roles.items()):
        resolved = sandbox.resolve(relative)
        entries.append(
            {
                "path": sandbox.relative(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
                "role": role,
            }
        )
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "entries": entries,
        "created_at": utc_now(),
    }


__all__ = [
    "ProvenanceInspector",
    "build_artifact_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "read_strict_json",
    "utc_now",
    "write_json_exclusive",
]
