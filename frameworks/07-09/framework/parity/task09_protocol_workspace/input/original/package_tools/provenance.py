"""Deterministic, package-safe projection of native xTB attempt provenance.

Raw native attempts are host-local audit evidence.  They may contain an
absolute executable path in ``result.json`` and ``stdout.log`` and therefore
must never be copied directly into a release.  This module verifies the
immutable attempt manifest, accepts only the frozen native-result schema, and
emits a closed projection that preserves hashes and scientific facts while
replacing host locations with fixed role tokens.

The module is release-layer code.  It is intentionally independent of the
agent runner and evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping


PROJECTION_SCHEMA_VERSION = "package-safe-xtb-provenance/2.0"
PROJECTION_KIND = "native_xtb_attempt"
XTB_EXECUTABLE_TOKEN = "<XTB_EXECUTABLE>"
ATTEMPT_WORKDIR_TOKEN = "<ATTEMPT_WORKDIR>"
EXPECTED_XTB_VERSION = "6.7.1"
EXPECTED_XTB_SHA256 = (
    "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
)

SHA256_RE = re.compile(r"^[0-9A-F]{64}$")
WINDOWS_DRIVE_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>|?*\x00]+"
)
WINDOWS_DRIVE_RELATIVE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:(?![\\/])[^:\s'\"<>|?*\x00]+"
)
WINDOWS_UNC_TEXT_RE = re.compile(
    r"(?<![\\])(?:\\\\){2,}(?:[?.](?:\\\\)+)?[^\\/\s'\"<>|?*\x00]+"
)
POSIX_HOST_ABSOLUTE_RE = re.compile(
    r"(?<![:A-Za-z0-9_])/(?:home|Users|private|tmp|var/tmp|workspace|mnt)/"
    r"[^\s'\"<>|\x00]+"
)
FILE_URI_RE = re.compile(r"(?i)\bfile:(?://|\\\\)[^\s'\"<>\x00]+")

RAW_RESULT_REQUIRED_FIELDS = frozenset(
    {
        "angle_deg",
        "angle_error_deg",
        "argv",
        "charge",
        "constraint",
        "converged",
        "duration_seconds",
        "environment",
        "final_angle_deg",
        "method",
        "optimization",
        "parse_error",
        "phase",
        "prepared_input",
        "profile_eligible",
        "reference_sha256",
        "requested_final_angle_tolerance_deg",
        "return_code",
        "schema_version",
        "shell",
        "structure_validation",
        "threads",
        "timed_out",
        "timeout_seconds",
        "total_energy_hartree",
        "uhf",
        "xtb",
    }
)
CONSTRAINT_FIELDS = frozenset(
    {
        "atom_indices_one_based",
        "atom_maps",
        "force_constant_au",
        "target_deg",
        "type",
    }
)
PREPARED_INPUT_FIELDS = frozenset(
    {
        "angle_deg",
        "input_sha256",
        "reference_sha256",
        "uff_angle_error_deg",
        "uff_final_angle_deg",
    }
)
XTB_FIELDS = frozenset(
    {"binary_sha256", "executable", "version", "version_exit_code"}
)
ENVIRONMENT_FIELDS = frozenset(
    {"MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
)
STRUCTURE_REQUIRED_FIELDS = frozenset(
    {
        "atom_count_ok",
        "atom_order_ok",
        "composition_ok",
        "connectivity_ok",
        "formal_charge_ok",
    }
)
STRUCTURE_OPTIONAL_FIELDS = frozenset(
    {
        "expected_bond_count",
        "inferred_bond_count",
        "missing_bonds",
        "unexpected_bonds",
    }
)
FROZEN_ARGV_TAIL = (
    "input.xyz",
    "--gfn",
    "2",
    "--chrg",
    "0",
    "--uhf",
    "0",
    "--opt",
    "tight",
    "--input",
    "xcontrol.inp",
)


class ProvenanceProjectionError(ValueError):
    """Raw provenance or a package-safe projection failed closed validation."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the benchmark canonical JSON representation."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceProjectionError("NON_CANONICAL_JSON_VALUE") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _is_link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or (callable(is_junction) and is_junction())
    )


def _load_json_bytes(data: bytes, error: str) -> Any:
    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON constant")

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProvenanceProjectionError(error) from exc


def _require_mapping(value: Any, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceProjectionError(error)
    return value


def _require_closed_fields(
    value: Any,
    required: frozenset[str],
    *,
    optional: frozenset[str] = frozenset(),
    error: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, error)
    keys = set(mapping)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        missing = sorted(required - keys)
        unknown = sorted(keys - required - optional)
        suffix = f":missing={missing}:unknown={unknown}"
        raise ProvenanceProjectionError(error + suffix)
    return mapping


def _require_sha256(value: Any, error: str) -> str:
    if not isinstance(value, str):
        raise ProvenanceProjectionError(error)
    normalized = value.upper()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ProvenanceProjectionError(error)
    return normalized


def _require_bool(value: Any, error: str) -> bool:
    if not isinstance(value, bool):
        raise ProvenanceProjectionError(error)
    return value


def _require_int(value: Any, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProvenanceProjectionError(error)
    return value


def _require_finite_number(
    value: Any,
    error: str,
    *,
    allow_none: bool = False,
) -> float | int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceProjectionError(error)
    if not math.isfinite(float(value)):
        raise ProvenanceProjectionError(error)
    return value


def _safe_flat_name(value: Any, error: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceProjectionError(error)
    pure = PurePosixPath(value)
    if (
        "\\" in value
        or len(pure.parts) != 1
        or pure.name in {"", ".", ".."}
        or ":" in pure.name
        or "\x00" in pure.name
        or pure.as_posix() != value
    ):
        raise ProvenanceProjectionError(error)
    return value


def _path_rule(value: str) -> str | None:
    if value in {XTB_EXECUTABLE_TOKEN, ATTEMPT_WORKDIR_TOKEN}:
        return None
    if "\x00" in value:
        return "nul"
    if FILE_URI_RE.search(value):
        return "file_uri"
    if WINDOWS_DRIVE_ABSOLUTE_RE.search(value):
        return "windows_drive_absolute"
    if WINDOWS_DRIVE_RELATIVE_RE.search(value):
        return "windows_drive_relative"
    if value.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        return "windows_unc_or_device"
    if POSIX_HOST_ABSOLUTE_RE.search(value):
        return "posix_host_absolute"
    if value.startswith("/") and not value.startswith("//"):
        return "posix_absolute"
    return None


def _walk_strings(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str):
                raise ProvenanceProjectionError("NON_STRING_JSON_KEY")
            yield from _walk_strings(value[key], f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{location}[{index}]")


def projection_host_path_findings(value: Any) -> list[dict[str, str]]:
    """Return path rules and JSON locations without returning sensitive values."""

    findings: list[dict[str, str]] = []
    for location, string in _walk_strings(value):
        rule = _path_rule(string)
        if rule is not None:
            findings.append({"location": location, "rule": rule})
    return findings


def _text_path_matches(content: str) -> Iterable[tuple[int, str]]:
    patterns = (
        ("windows_drive_absolute", WINDOWS_DRIVE_ABSOLUTE_RE),
        ("windows_drive_relative", WINDOWS_DRIVE_RELATIVE_RE),
        ("windows_unc_or_device", WINDOWS_UNC_TEXT_RE),
        ("posix_host_absolute", POSIX_HOST_ABSOLUTE_RE),
        ("file_uri", FILE_URI_RE),
    )
    occupied: set[tuple[int, int]] = set()
    for rule, pattern in patterns:
        for match in pattern.finditer(content):
            span = match.span()
            if span in occupied:
                continue
            occupied.add(span)
            yield content.count("\n", 0, match.start()) + 1, rule


def find_host_path_occurrences(attempt_root: Path) -> list[dict[str, Any]]:
    """Scan a raw attempt without returning the matched host path itself."""

    resolved = attempt_root.resolve(strict=True)
    if not resolved.is_dir() or _is_link_or_reparse(resolved):
        raise ProvenanceProjectionError("RAW_ATTEMPT_ROOT_UNSAFE")
    findings: list[dict[str, Any]] = []
    for path in sorted(resolved.iterdir(), key=lambda item: item.name):
        if not path.is_file() or _is_link_or_reparse(path):
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line, rule in _text_path_matches(content):
            findings.append({"path": path.name, "line": line, "rule": rule})
    return findings


def _tree_commitment(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda entry: str(entry["path"])):
        digest.update(
            (
                f"{item['path']}\0{item['size_bytes']}\0"
                f"{item['sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest().upper()


def _inspect_attempt(attempt_root: Path) -> dict[str, Any]:
    resolved = attempt_root.resolve(strict=True)
    if not resolved.is_dir() or _is_link_or_reparse(resolved):
        raise ProvenanceProjectionError("RAW_ATTEMPT_ROOT_UNSAFE")
    children = sorted(resolved.iterdir(), key=lambda item: item.name)
    for child in children:
        if _is_link_or_reparse(child):
            raise ProvenanceProjectionError("RAW_ATTEMPT_LINK_OR_REPARSE")
        if not child.is_file():
            raise ProvenanceProjectionError("RAW_ATTEMPT_NOT_FLAT")
    manifest_path = resolved / "attempt_manifest.json"
    if not manifest_path.is_file():
        raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_MISSING")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _require_closed_fields(
        _load_json_bytes(manifest_bytes, "RAW_ATTEMPT_MANIFEST_INVALID"),
        frozenset({"algorithm", "files", "schema_version"}),
        error="RAW_ATTEMPT_MANIFEST_SCHEMA",
    )
    if manifest["schema_version"] != "1.0" or manifest["algorithm"] != "SHA-256":
        raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_IDENTITY")
    raw_items = manifest["files"]
    if not isinstance(raw_items, list):
        raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_FILES")
    listed: dict[str, dict[str, Any]] = {}
    casefolded: set[str] = set()
    for raw_item in raw_items:
        item = _require_closed_fields(
            raw_item,
            frozenset({"path", "sha256", "size_bytes"}),
            error="RAW_ATTEMPT_MANIFEST_ENTRY",
        )
        name = _safe_flat_name(item["path"], "RAW_ATTEMPT_MANIFEST_PATH")
        if name == "attempt_manifest.json":
            raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_SELF_LISTED")
        folded = name.casefold()
        if name in listed or folded in casefolded:
            raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_DUPLICATE")
        size = _require_int(item["size_bytes"], "RAW_ATTEMPT_MANIFEST_SIZE")
        if size < 0:
            raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_SIZE")
        listed[name] = {
            "path": name,
            "size_bytes": size,
            "sha256": _require_sha256(
                item["sha256"], "RAW_ATTEMPT_MANIFEST_HASH"
            ),
        }
        casefolded.add(folded)
    actual_names = {
        path.name for path in children if path.name != "attempt_manifest.json"
    }
    if set(listed) != actual_names:
        raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_INVENTORY_MISMATCH")
    for name, item in listed.items():
        path = resolved / name
        if path.stat().st_size != item["size_bytes"]:
            raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_SIZE_MISMATCH")
        if sha256_file(path) != item["sha256"]:
            raise ProvenanceProjectionError("RAW_ATTEMPT_MANIFEST_HASH_MISMATCH")
    manifest_entry = {
        "path": "attempt_manifest.json",
        "size_bytes": len(manifest_bytes),
        "sha256": sha256_bytes(manifest_bytes),
    }
    all_entries = sorted(
        [*listed.values(), manifest_entry], key=lambda item: str(item["path"])
    )
    if "result.json" not in listed:
        raise ProvenanceProjectionError("RAW_ATTEMPT_RESULT_MISSING")
    result_bytes = (resolved / "result.json").read_bytes()
    result = _load_json_bytes(result_bytes, "RAW_ATTEMPT_RESULT_INVALID")
    return {
        "root": resolved,
        "listed": listed,
        "files": all_entries,
        "file_count": len(all_entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in all_entries),
        "manifest_sha256": manifest_entry["sha256"],
        "tree_sha256": _tree_commitment(all_entries),
        "result": result,
        "host_path_occurrences": find_host_path_occurrences(resolved),
    }


def _validate_raw_result(
    raw: Any,
    listed: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    result = _require_closed_fields(
        raw,
        RAW_RESULT_REQUIRED_FIELDS,
        error="RAW_RESULT_UNKNOWN_FIELDS",
    )
    if result["schema_version"] != "1.0":
        raise ProvenanceProjectionError("RAW_RESULT_SCHEMA_VERSION")
    if (
        result["method"] != "GFN2-xTB"
        or result["phase"] != "gas"
        or result["charge"] != 0
        or result["uhf"] != 0
        or result["threads"] != 1
        or result["optimization"] != "tight"
        or result["shell"] is not False
    ):
        raise ProvenanceProjectionError("RAW_RESULT_SCIENTIFIC_INVARIANT")
    angle = _require_int(result["angle_deg"], "RAW_RESULT_ANGLE")
    if angle < 0 or angle >= 360 or angle % 5:
        raise ProvenanceProjectionError("RAW_RESULT_ANGLE")
    for name in (
        "angle_error_deg",
        "duration_seconds",
        "final_angle_deg",
        "total_energy_hartree",
    ):
        _require_finite_number(result[name], f"RAW_RESULT_{name.upper()}", allow_none=True)
    _require_finite_number(
        result["requested_final_angle_tolerance_deg"],
        "RAW_RESULT_ANGLE_TOLERANCE",
    )
    _require_int(result["return_code"], "RAW_RESULT_RETURN_CODE")
    _require_int(result["timeout_seconds"], "RAW_RESULT_TIMEOUT")
    for name in ("converged", "profile_eligible", "timed_out"):
        _require_bool(result[name], f"RAW_RESULT_{name.upper()}")
    if result["parse_error"] is not None and not isinstance(
        result["parse_error"], str
    ):
        raise ProvenanceProjectionError("RAW_RESULT_PARSE_ERROR")

    constraint = _require_closed_fields(
        result["constraint"],
        CONSTRAINT_FIELDS,
        error="RAW_RESULT_CONSTRAINT_SCHEMA",
    )
    force_constant = _require_finite_number(
        constraint["force_constant_au"],
        "RAW_RESULT_CONSTRAINT_FORCE_CONSTANT",
    )
    if (
        constraint["type"] != "dihedral"
        or constraint["atom_maps"] != [6, 7, 8, 9]
        or constraint["atom_indices_one_based"] != [6, 7, 8, 9]
        or constraint["target_deg"] != angle
        or float(force_constant) != 1.0
    ):
        raise ProvenanceProjectionError("RAW_RESULT_CONSTRAINT_INVARIANT")

    prepared = _require_closed_fields(
        result["prepared_input"],
        PREPARED_INPUT_FIELDS,
        error="RAW_RESULT_PREPARED_INPUT_SCHEMA",
    )
    input_sha = _require_sha256(
        prepared["input_sha256"], "RAW_RESULT_INPUT_HASH"
    )
    reference_sha = _require_sha256(
        result["reference_sha256"], "RAW_RESULT_REFERENCE_HASH"
    )
    if (
        prepared["angle_deg"] != angle
        or _require_sha256(
            prepared["reference_sha256"], "RAW_RESULT_REFERENCE_HASH"
        )
        != reference_sha
        or "input.xyz" not in listed
        or listed["input.xyz"]["sha256"] != input_sha
    ):
        raise ProvenanceProjectionError("RAW_RESULT_PREPARED_INPUT_BINDING")
    _require_finite_number(
        prepared["uff_angle_error_deg"], "RAW_RESULT_UFF_ANGLE_ERROR"
    )
    _require_finite_number(
        prepared["uff_final_angle_deg"], "RAW_RESULT_UFF_FINAL_ANGLE"
    )

    environment = _require_closed_fields(
        result["environment"],
        ENVIRONMENT_FIELDS,
        error="RAW_RESULT_ENVIRONMENT_SCHEMA",
    )
    if any(environment[name] != "1" for name in ENVIRONMENT_FIELDS):
        raise ProvenanceProjectionError("RAW_RESULT_ENVIRONMENT_INVARIANT")

    xtb = _require_closed_fields(
        result["xtb"],
        XTB_FIELDS,
        error="RAW_RESULT_XTB_SCHEMA",
    )
    executable = xtb["executable"]
    if not isinstance(executable, str) or _path_rule(executable) not in {
        "windows_drive_absolute",
        "posix_absolute",
        "posix_host_absolute",
        "windows_unc_or_device",
    }:
        raise ProvenanceProjectionError("RAW_RESULT_XTB_EXECUTABLE_PATH")
    if (
        xtb["version"] != EXPECTED_XTB_VERSION
        or _require_sha256(xtb["binary_sha256"], "RAW_RESULT_XTB_HASH")
        != EXPECTED_XTB_SHA256
        or _require_int(xtb["version_exit_code"], "RAW_RESULT_XTB_VERSION_EXIT")
        != 0
    ):
        raise ProvenanceProjectionError("RAW_RESULT_XTB_IDENTITY")

    argv = result["argv"]
    if (
        not isinstance(argv, list)
        or any(not isinstance(item, str) for item in argv)
        or not argv
        or argv[0] != executable
        or tuple(argv[1:]) != FROZEN_ARGV_TAIL
    ):
        raise ProvenanceProjectionError("RAW_RESULT_ARGV_INVARIANT")
    for item in argv[1:]:
        if _path_rule(item) is not None or any(char in item for char in "\\/\x00\r\n"):
            raise ProvenanceProjectionError("RAW_RESULT_ARGV_PATH_ESCAPE")

    structure = _require_closed_fields(
        result["structure_validation"],
        STRUCTURE_REQUIRED_FIELDS,
        optional=STRUCTURE_OPTIONAL_FIELDS,
        error="RAW_RESULT_STRUCTURE_SCHEMA",
    )
    for name in STRUCTURE_REQUIRED_FIELDS:
        _require_bool(structure[name], "RAW_RESULT_STRUCTURE_VALUE")
    for name in ("expected_bond_count", "inferred_bond_count"):
        if name in structure:
            _require_int(structure[name], "RAW_RESULT_STRUCTURE_COUNT")
    for name in ("missing_bonds", "unexpected_bonds"):
        if name in structure and not isinstance(structure[name], list):
            raise ProvenanceProjectionError("RAW_RESULT_STRUCTURE_BONDS")
    return result


def _artifact_hash(
    listed: Mapping[str, Mapping[str, Any]], name: str
) -> str | None:
    item = listed.get(name)
    return None if item is None else str(item["sha256"])


def _projection_without_hash(
    inspection: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    listed = inspection["listed"]
    occurrences = list(inspection["host_path_occurrences"])
    files_with_paths = sorted({str(item["path"]) for item in occurrences})
    constraint = dict(result["constraint"])
    prepared = dict(result["prepared_input"])
    structure = dict(result["structure_validation"])
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_kind": PROJECTION_KIND,
        "source_binding": {
            "algorithm": (
                "tree-sha256-v1:"
                "ordinal(relative_path+NUL+size+NUL+file_sha256+LF)"
            ),
            "attempt_manifest_sha256": inspection["manifest_sha256"],
            "attempt_tree_sha256": inspection["tree_sha256"],
            "file_count": inspection["file_count"],
            "total_bytes": inspection["total_bytes"],
            "files": list(inspection["files"]),
        },
        "execution": {
            "executable_role": "xtb",
            "executable_version": result["xtb"]["version"],
            "executable_sha256": str(result["xtb"]["binary_sha256"]).upper(),
            "host_path_present": bool(occurrences),
            "host_path_redacted": XTB_EXECUTABLE_TOKEN,
            "working_directory_redacted": ATTEMPT_WORKDIR_TOKEN,
            "argv_template": [XTB_EXECUTABLE_TOKEN, *FROZEN_ARGV_TAIL],
            "shell": False,
            "environment": {
                name: result["environment"][name]
                for name in sorted(ENVIRONMENT_FIELDS)
            },
            "timeout_seconds": result["timeout_seconds"],
            "exit_code": result["return_code"],
            "timed_out": result["timed_out"],
            "converged": result["converged"],
        },
        "scientific_record": {
            "angle_deg": result["angle_deg"],
            "final_angle_deg": result["final_angle_deg"],
            "angle_error_deg": result["angle_error_deg"],
            "requested_final_angle_tolerance_deg": result[
                "requested_final_angle_tolerance_deg"
            ],
            "total_energy_hartree": result["total_energy_hartree"],
            "method": result["method"],
            "phase": result["phase"],
            "charge": result["charge"],
            "uhf": result["uhf"],
            "threads": result["threads"],
            "optimization": result["optimization"],
            "constraint": constraint,
            "reference_sha256": str(result["reference_sha256"]).upper(),
            "prepared_input": {
                "angle_deg": prepared["angle_deg"],
                "input_sha256": str(prepared["input_sha256"]).upper(),
                "reference_sha256": str(prepared["reference_sha256"]).upper(),
                "uff_angle_error_deg": prepared["uff_angle_error_deg"],
                "uff_final_angle_deg": prepared["uff_final_angle_deg"],
            },
            "structure_validation": structure,
            "profile_eligible": result["profile_eligible"],
            "parse_error_present": result["parse_error"] is not None,
        },
        "artifact_hashes": {
            "input_sha256": _artifact_hash(listed, "input.xyz"),
            "control_sha256": _artifact_hash(listed, "xcontrol.inp"),
            "output_sha256": _artifact_hash(listed, "xtbopt.xyz"),
            "result_sha256": _artifact_hash(listed, "result.json"),
            "stdout_sha256": _artifact_hash(listed, "stdout.log"),
            "stderr_sha256": _artifact_hash(listed, "stderr.log"),
        },
        "redaction": {
            "raw_host_path_occurrence_count": len(occurrences),
            "raw_files_with_host_paths": files_with_paths,
            "raw_values_persisted": False,
        },
    }


def build_package_safe_projection(attempt_root: Path) -> dict[str, Any]:
    """Verify a raw native attempt and return its deterministic projection."""

    inspection = _inspect_attempt(attempt_root)
    result = _validate_raw_result(inspection["result"], inspection["listed"])
    if not inspection["host_path_occurrences"]:
        raise ProvenanceProjectionError("RAW_ATTEMPT_HOST_PATH_NOT_DETECTED")
    payload = _projection_without_hash(inspection, result)
    payload["projection_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    validate_package_safe_projection(payload)
    return payload


def validate_package_safe_projection(payload: Any) -> dict[str, Any]:
    """Validate the closed projection and its self-binding hash."""

    top = _require_closed_fields(
        payload,
        frozenset(
            {
                "artifact_hashes",
                "execution",
                "projection_kind",
                "projection_sha256",
                "redaction",
                "schema_version",
                "scientific_record",
                "source_binding",
            }
        ),
        error="PROJECTION_SCHEMA",
    )
    if (
        top["schema_version"] != PROJECTION_SCHEMA_VERSION
        or top["projection_kind"] != PROJECTION_KIND
    ):
        raise ProvenanceProjectionError("PROJECTION_IDENTITY")
    source = _require_closed_fields(
        top["source_binding"],
        frozenset(
            {
                "algorithm",
                "attempt_manifest_sha256",
                "attempt_tree_sha256",
                "file_count",
                "files",
                "total_bytes",
            }
        ),
        error="PROJECTION_SOURCE_SCHEMA",
    )
    _require_sha256(
        source["attempt_manifest_sha256"], "PROJECTION_SOURCE_MANIFEST_HASH"
    )
    _require_sha256(
        source["attempt_tree_sha256"], "PROJECTION_SOURCE_TREE_HASH"
    )
    file_count = _require_int(source["file_count"], "PROJECTION_SOURCE_COUNT")
    total_bytes = _require_int(source["total_bytes"], "PROJECTION_SOURCE_BYTES")
    if file_count < 1 or total_bytes < 0 or not isinstance(source["files"], list):
        raise ProvenanceProjectionError("PROJECTION_SOURCE_INVENTORY")
    normalized_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in source["files"]:
        item = _require_closed_fields(
            raw_item,
            frozenset({"path", "sha256", "size_bytes"}),
            error="PROJECTION_SOURCE_ENTRY",
        )
        name = _safe_flat_name(item["path"], "PROJECTION_SOURCE_PATH")
        if name in seen:
            raise ProvenanceProjectionError("PROJECTION_SOURCE_DUPLICATE")
        seen.add(name)
        size = _require_int(item["size_bytes"], "PROJECTION_SOURCE_SIZE")
        if size < 0:
            raise ProvenanceProjectionError("PROJECTION_SOURCE_SIZE")
        normalized_files.append(
            {
                "path": name,
                "size_bytes": size,
                "sha256": _require_sha256(
                    item["sha256"], "PROJECTION_SOURCE_FILE_HASH"
                ),
            }
        )
    if (
        len(normalized_files) != file_count
        or sum(int(item["size_bytes"]) for item in normalized_files) != total_bytes
        or _tree_commitment(normalized_files) != source["attempt_tree_sha256"]
    ):
        raise ProvenanceProjectionError("PROJECTION_SOURCE_BINDING_MISMATCH")

    execution = _require_closed_fields(
        top["execution"],
        frozenset(
            {
                "argv_template",
                "converged",
                "environment",
                "executable_role",
                "executable_sha256",
                "executable_version",
                "exit_code",
                "host_path_present",
                "host_path_redacted",
                "shell",
                "timed_out",
                "timeout_seconds",
                "working_directory_redacted",
            }
        ),
        error="PROJECTION_EXECUTION_SCHEMA",
    )
    if (
        execution["executable_role"] != "xtb"
        or execution["executable_version"] != EXPECTED_XTB_VERSION
        or _require_sha256(
            execution["executable_sha256"], "PROJECTION_EXECUTABLE_HASH"
        )
        != EXPECTED_XTB_SHA256
        or execution["host_path_present"] is not True
        or execution["host_path_redacted"] != XTB_EXECUTABLE_TOKEN
        or execution["working_directory_redacted"] != ATTEMPT_WORKDIR_TOKEN
        or execution["shell"] is not False
        or execution["argv_template"]
        != [XTB_EXECUTABLE_TOKEN, *FROZEN_ARGV_TAIL]
    ):
        raise ProvenanceProjectionError("PROJECTION_EXECUTION_INVARIANT")
    environment = _require_closed_fields(
        execution["environment"],
        ENVIRONMENT_FIELDS,
        error="PROJECTION_ENVIRONMENT_SCHEMA",
    )
    if any(environment[name] != "1" for name in ENVIRONMENT_FIELDS):
        raise ProvenanceProjectionError("PROJECTION_ENVIRONMENT_INVARIANT")
    _require_int(execution["exit_code"], "PROJECTION_EXIT_CODE")
    _require_int(execution["timeout_seconds"], "PROJECTION_TIMEOUT")
    _require_bool(execution["converged"], "PROJECTION_CONVERGED")
    _require_bool(execution["timed_out"], "PROJECTION_TIMED_OUT")

    science = _require_closed_fields(
        top["scientific_record"],
        frozenset(
            {
                "angle_deg",
                "angle_error_deg",
                "charge",
                "constraint",
                "final_angle_deg",
                "method",
                "optimization",
                "parse_error_present",
                "phase",
                "prepared_input",
                "profile_eligible",
                "reference_sha256",
                "requested_final_angle_tolerance_deg",
                "structure_validation",
                "threads",
                "total_energy_hartree",
                "uhf",
            }
        ),
        error="PROJECTION_SCIENCE_SCHEMA",
    )
    if (
        science["method"] != "GFN2-xTB"
        or science["phase"] != "gas"
        or science["charge"] != 0
        or science["uhf"] != 0
        or science["threads"] != 1
        or science["optimization"] != "tight"
    ):
        raise ProvenanceProjectionError("PROJECTION_SCIENCE_INVARIANT")
    _require_sha256(science["reference_sha256"], "PROJECTION_REFERENCE_HASH")
    _require_bool(science["profile_eligible"], "PROJECTION_PROFILE_ELIGIBLE")
    _require_bool(
        science["parse_error_present"], "PROJECTION_PARSE_ERROR_PRESENT"
    )
    angle = _require_int(science["angle_deg"], "PROJECTION_ANGLE")
    if angle < 0 or angle >= 360 or angle % 5:
        raise ProvenanceProjectionError("PROJECTION_ANGLE")
    for name in (
        "angle_error_deg",
        "final_angle_deg",
        "total_energy_hartree",
    ):
        _require_finite_number(
            science[name],
            f"PROJECTION_{name.upper()}",
            allow_none=True,
        )
    _require_finite_number(
        science["requested_final_angle_tolerance_deg"],
        "PROJECTION_ANGLE_TOLERANCE",
    )
    constraint = _require_closed_fields(
        science["constraint"],
        CONSTRAINT_FIELDS,
        error="PROJECTION_CONSTRAINT_SCHEMA",
    )
    force_constant = _require_finite_number(
        constraint["force_constant_au"],
        "PROJECTION_CONSTRAINT_FORCE_CONSTANT",
    )
    if (
        constraint["type"] != "dihedral"
        or constraint["atom_maps"] != [6, 7, 8, 9]
        or constraint["atom_indices_one_based"] != [6, 7, 8, 9]
        or constraint["target_deg"] != angle
        or float(force_constant) != 1.0
    ):
        raise ProvenanceProjectionError("PROJECTION_CONSTRAINT_INVARIANT")
    prepared = _require_closed_fields(
        science["prepared_input"],
        PREPARED_INPUT_FIELDS,
        error="PROJECTION_PREPARED_INPUT_SCHEMA",
    )
    if (
        prepared["angle_deg"] != angle
        or _require_sha256(
            prepared["reference_sha256"], "PROJECTION_PREPARED_REFERENCE_HASH"
        )
        != science["reference_sha256"]
    ):
        raise ProvenanceProjectionError("PROJECTION_PREPARED_INPUT_BINDING")
    _require_sha256(
        prepared["input_sha256"], "PROJECTION_PREPARED_INPUT_HASH"
    )
    _require_finite_number(
        prepared["uff_angle_error_deg"], "PROJECTION_UFF_ANGLE_ERROR"
    )
    _require_finite_number(
        prepared["uff_final_angle_deg"], "PROJECTION_UFF_FINAL_ANGLE"
    )
    structure = _require_closed_fields(
        science["structure_validation"],
        STRUCTURE_REQUIRED_FIELDS,
        optional=STRUCTURE_OPTIONAL_FIELDS,
        error="PROJECTION_STRUCTURE_SCHEMA",
    )
    for name in STRUCTURE_REQUIRED_FIELDS:
        _require_bool(structure[name], "PROJECTION_STRUCTURE_VALUE")
    for name in ("expected_bond_count", "inferred_bond_count"):
        if name in structure:
            _require_int(structure[name], "PROJECTION_STRUCTURE_COUNT")
    for name in ("missing_bonds", "unexpected_bonds"):
        if name in structure and not isinstance(structure[name], list):
            raise ProvenanceProjectionError("PROJECTION_STRUCTURE_BONDS")

    artifacts = _require_closed_fields(
        top["artifact_hashes"],
        frozenset(
            {
                "control_sha256",
                "input_sha256",
                "output_sha256",
                "result_sha256",
                "stderr_sha256",
                "stdout_sha256",
            }
        ),
        error="PROJECTION_ARTIFACT_SCHEMA",
    )
    file_hashes = {
        str(item["path"]): str(item["sha256"]) for item in normalized_files
    }
    expected_artifacts = {
        "input_sha256": file_hashes.get("input.xyz"),
        "control_sha256": file_hashes.get("xcontrol.inp"),
        "output_sha256": file_hashes.get("xtbopt.xyz"),
        "result_sha256": file_hashes.get("result.json"),
        "stdout_sha256": file_hashes.get("stdout.log"),
        "stderr_sha256": file_hashes.get("stderr.log"),
    }
    if dict(artifacts) != expected_artifacts:
        raise ProvenanceProjectionError("PROJECTION_ARTIFACT_BINDING_MISMATCH")
    if prepared["input_sha256"] != artifacts["input_sha256"]:
        raise ProvenanceProjectionError("PROJECTION_INPUT_BINDING_MISMATCH")

    redaction = _require_closed_fields(
        top["redaction"],
        frozenset(
            {
                "raw_files_with_host_paths",
                "raw_host_path_occurrence_count",
                "raw_values_persisted",
            }
        ),
        error="PROJECTION_REDACTION_SCHEMA",
    )
    count = _require_int(
        redaction["raw_host_path_occurrence_count"],
        "PROJECTION_REDACTION_COUNT",
    )
    files = redaction["raw_files_with_host_paths"]
    if (
        count < 1
        or not isinstance(files, list)
        or not files
        or files != sorted(set(files))
        or any(
            _safe_flat_name(item, "PROJECTION_REDACTION_FILE") not in file_hashes
            for item in files
        )
        or redaction["raw_values_persisted"] is not False
    ):
        raise ProvenanceProjectionError("PROJECTION_REDACTION_INVARIANT")

    findings = projection_host_path_findings(top)
    if findings:
        raise ProvenanceProjectionError(
            "PROJECTION_HOST_PATH_PRESENT:"
            + ",".join(item["location"] for item in findings)
        )
    claimed_hash = _require_sha256(
        top["projection_sha256"], "PROJECTION_HASH_INVALID"
    )
    without_hash = dict(top)
    without_hash.pop("projection_sha256")
    expected_hash = sha256_bytes(canonical_json_bytes(without_hash))
    if claimed_hash != expected_hash:
        raise ProvenanceProjectionError("PROJECTION_HASH_MISMATCH")
    return {
        "passed": True,
        "projection_sha256": expected_hash,
        "source_attempt_tree_sha256": source["attempt_tree_sha256"],
        "source_attempt_manifest_sha256": source[
            "attempt_manifest_sha256"
        ],
        "source_file_count": file_count,
    }


def write_package_safe_projection(
    attempt_root: Path,
    destination: Path,
    *,
    exclusive: bool = True,
) -> dict[str, Any]:
    """Write a canonical projection outside the immutable raw attempt."""

    before = _inspect_attempt(attempt_root)
    raw_root = before["root"]
    target = destination.resolve(strict=False)
    try:
        target.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ProvenanceProjectionError("PROJECTION_DESTINATION_INSIDE_RAW_ATTEMPT")
    payload = build_package_safe_projection(raw_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    try:
        with destination.open(mode) as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise ProvenanceProjectionError("PROJECTION_DESTINATION_EXISTS") from exc
    after = _inspect_attempt(raw_root)
    if (
        before["tree_sha256"] != after["tree_sha256"]
        or before["manifest_sha256"] != after["manifest_sha256"]
    ):
        raise ProvenanceProjectionError("RAW_ATTEMPT_CHANGED_DURING_PROJECTION")
    return payload


# Stable concise aliases for release orchestration.
project_attempt = build_package_safe_projection
validate_projection = validate_package_safe_projection
write_projection = write_package_safe_projection


__all__ = [
    "ATTEMPT_WORKDIR_TOKEN",
    "EXPECTED_XTB_SHA256",
    "EXPECTED_XTB_VERSION",
    "PROJECTION_KIND",
    "PROJECTION_SCHEMA_VERSION",
    "ProvenanceProjectionError",
    "XTB_EXECUTABLE_TOKEN",
    "build_package_safe_projection",
    "canonical_json_bytes",
    "find_host_path_occurrences",
    "project_attempt",
    "projection_host_path_findings",
    "sha256_bytes",
    "sha256_file",
    "validate_package_safe_projection",
    "validate_projection",
    "write_package_safe_projection",
    "write_projection",
]
