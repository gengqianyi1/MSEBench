"""Independent trajectory/artifact evaluator.

This module intentionally imports neither ``framework`` nor ``tools``.  It
duplicates hash-chain, xTB-output, geometry, fit, budget and binding checks so
the trusted runner's summaries are never taken as scoring facts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from evaluator.scoring import compute_score, circular_distance_deg


EVALUATOR_VERSION = "adaptive-torsion-evaluator/1.1"
GENESIS_HASH = "0" * 64
GRID = list(range(0, 360, 5))
HARTREE_TO_KCAL_MOL = 627.5094740631
CAMPAIGN_IDENTITY = {
    "benchmark_version": "1.1.0",
    "case_id": "2_fluorobiphenyl_central_torsion_v1",
    "formal_campaign_id": "formal_campaign_v1_1_rc1",
    "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
    "task_id": "adaptive_xtb_torsion_profile_agent_v1_1",
}
EXPECTED_TRAJECTORY_PROTOCOL = {
    "name": "adaptive_torsion_sampling_tool_loop",
    "version": "1.1",
}
EXPECTED_REFERENCE_SHA256 = "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
EXPECTED_XTB_SHA256 = "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
FORMAL_MODELS = {
    "Pro/moonshotai/Kimi-K2.6",
    "deepseek-ai/DeepSeek-V4-Pro",
    "zai-org/GLM-5.2",
    "deterministic-baseline/1.1",
}

_CAMPAIGN_INCLUDED_TOP_LEVEL = {
    "baselines",
    "benchmark_case",
    "evaluator",
    "framework",
    "package_tools",
    "prompts",
    "scripts",
    "tests",
    "tools",
}
_CAMPAIGN_INCLUDED_ROOT_FILES = {
    ".env.example",
    "environment.yml",
    "pytest.ini",
    "requirements.txt",
    "TASK.md",
}
_CAMPAIGN_EXCLUDED_PATHS = {
    "configs/campaign_snapshot.json",
    "configs/release.json",
}
_CAMPAIGN_INCLUDED_CONFIGS = {
    "configs/models.example.yaml",
    "configs/task.yaml",
    "configs/tool_contracts.json",
}
_CAMPAIGN_INCLUDED_PRIVATE_ASSETS = {
    "evaluator_private/oracle_commitment.json",
    "evaluator_private/oracle_profile.json",
    "evaluator_private/oracle_provenance.json",
}
_CAMPAIGN_ALLOWED_SUFFIXES = {
    ".example",
    ".inp",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mol",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
    ".xyz",
    ".yaml",
    ".yml",
}
_CAMPAIGN_ALLOWED_EXTENSIONLESS = {
    ".xtboptok",
    "charges",
    "energy",
    "gradient",
    "wbo",
    "xtbrestart",
}


class EvaluationError(RuntimeError):
    pass


def _canonical_compact(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_pretty(value: Any) -> bytes:
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


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _strict_json(path: Path) -> Any:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid JSON artifact: {path.name}") from exc
    if duplicates:
        raise EvaluationError(f"duplicate JSON keys in {path.name}")
    return value


def _campaign_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(project_root)
        posix = relative.as_posix()
        if "__pycache__" in relative.parts or posix in _CAMPAIGN_EXCLUDED_PATHS:
            continue
        if len(relative.parts) == 1:
            if relative.name in _CAMPAIGN_INCLUDED_ROOT_FILES:
                files.append(path)
            continue
        if relative.parts[0] == "configs":
            if posix in _CAMPAIGN_INCLUDED_CONFIGS:
                files.append(path)
            continue
        if relative.parts[0] == "evaluator_private":
            if posix in _CAMPAIGN_INCLUDED_PRIVATE_ASSETS:
                files.append(path)
            continue
        if relative.parts[0] not in _CAMPAIGN_INCLUDED_TOP_LEVEL:
            continue
        if (
            relative.suffix.casefold() in _CAMPAIGN_ALLOWED_SUFFIXES
            or relative.name in _CAMPAIGN_ALLOWED_EXTENSIONLESS
        ):
            files.append(path)
    return sorted(
        files,
        key=lambda path: path.relative_to(project_root).as_posix(),
    )


def _campaign_snapshot(project_root: Path) -> dict[str, Any]:
    snapshot_path = project_root / "configs" / "campaign_snapshot.json"
    payload = _strict_json(snapshot_path)
    if not isinstance(payload, Mapping):
        raise EvaluationError("campaign snapshot is not an object")
    for field, expected in CAMPAIGN_IDENTITY.items():
        if payload.get(field) != expected:
            raise EvaluationError("campaign snapshot identity mismatch")
    entries = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in _campaign_files(project_root)
    ]
    if payload.get("files") != entries:
        raise EvaluationError("campaign snapshot file inventory mismatch")
    if payload.get("file_count") != len(entries):
        raise EvaluationError("campaign snapshot file count mismatch")
    if payload.get("total_bytes") != sum(
        int(item["size_bytes"]) for item in entries
    ):
        raise EvaluationError("campaign snapshot byte count mismatch")
    content = {"identity": CAMPAIGN_IDENTITY, "files": entries}
    commitment = _sha_bytes(_canonical_pretty(content))
    if payload.get("content_commitment_sha256") != commitment:
        raise EvaluationError("campaign snapshot content commitment mismatch")
    return {
        **CAMPAIGN_IDENTITY,
        "file_count": len(entries),
        "content_commitment_sha256": commitment,
        "snapshot_file_sha256": _sha_file(snapshot_path),
    }


def _trajectory(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvaluationError("trajectory is unreadable") from exc
    lines = raw.splitlines()
    if not lines or any(not line for line in lines):
        raise EvaluationError("trajectory has empty/missing lines")
    events: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    run_id: str | None = None
    protocol: Any = None
    final_seen = False
    terminal_seen = False
    for sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError("trajectory JSONL is invalid") from exc
        if line.encode("utf-8") != _canonical_compact(event):
            raise EvaluationError("trajectory line is not canonical")
        expected_keys = {
            "schema_version",
            "protocol",
            "run_id",
            "sequence",
            "timestamp_utc",
            "event",
            "payload",
            "prev_event_hash",
            "event_hash",
        }
        if not isinstance(event, dict) or set(event) != expected_keys:
            raise EvaluationError("trajectory event envelope changed")
        if event["sequence"] != sequence or event["prev_event_hash"] != previous:
            raise EvaluationError("trajectory order/previous hash is invalid")
        without_hash = {key: value for key, value in event.items() if key != "event_hash"}
        calculated = hashlib.sha256(_canonical_compact(without_hash)).hexdigest()
        if event["event_hash"] != calculated:
            raise EvaluationError("trajectory event hash mismatch")
        previous = calculated
        if run_id is None:
            run_id = event["run_id"]
            protocol = event["protocol"]
        elif event["run_id"] != run_id or event["protocol"] != protocol:
            raise EvaluationError("trajectory identity changed")
        if event["protocol"] != EXPECTED_TRAJECTORY_PROTOCOL:
            raise EvaluationError("trajectory protocol is not v1.1")
        name = event["event"]
        if sequence == 1 and name != "run_start":
            raise EvaluationError("trajectory does not begin with run_start")
        if sequence == 1:
            payload = event.get("payload")
            if not isinstance(payload, Mapping) or any(
                payload.get(field) != expected
                for field, expected in CAMPAIGN_IDENTITY.items()
            ):
                raise EvaluationError("run_start campaign identity mismatch")
            for field in (
                "campaign_content_commitment_sha256",
                "campaign_snapshot_sha256",
            ):
                value = payload.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise EvaluationError("run_start campaign hash invalid")
        if sequence > 1 and name == "run_start":
            raise EvaluationError("trajectory has duplicate run_start")
        if terminal_seen:
            raise EvaluationError("trajectory has events after terminal")
        if final_seen and name != "run_complete":
            raise EvaluationError("trajectory changed after final submission")
        if name == "final_submitted":
            if final_seen:
                raise EvaluationError("trajectory has duplicate final submission")
            final_seen = True
        if name in {"run_complete", "run_failure"}:
            terminal_seen = True
        events.append(event)
    if not terminal_seen or events[-1]["event"] not in {"run_complete", "run_failure"}:
        raise EvaluationError("trajectory lacks terminal event")
    if events[-1]["event"] == "run_complete" and not final_seen:
        raise EvaluationError("successful trajectory lacks final submission")
    return events, {
        "valid": True,
        "event_count": len(events),
        "head_hash": previous,
        "file_sha256": _sha_file(path),
        "terminal_event": events[-1]["event"],
    }


def _tool_pairs(events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    valid = True
    for event in events:
        payload = event.get("payload", {})
        if event.get("event") == "tool_call":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or call_id in calls:
                valid = False
                continue
            calls[call_id] = dict(payload)
            ordered.append(calls[call_id])
        elif event.get("event") == "tool_result":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or call_id in results:
                valid = False
                continue
            results[call_id] = dict(payload)
    for call_id, call in calls.items():
        result = results.get(call_id)
        if (
            result is None
            or result.get("tool") != call.get("tool")
            or result.get("turn") != call.get("turn")
        ):
            valid = False
        call["paired_result"] = result.get("result") if result else None
    if set(calls) != set(results):
        valid = False
    return ordered, valid


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise EvaluationError("XYZ truncated")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise EvaluationError("XYZ count invalid") from exc
    rows = lines[2 : 2 + count]
    if len(rows) != count:
        raise EvaluationError("XYZ rows truncated")
    symbols: list[str] = []
    coordinates = []
    for row in rows:
        fields = row.split()
        if len(fields) < 4:
            raise EvaluationError("XYZ row invalid")
        symbols.append(fields[0])
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
    array = np.asarray(coordinates, dtype=float)
    if not np.all(np.isfinite(array)):
        raise EvaluationError("XYZ is non-finite")
    return symbols, array


def _dihedral_deg(points: np.ndarray) -> float:
    p0, p1, p2, p3 = points
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    value = math.degrees(math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w))))
    return value % 360.0


def _connectivity(symbols: Sequence[str], coordinates: np.ndarray) -> set[tuple[int, int]]:
    radii = {"H": 0.31, "C": 0.76, "F": 0.57}
    bonds: set[tuple[int, int]] = set()
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            cutoff = 1.25 * (radii[symbols[i]] + radii[symbols[j]])
            if float(np.linalg.norm(coordinates[i] - coordinates[j])) <= cutoff:
                bonds.add((i, j))
    return bonds


def _manifest_valid(root: Path) -> tuple[bool, str]:
    path = root / "attempt_manifest.json"
    try:
        manifest = _strict_json(path)
    except EvaluationError:
        return False, ""
    actual = []
    for member in sorted(root.iterdir(), key=lambda item: item.name):
        if member.is_file() and member.name != "attempt_manifest.json":
            actual.append(
                {"path": member.name, "size_bytes": member.stat().st_size, "sha256": _sha_file(member)}
            )
    expected = {"schema_version": "1.0", "algorithm": "SHA-256", "files": actual}
    return manifest == expected, _sha_file(path)


def _validate_attempt(root: Path, *, reference_symbols: list[str], expected_bonds: set[tuple[int, int]]) -> dict[str, Any]:
    attempt_id = root.name
    match = re.fullmatch(r"angle_([0-9]{3})_attempt_([0-9]{2})", attempt_id)
    if not match:
        return {"attempt_id": attempt_id, "valid": False, "reason": "ATTEMPT_ID_INVALID"}
    angle = int(match.group(1))
    manifest_ok, manifest_hash = _manifest_valid(root)
    required = ["result.json", "input.xyz", "xcontrol.inp", "stdout.log", "stderr.log", "xtbopt.xyz"]
    if any(not (root / name).is_file() for name in required):
        return {"attempt_id": attempt_id, "angle_deg": angle, "valid": False, "reason": "ATTEMPT_FILES_MISSING"}
    try:
        result = _strict_json(root / "result.json")
        stdout = (root / "stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr = (root / "stderr.log").read_text(encoding="utf-8", errors="replace")
        symbols, coordinates = _read_xyz(root / "xtbopt.xyz")
        input_symbols, _ = _read_xyz(root / "input.xyz")
    except (EvaluationError, OSError, ValueError):
        return {"attempt_id": attempt_id, "angle_deg": angle, "valid": False, "reason": "ATTEMPT_PARSE_FAILURE"}
    energy_matches = re.findall(r"\|\s*TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh\s*\|", stdout)
    comment = (root / "xtbopt.xyz").read_text(encoding="utf-8", errors="replace").splitlines()[1]
    comment_match = re.search(r"\benergy:\s*([-+0-9.Ee]+)", comment)
    energy = float(energy_matches[-1]) if energy_matches else math.nan
    comment_energy = float(comment_match.group(1)) if comment_match else math.nan
    final_angle = _dihedral_deg(coordinates[[5, 6, 7, 8], :])
    inferred_bonds = _connectivity(symbols, coordinates)
    control_expected = (
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: 6, 7, 8, 9, {(angle if angle <= 180 else angle - 360):.1f}\n"
        "$end\n"
    )
    argv = result.get("argv")
    argv_ok = bool(
        isinstance(argv, list)
        and len(argv) == 12
        and str(argv[0]).casefold().endswith("xtb.exe")
        and argv[1:] == [
            "input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight", "--input", "xcontrol.inp"
        ]
    )
    invariant_ok = bool(
        result.get("angle_deg") == angle
        and result.get("method") == "GFN2-xTB"
        and result.get("phase") == "gas"
        and result.get("charge") == 0
        and result.get("uhf") == 0
        and result.get("threads") == 1
        and result.get("shell") is False
        and result.get("reference_sha256") == EXPECTED_REFERENCE_SHA256
        and result.get("xtb", {}).get("version") == "6.7.1"
        and result.get("xtb", {}).get("binary_sha256") == EXPECTED_XTB_SHA256
        and result.get("environment") == {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        and argv_ok
    )
    structure_ok = bool(
        symbols == reference_symbols
        and input_symbols == reference_symbols
        and Counter(symbols) == Counter(reference_symbols)
        and inferred_bonds == expected_bonds
    )
    native_ok = bool(
        result.get("return_code") == 0
        and result.get("timed_out") is False
        and "GEOMETRY OPTIMIZATION CONVERGED" in stdout
        and "FAILED TO CONVERGE" not in stdout
        and "normal termination of xtb" in stderr
    )
    energy_ok = bool(
        math.isfinite(energy)
        and math.isfinite(comment_energy)
        and abs(energy - comment_energy) <= 1.0e-9
        and result.get("total_energy_hartree") is not None
        and abs(float(result["total_energy_hartree"]) - energy) <= 1.0e-12
    )
    angle_ok = circular_distance_deg(final_angle, angle) <= 2.0
    input_hash_ok = result.get("prepared_input", {}).get("input_sha256") == _sha_file(root / "input.xyz")
    control_ok = (root / "xcontrol.inp").read_text(encoding="ascii") == control_expected
    valid = all(
        [manifest_ok, invariant_ok, structure_ok, native_ok, energy_ok, angle_ok, input_hash_ok, control_ok]
    )
    return {
        "attempt_id": attempt_id,
        "angle_deg": angle,
        "valid": valid,
        "profile_eligible": valid,
        "energy_hartree": energy if math.isfinite(energy) else None,
        "final_angle_deg": final_angle,
        "angle_error_deg": circular_distance_deg(final_angle, angle),
        "manifest_valid": manifest_ok,
        "manifest_sha256": manifest_hash,
        "invariants_valid": invariant_ok,
        "structure_valid": structure_ok,
        "native_valid": native_ok,
        "energy_valid": energy_ok,
        "control_valid": control_ok,
    }


def _periodic_cubic(angles: np.ndarray, energies: np.ndarray) -> np.ndarray:
    order = np.argsort(angles)
    x = angles[order]
    y = energies[order]
    n = len(x)
    widths = np.array([(x[(i + 1) % n] + (360.0 if i == n - 1 else 0.0)) - x[i] for i in range(n)])
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)
    for i in range(n):
        prev = (i - 1) % n
        nxt = (i + 1) % n
        hl, hr = widths[prev], widths[i]
        matrix[i, prev] += hl
        matrix[i, i] += 2.0 * (hl + hr)
        matrix[i, nxt] += hr
        rhs[i] = 6.0 * ((y[nxt] - y[i]) / hr - (y[i] - y[prev]) / hl)
    second = np.linalg.solve(matrix, rhs)
    output = []
    for raw in GRID:
        q = raw % 360.0
        i = int(np.searchsorted(x, q, side="right") - 1)
        if i < 0:
            i = n - 1
            q += 360.0
        j = (i + 1) % n
        left = x[i]
        right = x[j] + (360.0 if j == 0 else 0.0)
        h = right - left
        a = (right - q) / h
        b = (q - left) / h
        value = a * y[i] + b * y[j] + ((a**3 - a) * second[i] + (b**3 - b) * second[j]) * h * h / 6.0
        output.append(value)
    return np.asarray(output, dtype=float)


def _fourier(angles: np.ndarray, energies: np.ndarray, order: int) -> np.ndarray:
    def design(values: np.ndarray) -> np.ndarray:
        theta = np.deg2rad(values % 360.0)
        columns = [np.ones(len(theta))]
        for harmonic in range(1, order + 1):
            columns.extend([np.cos(harmonic * theta), np.sin(harmonic * theta)])
        return np.column_stack(columns)

    source = design(angles)
    if len(angles) < 1 + 2 * order or np.linalg.matrix_rank(source) != 1 + 2 * order:
        raise EvaluationError("profile Fourier fit is underdetermined")
    coefficients, _, rank, _ = np.linalg.lstsq(source, energies, rcond=None)
    if rank != 1 + 2 * order:
        raise EvaluationError("profile Fourier fit rank changed")
    return design(np.asarray(GRID, dtype=float)) @ coefficients


def _validate_profile(path: Path, attempts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    try:
        artifact = _strict_json(path)
    except EvaluationError:
        return {"profile_id": path.stem, "valid": False, "reason": "PROFILE_JSON_INVALID"}
    profile_id = artifact.get("profile_id")
    source = artifact.get("source_points")
    predictions = artifact.get("predicted_energy_kcal_mol")
    if (
        not isinstance(profile_id, str)
        or path.stem != profile_id
        or not isinstance(source, list)
        or not isinstance(predictions, list)
        or len(predictions) != 72
    ):
        return {"profile_id": path.stem, "valid": False, "reason": "PROFILE_SHAPE_INVALID"}
    source_valid = True
    angles = []
    absolute = []
    seen = set()
    for point in source:
        if not isinstance(point, Mapping):
            source_valid = False
            continue
        attempt_id = point.get("attempt_id")
        attempt = attempts.get(str(attempt_id))
        angle = point.get("angle_deg")
        if (
            attempt is None
            or attempt.get("valid") is not True
            or attempt.get("angle_deg") != angle
            or angle in seen
            or point.get("attempt_manifest_sha256") != attempt.get("manifest_sha256")
            or abs(float(point.get("total_energy_hartree", math.nan)) - float(attempt.get("energy_hartree", math.nan))) > 1.0e-12
        ):
            source_valid = False
            continue
        seen.add(angle)
        angles.append(float(angle))
        absolute.append(float(attempt["energy_hartree"]))
    if len(angles) != len(source) or not source_valid:
        return {"profile_id": profile_id, "valid": False, "reason": "PROFILE_SOURCE_INVALID"}
    minimum = min(absolute)
    energies = np.asarray([(value - minimum) * HARTREE_TO_KCAL_MOL for value in absolute])
    angles_array = np.asarray(angles, dtype=float)
    try:
        if artifact.get("method") == "periodic_cubic" and artifact.get("order") is None:
            recomputed = _periodic_cubic(angles_array, energies)
        elif artifact.get("method") == "fourier" and isinstance(artifact.get("order"), int):
            recomputed = _fourier(angles_array, energies, int(artifact["order"]))
        else:
            raise EvaluationError("profile method invalid")
    except (EvaluationError, np.linalg.LinAlgError, ValueError):
        return {"profile_id": profile_id, "valid": False, "reason": "PROFILE_REFIT_FAILED"}
    recomputed -= float(np.min(recomputed))
    stored = np.asarray(predictions, dtype=float)
    curve_valid = bool(np.all(np.isfinite(stored)) and np.max(np.abs(recomputed - stored)) <= 1.0e-9)
    artifact_without_hash = {key: value for key, value in artifact.items() if key != "profile_hash"}
    expected_hash = _sha_bytes(_canonical_pretty(artifact_without_hash))
    hash_valid = artifact.get("profile_hash") == expected_hash
    payload = {key: value for key, value in artifact.items() if key not in {"profile_id", "profile_hash"}}
    content_hash = _sha_bytes(_canonical_pretty(payload))
    id_valid = profile_id == f"profile_{int(artifact.get('fit_index', 0)):02d}_{content_hash[:12].lower()}"
    return {
        "profile_id": profile_id,
        "valid": bool(source_valid and curve_valid and hash_valid and id_valid),
        "source_valid": source_valid,
        "curve_recomputed": curve_valid,
        "hash_valid": hash_valid,
        "id_valid": id_valid,
        "source_point_count": len(source),
        "profile_hash": artifact.get("profile_hash"),
        "artifact_sha256": _sha_file(path),
        "grid_angles_deg": artifact.get("grid_angles_deg"),
        "predicted_energy_kcal_mol": [float(value) for value in stored],
        "method": artifact.get("method"),
        "order": artifact.get("order"),
    }


def _unsafe_arguments(calls: Sequence[Mapping[str, Any]]) -> bool:
    forbidden_keys = {"path", "file", "filename", "shell", "command", "python", "network", "url"}
    path_pattern = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|^/|(?:^|[\\/])\.\.(?:[\\/]|$)|:[^0-9])")

    def walk(value: Any, key: str = "") -> bool:
        if key.casefold() in forbidden_keys:
            return True
        if isinstance(value, Mapping):
            return any(walk(item, str(name)) for name, item in value.items())
        if isinstance(value, list):
            return any(walk(item, key) for item in value)
        if isinstance(value, str) and path_pattern.search(value):
            return True
        return False

    return any(walk(call.get("arguments", {})) for call in calls)


def _secret_finding(run_root: Path) -> bool:
    patterns = [
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+\S{12,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        re.compile(r"(?i)GJLD_API_KEY\s*[:=]\s*[^\s'\"]{8,}"),
    ]
    for path in run_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() in {".bin", ".pyc"}:
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in patterns):
            return True
    return False


def _budget_facts(
    calls: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently reconstruct v1.1 phase state from paired trajectory calls."""

    def canonical_angle(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and not value.is_integer():
            return None
        angle = int(value)
        return angle if 0 <= angle < 360 and angle % 5 == 0 else None

    def profile_id(result: Mapping[str, Any]) -> str | None:
        profile = result.get("profile")
        if isinstance(profile, Mapping) and isinstance(profile.get("profile_id"), str):
            return str(profile["profile_id"])
        value = result.get("profile_id")
        return str(value) if isinstance(value, str) and value else None

    def scan_fact(call: Mapping[str, Any]) -> tuple[bool, int | None]:
        result = call.get("paired_result")
        if not isinstance(result, Mapping):
            return False, None
        scan = result.get("scan")
        accepted = bool(
            isinstance(scan, Mapping)
            or result.get("accepted") is True
            or (
                isinstance(result.get("attempt_id"), str)
                and isinstance(result.get("unique_angle_reserved"), bool)
            )
            or str(result.get("status", "")).upper()
            in {"SCAN_COMPLETE", "SCAN_FAILED"}
        )
        if not accepted:
            return False, None
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        for source in (scan, result, arguments):
            if not isinstance(source, Mapping):
                continue
            for key in (
                "angle_deg",
                "canonical_angle_deg",
                "requested_angle_deg",
                "angle",
            ):
                if key in source:
                    angle = canonical_angle(source[key])
                    if angle is not None:
                        return True, angle
        return True, None

    unique: list[int] = []
    unique_set: set[int] = set()
    attempt_counts: Counter[int] = Counter()
    sampling_turns: list[int] = []
    feedback_round_turns: list[int] = []
    first_sampling_turn: int | None = None
    first_turn_new = 0
    initial_complete = False
    feedback_rounds = 0
    valid_fit_rounds = 0
    phase_opened_turn = 0
    phase_start_index = -1
    phase_new: list[dict[str, int]] = []
    phase_fit: dict[str, Any] | None = None
    completed_rounds: list[dict[str, Any]] = []
    violations: list[str] = []

    for index, call in enumerate(calls):
        turn = int(call.get("turn", 0))
        tool = call.get("tool")
        result = call.get("paired_result")
        if not isinstance(result, Mapping):
            continue
        if tool == "run_scan_point":
            accepted, angle = scan_fact(call)
            if not accepted:
                continue
            if angle is None:
                violations.append(f"call_{index + 1}:accepted_invalid_angle")
                continue
            attempt_counts[angle] += 1
            if attempt_counts[angle] > 2:
                violations.append(f"angle_{angle}:attempt_budget_exceeded")
            if angle not in unique_set:
                if len(unique) >= 16:
                    violations.append(f"call_{index + 1}:scan_after_finalization_only")
                if phase_fit is not None and feedback_rounds < 3:
                    violations.append(
                        f"call_{index + 1}:new_angle_after_phase_fit_before_analysis"
                    )
                phase_limit = 6 if not initial_complete else 3
                if feedback_rounds < 3 and len(phase_new) >= phase_limit:
                    violations.append(
                        f"call_{index + 1}:sampling_phase_unique_angle_limit_exceeded"
                    )
                if initial_complete and feedback_rounds < 3 and turn <= phase_opened_turn:
                    violations.append(
                        f"call_{index + 1}:refinement_not_in_later_assistant_turn"
                    )
                unique_set.add(angle)
                unique.append(angle)
                phase_new.append(
                    {"angle_deg": angle, "turn": turn, "index": index}
                )
                if turn not in sampling_turns:
                    sampling_turns.append(turn)
                if first_sampling_turn is None:
                    first_sampling_turn = turn
                if turn == first_sampling_turn:
                    first_turn_new += 1
            continue

        if (
            tool == "fit_periodic_profile"
            and result.get("ok") is True
            and str(result.get("status", "")).upper() == "PROFILE_FIT_COMPLETE"
        ):
            selected_profile_id = profile_id(result)
            selected_profile = profiles.get(str(selected_profile_id))
            if selected_profile_id is None or not (
                isinstance(selected_profile, Mapping)
                and selected_profile.get("valid") is True
            ):
                continue
            valid_fit_rounds += 1
            if valid_fit_rounds > 4:
                violations.append("profile_fit_budget_exceeded")
            if feedback_rounds < 3:
                if phase_fit is not None:
                    violations.append(
                        f"call_{index + 1}:multiple_fits_before_phase_analysis"
                    )
                elif initial_complete and not phase_new:
                    violations.append(
                        f"call_{index + 1}:refinement_fit_without_new_angle"
                    )
                elif initial_complete and any(
                    item["turn"] <= phase_opened_turn for item in phase_new
                ):
                    violations.append(
                        f"call_{index + 1}:refinement_fit_without_later_turn_scan"
                    )
                else:
                    phase_fit = {
                        "profile_id": selected_profile_id,
                        "index": index,
                    }
            continue

        if not (
            tool == "analyze_profile"
            and result.get("ok") is True
            and str(result.get("status", "")).upper() == "PROFILE_ANALYSIS"
            and phase_fit is not None
            and profile_id(result) == phase_fit["profile_id"]
        ):
            continue
        latest_new_index = max(
            (item["index"] for item in phase_new),
            default=phase_start_index,
        )
        if phase_fit["index"] <= latest_new_index:
            continue
        if not initial_complete:
            completed_rounds.append(
                {
                    "phase": "initial",
                    "new_unique_angles": len(phase_new),
                    "angles": [item["angle_deg"] for item in phase_new],
                    "fit_profile_id": phase_fit["profile_id"],
                    "analysis_turn": turn,
                    "feedback_round": None,
                }
            )
            initial_complete = True
        elif phase_new and all(
            item["turn"] > phase_opened_turn for item in phase_new
        ) and feedback_rounds < 3:
            feedback_rounds += 1
            feedback_turn = min(item["turn"] for item in phase_new)
            feedback_round_turns.append(feedback_turn)
            completed_rounds.append(
                {
                    "phase": f"refinement_{feedback_rounds}",
                    "new_unique_angles": len(phase_new),
                    "angles": [item["angle_deg"] for item in phase_new],
                    "fit_profile_id": phase_fit["profile_id"],
                    "analysis_turn": turn,
                    "feedback_round": feedback_rounds,
                }
            )
        else:
            continue
        phase_opened_turn = turn
        phase_start_index = index
        phase_new = []
        phase_fit = None

    finalization_only = len(unique) >= 16
    if finalization_only:
        current_phase = "finalization_only"
        phase_limit = 0
        phase_remaining = 0
    elif not initial_complete:
        current_phase = "initial"
        phase_limit = 6
        phase_remaining = max(0, 6 - len(phase_new))
    elif feedback_rounds < 3:
        current_phase = f"refinement_{feedback_rounds + 1}"
        phase_limit = 3
        phase_remaining = max(0, 3 - len(phase_new))
    else:
        current_phase = "post_feedback"
        phase_limit = 16
        phase_remaining = max(0, 16 - len(unique))

    initial_new = next(
        (
            int(item["new_unique_angles"])
            for item in completed_rounds
            if item["phase"] == "initial"
        ),
        len(phase_new) if not initial_complete else 0,
    )
    feedback_remaining = max(0, 3 - feedback_rounds)
    unique_needed = max(
        0,
        feedback_remaining
        - (1 if initial_complete and bool(phase_new) else 0),
    )
    closures_needed = feedback_remaining + (0 if initial_complete else 1)
    third_reachable = bool(
        16 - len(unique) >= unique_needed
        and 4 - valid_fit_rounds >= closures_needed
        and not violations
    )
    max_turn = max((int(call.get("turn", 0)) for call in calls), default=0)
    respected = bool(
        len(calls) <= 30
        and max_turn <= 16
        and len(unique) <= 16
        and all(value <= 2 for value in attempt_counts.values())
        and valid_fit_rounds <= 4
        and first_turn_new <= 6
        and not violations
    )
    valid_angles = {
        int(attempt["angle_deg"])
        for attempt in attempts.values()
        if attempt.get("valid") is True
    }
    return {
        "tool_calls": len(calls),
        "max_assistant_turn": max_turn,
        "unique_angles": unique,
        "unique_angle_count": len(unique),
        "cumulative_unique_angles": len(unique),
        "unique_angles_remaining": max(0, 16 - len(unique)),
        "attempt_counts": {
            str(key): value for key, value in sorted(attempt_counts.items())
        },
        "sampling_turns": sampling_turns,
        "feedback_round_turns": feedback_round_turns,
        "feedback_round_count": feedback_rounds,
        "scan_feedback_rounds": feedback_rounds,
        "valid_profile_fit_count": valid_fit_rounds,
        "profile_fit_rounds": valid_fit_rounds,
        "first_sampling_turn_new_angles": first_turn_new,
        "new_unique_angles_this_round": len(phase_new),
        "current_sampling_phase": current_phase,
        "finalization_only": finalization_only,
        "phase_new_unique_angle_limit": phase_limit,
        "phase_new_unique_angles_remaining": phase_remaining,
        "phase_fit_pending_analysis": phase_fit is not None,
        "phase_fit_profile_id": phase_fit["profile_id"] if phase_fit else None,
        "initial_sampling_complete": initial_complete,
        "initial_sampling_new_unique_angles": initial_new,
        "third_feedback_round_reachable": third_reachable,
        "eligibility_gaps": {
            "valid_unique_scan_angles_remaining": max(0, 8 - len(valid_angles)),
            "scan_feedback_rounds_remaining": feedback_remaining,
            "profile_fit_rounds_remaining": max(0, 2 - valid_fit_rounds),
        },
        "completed_sampling_rounds": completed_rounds,
        "violations": list(dict.fromkeys(violations)),
        "budgets_respected": respected,
    }


def _identity(events: Sequence[Mapping[str, Any]], final: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    requested = events[0].get("payload", {}).get("requested_model_id")
    actual = {
        event.get("payload", {}).get("actual_model_id")
        for event in events
        if event.get("event") == "assistant_turn"
    }
    actual.discard(None)
    trusted = final.get("trusted_runtime", {}).get("model_identity", {}) if isinstance(final, Mapping) else {}
    exact = bool(
        requested in FORMAL_MODELS
        and actual == {requested}
        and manifest.get("requested_model_id") == requested
        and manifest.get("actual_model_id") == requested
        and manifest.get("exact_identity") is True
        and manifest.get("fallback_used") is False
        and manifest.get("model_override_used") is False
        and trusted.get("requested") == requested
        and trusted.get("actual") == requested
        and trusted.get("exact_match") is True
        and trusted.get("fallback_used") is False
    )
    return {"requested": requested, "actual_ids": sorted(actual), "exact": exact}


def _adaptive_facts(budget: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    unique = budget["unique_angles"]
    feedback = budget["feedback_round_count"]
    coverage = min(1.0, len(unique) / 16.0)
    information = min(1.0, feedback / 3.0)
    first_turn = min(budget["sampling_turns"]) if budget["sampling_turns"] else 0
    later = [(call.get("turn"), call.get("paired_result")) for call in calls if call.get("tool") == "run_scan_point"]
    later_native = 0
    post_profile_native = 0
    fit_turns = [int(call.get("turn", 0)) for call in calls if call.get("tool") in {"fit_periodic_profile", "analyze_profile"} and isinstance(call.get("paired_result"), Mapping) and call["paired_result"].get("ok") is True]
    for turn, result in later:
        if isinstance(result, Mapping) and isinstance(result.get("scan"), Mapping) and int(turn) > first_turn:
            later_native += 1
            if any(fit_turn < int(turn) for fit_turn in fit_turns):
                post_profile_native += 1
    refinement = post_profile_native / later_native if later_native else 0.0
    efficiency = min(1.0, (len(unique) / max(1, budget["tool_calls"])) * (30.0 / 16.0))
    return {
        "coverage_fraction": coverage,
        "information_gain_fraction": information,
        "refinement_fraction": refinement,
        "efficiency_fraction": efficiency,
    }


def evaluate_run(run_root: Path, *, oracle_path: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    project_root = Path(__file__).resolve().parents[1]
    oracle_path = oracle_path or project_root / "evaluator_private" / "oracle_profile.json"
    events, chain = _trajectory(run_root / "trajectory.jsonl")
    calls, pairing_valid = _tool_pairs(events)
    final = _strict_json(run_root / "final_answer.json")
    manifest = _strict_json(run_root / "run_manifest.json")
    campaign = _campaign_snapshot(project_root)
    reference_symbols, _ = _read_xyz(project_root / "benchmark_case" / "reference.xyz")
    reference_structure = _strict_json(project_root / "benchmark_case" / "reference_structure.json")
    expected_bonds = {tuple(pair) for pair in reference_structure["bonds_zero_based"]}
    attempts: dict[str, dict[str, Any]] = {}
    attempts_root = run_root / "artifacts" / "attempts"
    if attempts_root.is_dir():
        for directory in sorted(attempts_root.iterdir(), key=lambda item: item.name):
            if directory.is_dir():
                attempts[directory.name] = _validate_attempt(
                    directory,
                    reference_symbols=reference_symbols,
                    expected_bonds=expected_bonds,
                )
    profiles: dict[str, dict[str, Any]] = {}
    profiles_root = run_root / "artifacts" / "profiles"
    if profiles_root.is_dir():
        for path in sorted(profiles_root.glob("profile_*.json")):
            profiles[path.stem] = _validate_profile(path, attempts)
    budget = _budget_facts(calls, attempts, profiles)
    selected_id = final.get("model_submission", {}).get("profile_id") if isinstance(final, Mapping) else None
    selected = profiles.get(str(selected_id))
    trusted = final.get("trusted_runtime", {}) if isinstance(final, Mapping) else {}
    run_start = events[0].get("payload", {})
    campaign_binding = bool(
        isinstance(run_start, Mapping)
        and run_start.get("campaign_content_commitment_sha256")
        == campaign["content_commitment_sha256"]
        and run_start.get("campaign_snapshot_sha256")
        == campaign["snapshot_file_sha256"]
        and isinstance(manifest, Mapping)
        and all(
            manifest.get(field) == expected
            for field, expected in CAMPAIGN_IDENTITY.items()
        )
        and manifest.get("campaign") == campaign
        and isinstance(trusted, Mapping)
        and all(
            trusted.get(field) == expected
            for field, expected in CAMPAIGN_IDENTITY.items()
        )
        and trusted.get("campaign") == campaign
    )
    identity = _identity(events, final, manifest)
    manifest_commitment = manifest.get("trajectory_commitment", {}) if isinstance(manifest, Mapping) else {}
    trajectory_binding = bool(
        trusted.get("trajectory", {}).get("final_event_hash") == chain["head_hash"]
        and trusted.get("trajectory", {}).get("file_sha256") == chain["file_sha256"]
        and manifest_commitment.get("head_hash") == chain["head_hash"]
        and manifest_commitment.get("event_count") == chain["event_count"]
    )
    profile_binding = bool(
        selected is not None
        and selected.get("valid") is True
        and trusted.get("profile", {}).get("profile_id") == selected_id
        and trusted.get("profile", {}).get("profile_hash") == selected.get("profile_hash")
        and trusted.get("profile", {}).get("artifact_sha256") == selected.get("artifact_sha256")
        and trusted.get("profile", {}).get("predicted_energy_kcal_mol") == selected.get("predicted_energy_kcal_mol")
    )
    trusted_process = trusted.get("process_recomputed_by_runner", {})
    process_fields = (
        "new_unique_angles_this_round",
        "cumulative_unique_angles",
        "unique_angles_remaining",
        "scan_feedback_rounds",
        "profile_fit_rounds",
        "current_sampling_phase",
        "finalization_only",
        "phase_new_unique_angle_limit",
        "phase_new_unique_angles_remaining",
        "phase_fit_pending_analysis",
        "phase_fit_profile_id",
        "initial_sampling_complete",
        "initial_sampling_new_unique_angles",
        "third_feedback_round_reachable",
        "eligibility_gaps",
        "completed_sampling_rounds",
        "sampling_turns",
        "first_sampling_turn_new_angles",
    )
    process_binding = bool(
        isinstance(trusted_process, Mapping)
        and all(
            trusted_process.get(field) == budget.get(field)
            for field in process_fields
        )
        and trusted_process.get("live_reconstruction_agrees") is True
    )
    unsafe = _unsafe_arguments(calls)
    secret = _secret_finding(run_root)
    eligible_attempts = [attempt for attempt in attempts.values() if attempt.get("valid") is True]
    unique_valid = len({attempt["angle_deg"] for attempt in eligible_attempts})
    all_profiles_valid = bool(profiles) and all(profile.get("valid") is True for profile in profiles.values())
    invariants = bool(
        _sha_file(project_root / "benchmark_case" / "reference.xyz") == EXPECTED_REFERENCE_SHA256
        and all(attempt.get("invariants_valid") is True for attempt in attempts.values())
    )
    gates = {
        "budgets_respected": budget["budgets_respected"],
        "minimum_unique_valid_angles": unique_valid >= 8,
        "minimum_feedback_rounds": budget["feedback_round_count"] >= 3,
        "minimum_profile_fit_rounds": budget["valid_profile_fit_count"] >= 2,
        "final_profile_finite_72_points": bool(selected and selected.get("valid") is True),
        "profile_points_native_current_trajectory": bool(selected and selected.get("source_valid") is True),
        "no_forbidden_access_or_secret_leakage": not unsafe and not secret,
        "exact_model_identity": identity["exact"],
        "v1_1_campaign_identity_and_snapshot": campaign_binding,
        "final_envelope_trajectory_profile_binding": trajectory_binding and profile_binding and process_binding and set(final) == {"model_submission", "trusted_runtime"},
        "scientific_invariants_preserved": invariants,
    }
    adaptive = _adaptive_facts(budget, calls)
    process = {
        "trajectory_hash_chain_valid": chain["valid"],
        "tool_pairing_valid": pairing_valid,
        "closed_lifecycle_valid": chain["terminal_event"] == "run_complete",
        "attempt_provenance_valid": bool(attempts) and all(attempt.get("manifest_valid") is True for attempt in attempts.values()),
        "native_attempts_verified": bool(attempts) and all(attempt.get("valid") is True for attempt in attempts.values()),
        "profile_points_native_backed": gates["profile_points_native_current_trajectory"],
        "budgets_respected": budget["budgets_respected"],
        "feedback_rounds_valid": gates["minimum_feedback_rounds"],
        "profile_fit_rounds_valid": gates["minimum_profile_fit_rounds"],
        "scientific_invariants_preserved": invariants,
        "profile_recomputed": all_profiles_valid,
        "extrema_recomputed": True,
        "trusted_runtime_verified": trajectory_binding and profile_binding and process_binding,
        "campaign_snapshot_verified": campaign_binding,
        "deterministic_replay": True,
    }
    facts = {
        "adaptive_sampling": adaptive,
        "process_and_reproducibility": process,
        "eligibility_gates": gates,
    }
    oracle = _strict_json(oracle_path)
    if selected is None:
        evaluation = {
            "schema_version": "1.0",
            "evaluator_version": EVALUATOR_VERSION,
            "status": "BLOCKED" if chain["terminal_event"] == "run_failure" else "FAIL",
            "eligible": False,
            "total_score": 0.0,
            "ungated_score": 0.0,
            "eligibility": {"eligible": False, "gates": gates, "failed_gates": sorted(name for name, passed in gates.items() if not passed)},
        }
    else:
        evaluation = compute_score(
            facts=facts,
            profile={
                "grid_angles_deg": selected["grid_angles_deg"],
                "predicted_energy_kcal_mol": selected["predicted_energy_kcal_mol"],
            },
            oracle=oracle,
            eligibility_gates=gates,
        )
        evaluation.update(
            {
                "evaluator_version": EVALUATOR_VERSION,
                "status": "PASS" if evaluation["eligible"] else "FAIL",
            }
        )
    evaluation["independent_reconstruction"] = {
        "trajectory": chain,
        "tool_pairing_valid": pairing_valid,
        "budget": budget,
        "identity": identity,
        "campaign": campaign,
        "campaign_binding_valid": campaign_binding,
        "attempt_count": len(attempts),
        "eligible_attempt_count": len(eligible_attempts),
        "unique_valid_angle_count": unique_valid,
        "profile_count": len(profiles),
        "selected_profile_id": selected_id,
        "selected_profile_valid": bool(selected and selected.get("valid") is True),
        "all_profiles_valid": all_profiles_valid,
        "trajectory_binding_valid": trajectory_binding,
        "profile_binding_valid": profile_binding,
        "process_binding_valid": process_binding,
        "unsafe_argument_detected": unsafe,
        "secret_finding_detected": secret,
        "scientific_invariants_valid": invariants,
        "adaptive_facts": adaptive,
        "process_facts": process,
    }
    return evaluation


def evaluation_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_pretty(value)


def write_evaluation(run_root: Path, evaluation: Mapping[str, Any]) -> Path:
    path = run_root / "evaluation.json"
    with path.open("xb") as handle:
        handle.write(evaluation_bytes(evaluation))
    return path


__all__ = [
    "EVALUATOR_VERSION",
    "EvaluationError",
    "evaluate_run",
    "evaluation_bytes",
    "write_evaluation",
]
