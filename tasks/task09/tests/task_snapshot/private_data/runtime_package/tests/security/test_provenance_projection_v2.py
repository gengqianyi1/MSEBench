from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from package_tools.provenance import (
    ATTEMPT_WORKDIR_TOKEN,
    EXPECTED_XTB_SHA256,
    ProvenanceProjectionError,
    XTB_EXECUTABLE_TOKEN,
    build_package_safe_projection,
    canonical_json_bytes,
    find_host_path_occurrences,
    projection_host_path_findings,
    sha256_bytes,
    sha256_file,
    validate_package_safe_projection,
    write_package_safe_projection,
)


REFERENCE_SHA256 = "9" * 64


def _host_executable() -> str:
    # Construct at runtime so the released test source contains no real or
    # synthetic host-absolute path literal.
    return "C" + ":" + "\\" + "\\".join(
        ("Users", "fixture-user", "conda", "Library", "bin", "xtb.exe")
    )


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "attempt_manifest.json":
            entries.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    (root / "attempt_manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0",
                "algorithm": "SHA-256",
                "files": entries,
            }
        )
    )


def _rewrite_result(root: Path, value: dict) -> None:
    (root / "result.json").write_bytes(canonical_json_bytes(value))
    _write_manifest(root)


def _attempt(root: Path, angle: int) -> Path:
    root.mkdir(parents=True)
    executable = _host_executable()
    (root / "input.xyz").write_text(
        f"2\nangle {angle}\nC 0 0 0\nH 0 0 1\n",
        encoding="ascii",
        newline="\n",
    )
    (root / "xcontrol.inp").write_text(
        f"$constrain\n  dihedral: 6, 7, 8, 9, {angle}.0\n$end\n",
        encoding="ascii",
        newline="\n",
    )
    (root / "xtbopt.xyz").write_text(
        f"2\nenergy: {-35.0 + angle / 100000.0:.12f}\n"
        "C 0 0 0\nH 0 0 1\n",
        encoding="ascii",
        newline="\n",
    )
    (root / "stdout.log").write_text(
        "GEOMETRY OPTIMIZATION CONVERGED\n"
        f"program call : {executable} input.xyz --gfn 2 --chrg 0 "
        "--uhf 0 --opt tight --input xcontrol.inp\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "stderr.log").write_text("", encoding="utf-8", newline="\n")
    input_sha = sha256_file(root / "input.xyz")
    result = {
        "schema_version": "1.0",
        "angle_deg": angle,
        "method": "GFN2-xTB",
        "phase": "gas",
        "charge": 0,
        "uhf": 0,
        "threads": 1,
        "optimization": "tight",
        "constraint": {
            "type": "dihedral",
            "atom_maps": [6, 7, 8, 9],
            "atom_indices_one_based": [6, 7, 8, 9],
            "target_deg": angle,
            "force_constant_au": 1.0,
        },
        "reference_sha256": REFERENCE_SHA256,
        "prepared_input": {
            "angle_deg": angle,
            "input_sha256": input_sha,
            "reference_sha256": REFERENCE_SHA256,
            "uff_angle_error_deg": 0.01,
            "uff_final_angle_deg": float(angle),
        },
        "argv": [
            executable,
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
        ],
        "shell": False,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "xtb": {
            "binary_sha256": EXPECTED_XTB_SHA256,
            "executable": executable,
            "version": "6.7.1",
            "version_exit_code": 0,
        },
        "timeout_seconds": 180,
        "timed_out": False,
        "duration_seconds": 0.25,
        "return_code": 0,
        "converged": True,
        "total_energy_hartree": -35.0 + angle / 100000.0,
        "final_angle_deg": float(angle),
        "angle_error_deg": 0.0,
        "requested_final_angle_tolerance_deg": 2.0,
        "structure_validation": {
            "atom_count_ok": True,
            "atom_order_ok": True,
            "composition_ok": True,
            "formal_charge_ok": True,
            "connectivity_ok": True,
            "expected_bond_count": 1,
            "inferred_bond_count": 1,
            "missing_bonds": [],
            "unexpected_bonds": [],
        },
        "parse_error": None,
        "profile_eligible": True,
    }
    (root / "result.json").write_bytes(canonical_json_bytes(result))
    _write_manifest(root)
    return root


def _all_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _rehash_projection(payload: dict) -> None:
    payload.pop("projection_sha256", None)
    payload["projection_sha256"] = sha256_bytes(canonical_json_bytes(payload))


def test_historical_twelve_match_shape_projects_to_zero_host_paths(
    tmp_path: Path,
) -> None:
    attempts = [
        _attempt(tmp_path / f"angle_{angle:03d}" / "attempt_001", angle)
        for angle in (0, 90, 180, 270)
    ]
    findings = [
        item
        for attempt in attempts
        for item in find_host_path_occurrences(attempt)
    ]
    assert len(findings) == 12
    assert {item["rule"] for item in findings} == {
        "windows_drive_absolute"
    }
    assert {
        item["path"] for item in findings
    } == {"result.json", "stdout.log"}
    assert sum(item["path"] == "result.json" for item in findings) == 8
    assert sum(item["path"] == "stdout.log" for item in findings) == 4

    for attempt in attempts:
        before = _all_file_hashes(attempt)
        first = build_package_safe_projection(attempt)
        second = build_package_safe_projection(attempt)
        assert canonical_json_bytes(first) == canonical_json_bytes(second)
        assert validate_package_safe_projection(first)["passed"] is True
        assert projection_host_path_findings(first) == []
        assert first["redaction"]["raw_host_path_occurrence_count"] == 3
        assert first["redaction"]["raw_files_with_host_paths"] == [
            "result.json",
            "stdout.log",
        ]
        assert first["execution"]["host_path_redacted"] == XTB_EXECUTABLE_TOKEN
        assert (
            first["execution"]["working_directory_redacted"]
            == ATTEMPT_WORKDIR_TOKEN
        )
        assert first["execution"]["argv_template"][0] == XTB_EXECUTABLE_TOKEN
        source_hashes = {
            item["path"]: item["sha256"]
            for item in first["source_binding"]["files"]
        }
        assert (
            first["artifact_hashes"]["input_sha256"]
            == source_hashes["input.xyz"]
        )
        assert (
            first["artifact_hashes"]["output_sha256"]
            == source_hashes["xtbopt.xyz"]
        )
        encoded = canonical_json_bytes(first)
        assert b"fixture-user" not in encoded
        assert b"Users" not in encoded
        assert _all_file_hashes(attempt) == before


def test_unknown_path_like_raw_field_is_rejected_closed_schema(
    tmp_path: Path,
) -> None:
    attempt = _attempt(tmp_path / "attempt", 0)
    result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    result["debug_path"] = _host_executable()
    _rewrite_result(attempt, result)
    with pytest.raises(
        ProvenanceProjectionError, match="RAW_RESULT_UNKNOWN_FIELDS"
    ):
        build_package_safe_projection(attempt)


def test_raw_manifest_hash_tamper_is_rejected(tmp_path: Path) -> None:
    attempt = _attempt(tmp_path / "attempt", 90)
    with (attempt / "stdout.log").open("ab") as handle:
        handle.write(b"tamper\n")
    with pytest.raises(
        ProvenanceProjectionError,
        match="RAW_ATTEMPT_MANIFEST_(SIZE|HASH)_MISMATCH",
    ):
        build_package_safe_projection(attempt)


def test_model_or_executable_override_in_argv_is_rejected(
    tmp_path: Path,
) -> None:
    attempt = _attempt(tmp_path / "attempt", 180)
    result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    result["argv"][3] = "1"
    _rewrite_result(attempt, result)
    with pytest.raises(
        ProvenanceProjectionError, match="RAW_RESULT_ARGV_INVARIANT"
    ):
        build_package_safe_projection(attempt)


def test_projection_schema_rejects_unknown_or_host_path_field(
    tmp_path: Path,
) -> None:
    projection = build_package_safe_projection(
        _attempt(tmp_path / "attempt", 270)
    )
    unknown = copy.deepcopy(projection)
    unknown["scientific_record"]["debug_path"] = _host_executable()
    _rehash_projection(unknown)
    with pytest.raises(
        ProvenanceProjectionError, match="PROJECTION_SCIENCE_SCHEMA"
    ):
        validate_package_safe_projection(unknown)

    hidden = copy.deepcopy(projection)
    hidden["scientific_record"]["method"] = _host_executable()
    _rehash_projection(hidden)
    with pytest.raises(ProvenanceProjectionError):
        validate_package_safe_projection(hidden)


def test_projection_hash_and_artifact_binding_are_independently_checked(
    tmp_path: Path,
) -> None:
    projection = build_package_safe_projection(
        _attempt(tmp_path / "attempt", 0)
    )
    bad_hash = copy.deepcopy(projection)
    bad_hash["projection_sha256"] = "0" * 64
    with pytest.raises(
        ProvenanceProjectionError, match="PROJECTION_HASH_MISMATCH"
    ):
        validate_package_safe_projection(bad_hash)

    bad_artifact = copy.deepcopy(projection)
    bad_artifact["artifact_hashes"]["stdout_sha256"] = "A" * 64
    _rehash_projection(bad_artifact)
    with pytest.raises(
        ProvenanceProjectionError,
        match="PROJECTION_ARTIFACT_BINDING_MISMATCH",
    ):
        validate_package_safe_projection(bad_artifact)


def test_write_projection_is_exclusive_and_does_not_modify_raw_attempt(
    tmp_path: Path,
) -> None:
    attempt = _attempt(tmp_path / "raw" / "attempt", 90)
    before = _all_file_hashes(attempt)
    destination = tmp_path / "release" / "provenance.json"
    payload = write_package_safe_projection(attempt, destination)
    assert destination.read_bytes() == canonical_json_bytes(payload)
    assert _all_file_hashes(attempt) == before
    with pytest.raises(
        ProvenanceProjectionError, match="PROJECTION_DESTINATION_EXISTS"
    ):
        write_package_safe_projection(attempt, destination)


def test_projection_destination_inside_raw_attempt_is_rejected(
    tmp_path: Path,
) -> None:
    attempt = _attempt(tmp_path / "attempt", 0)
    with pytest.raises(
        ProvenanceProjectionError,
        match="PROJECTION_DESTINATION_INSIDE_RAW_ATTEMPT",
    ):
        write_package_safe_projection(
            attempt, attempt / "package-safe-provenance.json"
        )


def test_unc_posix_and_file_uri_detection_returns_no_values() -> None:
    unc = "\\" + "\\" + "server" + "\\" + "share" + "\\" + "secret"
    value = {
        "unc": unc,
        "posix": "/" + "home/fixture/private.txt",
        "uri": "file:" + "/" * 3 + "tmp/private.txt",
    }
    findings = projection_host_path_findings(value)
    assert {item["rule"] for item in findings} == {
        "file_uri",
        "posix_host_absolute",
        "windows_unc_or_device",
    }
    assert all(set(item) == {"location", "rule"} for item in findings)
