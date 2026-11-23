"""Task-limited parser for raw xTB output directories.

Only fields required by the three benchmark tasks are returned.  The parser is
purely observational: it does not recommend next actions, plan retries, or
compute task-level answers.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import geometry_dihedral, read_xyz, read_xyz_frames
from common.io_utils import native_path, sha256_file

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_VERSION = re.compile(r"(?:xtb\s+version|version)\s*[:=]?\s*v?([0-9]+(?:\.[0-9]+){1,2})", re.I)


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _request(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "request.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _arguments(request: dict[str, Any]) -> list[str]:
    value = request.get("arguments")
    return [str(item) for item in value] if isinstance(value, list) else []


def _argument_value(arguments: list[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _parse_protocol(text: str, request: dict[str, Any]) -> dict[str, Any]:
    arguments = _arguments(request)
    version_match = _VERSION.search(text)
    version = version_match.group(1) if version_match else None
    gfn_value = _argument_value(arguments, "--gfn")
    if gfn_value is None:
        match = re.search(r"(?:Hamiltonian\s*:\s*GFN|GFN)(\d)(?:-xTB)?", text, re.I)
        gfn_value = match.group(1) if match else None
    model = "none"
    solvent = None
    for option, name in (("--alpb", "alpb"), ("--gbsa", "gbsa")):
        value = _argument_value(arguments, option)
        if value is not None:
            model, solvent = name, value
            break
    if solvent is None:
        # Do not interpret prose such as "ALPB and GBSA" in the xTB banner as
        # an active solvent. A stdout-only fallback must contain an explicitly
        # labelled solvent; request arguments are authoritative when present.
        match = re.search(r"\b(ALPB|GBSA)\s+solvent\s*[:=]\s*([A-Za-z0-9_.-]+)", text, re.I)
        if match:
            model, solvent = match.group(1).lower(), match.group(2)
    charge_value = _argument_value(arguments, "--chrg") or _argument_value(arguments, "--charge")
    uhf_value = _argument_value(arguments, "--uhf")
    if charge_value is None:
        match = re.search(r"(?:molecular\s+charge|net\s+charge)\s*[:=]?\s*(-?\d+)", text, re.I)
        charge_value = match.group(1) if match else None
    if uhf_value is None:
        match = re.search(r"(?:number\s+of\s+unpaired\s+electrons|unpaired\s+electrons|uhf)\s*[:=]?\s*(\d+)", text, re.I)
        uhf_value = match.group(1) if match else None
    temperature = None
    match = re.search(rf"temperature\s*[:=]?\s*({_FLOAT})\s*K", text, re.I)
    if match:
        temperature = _as_float(match.group(1))
    rotor = None
    match = re.search(rf"(?:sthr|rotor[^\n]*cutoff)\s*[:=]\s*({_FLOAT})\s*(?:cm(?:\^-?1|-1)?)?", text, re.I)
    if match:
        rotor = _as_float(match.group(1))
    return {
        "version": version,
        "gfn": int(gfn_value) if gfn_value is not None and str(gfn_value).lstrip("+-").isdigit() else None,
        "solvent_model": model,
        "solvent": solvent,
        "charge": int(charge_value) if charge_value is not None and str(charge_value).lstrip("+-").isdigit() else None,
        "uhf": int(uhf_value) if uhf_value is not None and str(uhf_value).lstrip("+-").isdigit() else None,
        "temperature_k": temperature,
        "thermo_rotor_cutoff_cm1": rotor,
        "arguments": arguments,
    }


def _detailed_input_text(job_dir: Path, request: dict[str, Any]) -> tuple[str, str | None]:
    names: list[str] = []
    copied = request.get("detailed_input_job_copy")
    if isinstance(copied, str):
        names.append(copied)
    names.extend(["detailed.inp", "xtb.inp"])
    for name in dict.fromkeys(names):
        path = job_dir / name
        if path.is_file():
            return _read_text(path), name
    return "", None


def _parse_thermochemistry(text: str, detailed_text: str) -> dict[str, Any]:
    temperatures: list[float] = []
    thermo_block = re.search(r"\$thermo\b(.*?)(?:\$end|\Z)", detailed_text, re.I | re.S)
    rotor = None
    if thermo_block:
        block = thermo_block.group(1)
        match = re.search(r"\btemp\s*=\s*([^\n]+)", block, re.I)
        if match:
            # `$thermo temp=298.15, sthr=50` is valid on one line; stop before
            # the next named option while preserving comma-separated temp lists.
            temperature_values = re.split(r",?\s*[A-Za-z][A-Za-z0-9_]*\s*=", match.group(1), maxsplit=1)[0]
            temperatures = [_as_float(value) for value in re.findall(_FLOAT, temperature_values)]
        match = re.search(rf"\bsthr\s*=\s*({_FLOAT})", block, re.I)
        if match:
            rotor = _as_float(match.group(1))
    if not temperatures:
        # xTB marks the temperature selected for the final thermodynamic summary.
        marked = re.findall(rf"^\s*({_FLOAT})\s+.*?\(used\)\s*$", text, re.I | re.M)
        if marked:
            temperatures = [_as_float(marked[-1])]
        else:
            setup = re.search(rf"temperature\s*[:=]?\s*({_FLOAT})\s*K", text, re.I)
            if setup:
                temperatures = [_as_float(setup.group(1))]
            else:
                # Gas-phase xTB Hessian output may omit the setup-style
                # ``Temperature ... K`` line while still printing the selected
                # temperature in the final thermodynamic summary table.
                table_rows = re.findall(
                    rf"^\s*T/K\b[^\n]*\n(?:\s*-+[^\n]*\n)?\s*({_FLOAT})(?=\s)",
                    text,
                    re.I | re.M,
                )
                if table_rows:
                    temperatures = [_as_float(value) for value in table_rows]
    if rotor is None:
        match = re.search(rf"rotor\s+cutoff\s+({_FLOAT})\s*cm", text, re.I)
        if match:
            rotor = _as_float(match.group(1))
    return {
        "performed": bool(re.search(r"Thermodynamic Functions|THERMODYNAMIC", text, re.I)),
        "temperatures_k": temperatures,
        "temperature_k": temperatures[-1] if temperatures else None,
        "rotor_cutoff_cm1": rotor,
        "model": "mRRHO" if re.search(r"mRRHO|G\(RRHO\)|rotor\s+cutoff", text, re.I) else None,
    }


def _last_match(text: str, patterns: list[re.Pattern[str]]) -> float | None:
    values: list[float] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                values.append(_as_float(match.group(1)))
            except (ValueError, OverflowError):
                pass
    return values[-1] if values else None


def _parse_energies(text: str) -> dict[str, float]:
    patterns = {
        "total_energy_hartree": [
            re.compile(rf"TOTAL\s+ENERGY\s+({_FLOAT})", re.I),
            re.compile(rf"\|\s*TOTAL\s+ENERGY\s+({_FLOAT})", re.I),
        ],
        "gibbs_free_energy_hartree": [
            re.compile(rf"TOTAL\s+FREE\s+ENERGY\s+({_FLOAT})", re.I),
            re.compile(rf"Gibbs\s+free\s+energy\s*[:=]?\s*({_FLOAT})", re.I),
        ],
        "enthalpy_hartree": [
            re.compile(rf"TOTAL\s+ENTHALPY\s+({_FLOAT})", re.I),
            re.compile(rf"enthalpy\s*[:=]?\s*({_FLOAT})", re.I),
        ],
        "zpve_hartree": [
            re.compile(rf"\bZPVE\b\s*[:=]?\s*({_FLOAT})", re.I),
            re.compile(rf"zero\s+point\s+energy\s*[:=]?\s*({_FLOAT})", re.I),
        ],
    }
    result: dict[str, float] = {}
    for key, variants in patterns.items():
        value = _last_match(text, variants)
        if value is not None and math.isfinite(value):
            result[key] = value
    return result


def _atom_count_and_linearity(job_dir: Path, request: dict[str, Any]) -> tuple[int | None, bool | None]:
    candidates = [request.get("geometry_job_copy"), "xtbopt.xyz", "xtblast.xyz"]
    for name in candidates:
        if not isinstance(name, str):
            continue
        path = job_dir / name
        if not path.is_file():
            continue
        try:
            geometry = read_xyz(path)
        except ValueError:
            continue
        coordinates = np.asarray(geometry.coordinates, dtype=float)
        if len(coordinates) <= 2:
            return len(coordinates), True
        centered = coordinates - coordinates.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        scale = max(float(singular[0]), 1.0)
        return len(coordinates), bool(float(singular[1]) / scale < 1e-6)
    return None, None


def _stdout_frequency_values(text: str) -> list[float]:
    values: list[float] = []
    in_printout = False
    for line in text.splitlines():
        if "Frequency Printout" in line or "projected vibrational frequencies" in line.lower():
            in_printout = True
            continue
        if in_printout and re.match(r"\s*eigval\s*:", line, re.I):
            values.extend(_as_float(value) for value in re.findall(_FLOAT, line.split(":", 1)[1]))
        elif in_printout and values and re.search(r"reduced masses|IR intensities|output can be read", line, re.I):
            break
    return values


def _vibspectrum_values(raw: str) -> tuple[list[float], list[float]]:
    frequencies: list[float] = []
    intensities: list[float] = []
    # Turbomole-style $vibrational spectrum table written by xTB.  Requiring an
    # integer mode index and numeric frequency prevents arbitrary coordinate
    # rows from being interpreted as modes.
    in_spectrum = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("$vibrational spectrum"):
            in_spectrum = True
            continue
        if in_spectrum and stripped.lower().startswith("$end"):
            break
        if in_spectrum:
            fields = stripped.split()
            if len(fields) >= 3 and fields[0].isdigit():
                try:
                    frequency = _as_float(fields[2])
                except ValueError:
                    continue
                frequencies.append(frequency)
                if len(fields) >= 4:
                    try:
                        intensities.append(_as_float(fields[3]))
                    except ValueError:
                        pass
    # Keep compatibility with the explicit stdout-like representation used in
    # parser fixtures, but only accept labelled lines.
    if not frequencies:
        for line in raw.splitlines():
            if re.match(r"\s*(?:eigval|frequency|freq\.)\s*:", line, re.I):
                frequencies.extend(_as_float(value) for value in re.findall(_FLOAT, line.split(":", 1)[1]))
            elif re.match(r"\s*IR\s+intens", line, re.I):
                intensities.extend(_as_float(value) for value in re.findall(_FLOAT, line.split(":", 1)[1]))
    return frequencies, intensities


def _parse_vibrations(job_dir: Path, text: str, request: dict[str, Any]) -> dict[str, Any]:
    raw_frequencies: list[float] = []
    intensities: list[float] = []
    source = None
    for name in ("vibspectrum", "vibspectrum.out"):
        path = job_dir / name
        if not path.exists():
            continue
        raw_frequencies, intensities = _vibspectrum_values(_read_text(path))
        if raw_frequencies:
            source = name
            break
    if not raw_frequencies:
        raw_frequencies = _stdout_frequency_values(text)
        source = "stdout.txt" if raw_frequencies else None
    atom_count, linear = _atom_count_and_linearity(job_dir, request)
    expected = None if atom_count is None else max(0, 3 * atom_count - (5 if linear else 6))
    # Official stdout contains 3N eigenvalues including the projected zero
    # translations/rotations.  The public API exposes only the expected 3N-5/6
    # vibrational modes while retaining raw count for audit.
    frequencies = list(raw_frequencies)
    if atom_count is not None and expected is not None and len(raw_frequencies) == 3 * atom_count:
        frequencies = raw_frequencies[-expected:] if expected else []
    arguments = _arguments(request)
    requested = any(value in {"--hess", "--ohess", "--bhess"} for value in arguments)
    artifact = any((job_dir / name).is_file() for name in ("hessian", "vibspectrum", "vibspectrum.out"))
    performed = bool(requested and (artifact or "Frequency Printout" in text) and raw_frequencies)
    complete = bool(performed and expected is not None and len(frequencies) == expected)
    return {
        "source_file": source,
        "hessian_requested": requested,
        "hessian_performed": performed,
        "atom_count": atom_count,
        "linear_molecule": linear,
        "expected_mode_count": expected,
        "raw_eigenvalue_count": len(raw_frequencies),
        "parsed_mode_count": len(frequencies),
        "complete": complete,
        "frequencies_cm1": frequencies,
        "ir_intensities": intensities,
        "significant_imaginary_frequencies_cm1": [value for value in frequencies if value < -30.0],
        "small_imaginary_warnings_cm1": [value for value in frequencies if -30.0 <= value < 0.0],
    }


def _parse_optimized_geometry(job_dir: Path) -> dict[str, Any] | None:
    # For optimization jobs prefer xTB's optimized outputs.  A standalone
    # Hessian intentionally evaluates an already-optimized registered input
    # and normally emits no new ``xtbopt.xyz``; in that case the copied input
    # is the exact evaluated geometry and must remain available to provenance
    # and geometry-based graders.
    candidates = [job_dir / "xtbopt.xyz", job_dir / "xtblast.xyz"]
    candidates.extend(sorted(job_dir.glob("input*.xyz")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            geometry = read_xyz(path)
        except ValueError:
            continue
        result = geometry.as_dict()
        result.update({"source_file": path.name, "sha256": sha256_file(path)})
        return result
    return None


def _recursive_find_key(value: Any, normalized_keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in normalized_keys:
                return item
        for item in value.values():
            found = _recursive_find_key(item, normalized_keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _recursive_find_key(item, normalized_keys)
            if found is not None:
                return found
    return None


def _parse_xtbout_json(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "xtbout.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"source_file": "xtbout.json", "parse_error": True}
    charges = _recursive_find_key(payload, {"partialcharges", "charges", "atomiccharges"})
    dipole = _recursive_find_key(payload, {"dipole", "dipolemoment", "moleculardipole"})
    energy = _recursive_find_key(payload, {"totalenergy", "energy"})
    result: dict[str, Any] = {"source_file": "xtbout.json", "sha256": sha256_file(path)}
    if isinstance(charges, list) and all(isinstance(value, (int, float)) for value in charges):
        result["atomic_charges"] = [float(value) for value in charges]
    if isinstance(dipole, list) and all(isinstance(value, (int, float)) for value in dipole):
        result["dipole"] = [float(value) for value in dipole]
    if isinstance(energy, (int, float)):
        result["total_energy_hartree"] = float(energy)
    return result


def _parse_detailed_scan(job_dir: Path, request: dict[str, Any]) -> tuple[list[int] | None, list[float]]:
    candidates: list[Path] = []
    copied = request.get("detailed_input_job_copy")
    if isinstance(copied, str):
        candidates.append(job_dir / copied)
    candidates.extend([job_dir / "detailed.inp", job_dir / "xtb.inp"])
    text = ""
    for path in candidates:
        if path.exists():
            text = _read_text(path)
            break
    if not text:
        return None, []
    coordinate = None
    constraints: list[list[int] | None] = []
    constrain_block = re.search(r"\$constrain\b(.*?)(?=\$[A-Za-z]|\Z)", text, re.I | re.S)
    if constrain_block:
        for line in constrain_block.group(1).splitlines():
            if not re.search(r"^\s*(?:distance|angle|dihedral)\s*:", line, re.I):
                continue
            match = re.search(r"dihedral\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line, re.I)
            constraints.append([int(match.group(index)) for index in range(1, 5)] if match else None)
    targets: list[float] = []
    scan_block = re.search(r"\$scan\b(.*?)(?:\$end|\Z)", text, re.I | re.S)
    if scan_block:
        for line in scan_block.group(1).splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or ":" not in stripped or stripped.lower().startswith("mode"):
                continue
            lhs, rhs = stripped.split(":", 1)
            if lhs.strip().lower() == "dihedral":
                pieces = rhs.split(";", 1)
                atoms = re.findall(r"\d+", pieces[0])[:4]
                if len(atoms) == 4:
                    coordinate = [int(value) for value in atoms]
                rhs = pieces[1] if len(pieces) == 2 else ""
            elif lhs.strip().isdigit():
                constraint_index = int(lhs.strip()) - 1
                if 0 <= constraint_index < len(constraints) and constraints[constraint_index] is not None:
                    coordinate = constraints[constraint_index]
            else:
                continue
            values = [_as_float(value) for value in re.findall(_FLOAT, rhs)]
            if len(values) >= 3:
                start, stop, count_value = values[:3]
                count = int(round(count_value))
                if count >= 2 and abs(count - count_value) < 1e-6:
                    targets.extend(np.linspace(start, stop, count).tolist())
    return coordinate, targets


def _energy_from_comment(comment: str) -> float | None:
    for pattern in (
        re.compile(rf"\bSCF\s+done\s+({_FLOAT})", re.I),
        re.compile(rf"(?:energy|E)\s*[:=]\s*({_FLOAT})", re.I),
        re.compile(rf"^\s*({_FLOAT})\s*(?:Eh|hartree)?\b", re.I),
    ):
        match = pattern.search(comment)
        if match:
            value = _as_float(match.group(1))
            if -100000.0 < value < 10000.0:
                return value
    return None


def _target_from_comment(comment: str) -> float | None:
    match = re.search(rf"(?:target|angle|dihedral)\s*[:=]\s*({_FLOAT})", comment, re.I)
    return _as_float(match.group(1)) if match else None


def _parse_multixyz_scan(path: Path, coordinate: list[int] | None, targets: list[float]) -> list[dict[str, Any]]:
    try:
        frames = read_xyz_frames(path)
    except ValueError:
        return []
    points: list[dict[str, Any]] = []
    geometry_hash = sha256_file(path)
    for index, frame in enumerate(frames):
        energy = _energy_from_comment(frame.comment)
        target = _target_from_comment(frame.comment)
        if target is None and index < len(targets):
            target = float(targets[index])
        actual = None
        if coordinate is not None:
            try:
                actual = geometry_dihedral(frame, coordinate)
            except (ValueError, IndexError):
                actual = None
        points.append(
            {
                "point_index": index,
                "target_angle_deg": target,
                "actual_angle_deg": actual,
                "energy_hartree": energy,
                "converged": energy is not None,
                "geometry_frame_index": index,
                "geometry_sha256": geometry_hash,
            }
        )
    return points


def _parse_scan(job_dir: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    coordinate, targets = _parse_detailed_scan(job_dir, request)
    points: list[dict[str, Any]] = []
    sources: list[str] = []
    for name in ("xtbscan.xyz", "xtbscan.log"):
        path = job_dir / name
        if not path.exists():
            continue
        found = _parse_multixyz_scan(path, coordinate, targets)
        if found:
            if not points or sum(point["actual_angle_deg"] is not None for point in found) > sum(
                point["actual_angle_deg"] is not None for point in points
            ):
                points = found
            sources.append(name)
    if not points:
        return None
    return {
        "coordinate_atom_ids": coordinate,
        "target_schedule_deg": targets,
        "points": points,
        "source_files": sources,
    }


def _failure_evidence(text: str, *, limit: int = 12) -> tuple[str | None, list[str]]:
    categories = [
        ("timeout", re.compile(r"timed?\s*out|timeout", re.I)),
        ("scc_failure", re.compile(r"SCC.*(?:not|failed).*converg|SCF.*(?:not|failed).*converg", re.I)),
        ("optimization_failure", re.compile(r"geometry.*(?:not|failed).*converg|optimization.*failed", re.I)),
        ("input_error", re.compile(r"input.*error|unknown option|cannot read|failed to open", re.I)),
        ("segmentation_fault", re.compile(r"segmentation fault|SIGSEGV", re.I)),
        ("memory_limit", re.compile(r"out of memory|cannot allocate memory|killed", re.I)),
        ("abnormal_termination", re.compile(r"abnormal termination|fatal error|ERROR", re.I)),
    ]
    evidence: list[str] = []
    category = None
    lines = text.splitlines()
    for name, pattern in categories:
        matches = [line.strip() for line in lines if pattern.search(line)]
        if matches and category is None:
            category = name
        for line in matches:
            if line and line not in evidence:
                evidence.append(line[:500])
                if len(evidence) >= limit:
                    return category, evidence
    return category, evidence


def parse_xtb_output(output_path: str | Path) -> dict[str, Any]:
    job_dir = native_path(output_path)
    if job_dir.is_file():
        job_dir = job_dir.parent
    if not job_dir.is_dir():
        raise FileNotFoundError(job_dir)
    request = _request(job_dir)
    stdout = _read_text(job_dir / "stdout.txt")
    stderr = _read_text(job_dir / "stderr.txt")
    combined = stdout + "\n" + stderr
    detailed_text, detailed_source = _detailed_input_text(job_dir, request)
    protocol = _parse_protocol(combined, request)
    thermochemistry = _parse_thermochemistry(combined, detailed_text)
    if protocol["temperature_k"] is None:
        protocol["temperature_k"] = thermochemistry["temperature_k"]
    if protocol["thermo_rotor_cutoff_cm1"] is None:
        protocol["thermo_rotor_cutoff_cm1"] = thermochemistry["rotor_cutoff_cm1"]
    energies = _parse_energies(combined)
    xtbout = _parse_xtbout_json(job_dir)
    if "total_energy_hartree" not in energies and isinstance(xtbout.get("total_energy_hartree"), (int, float)):
        energies["total_energy_hartree"] = float(xtbout["total_energy_hartree"])
    vibrations = _parse_vibrations(job_dir, combined, request)
    optimized = _parse_optimized_geometry(job_dir)
    scan = _parse_scan(job_dir, request)
    failure_category, failure_lines = _failure_evidence(combined)
    normal = bool(re.search(r"normal termination of xtb", combined, re.I))
    scc_failure = bool(re.search(r"SCC.*(?:not|failed).*converg|SCF.*(?:not|failed).*converg", combined, re.I))
    scc_converged = not scc_failure and (normal or "total_energy_hartree" in energies)
    optimization_requested = any(
        value == "--opt" or value.startswith("--opt=") or value == "--ohess" for value in _arguments(request)
    )
    optimization_converged = bool(
        re.search(r"GEOMETRY\s+OPTIMIZATION\s+CONVERGED|geometry optimization converged", combined, re.I)
    ) or bool(optimization_requested and optimized is not None and normal)
    result: dict[str, Any] = {
        "schema_version": "2.0",
        "output_directory": str(job_dir),
        "protocol": protocol,
        "status": {
            "normal_termination": normal,
            "scc_converged": scc_converged,
            "optimization_requested": optimization_requested,
            "optimization_converged": optimization_converged,
        },
        "energies": energies,
        "thermochemistry": thermochemistry,
        "vibrations": vibrations,
        "optimized_geometry": optimized,
        "properties": xtbout,
        "scan": scan,
        "failure": {"category": failure_category, "evidence_lines": failure_lines},
        "source_files": {
            path.name: sha256_file(path)
            for path in sorted(job_dir.iterdir())
            if path.is_file() and path.name in {"stdout.txt", "stderr.txt", "xtbopt.xyz", "vibspectrum", "vibspectrum.out", "hessian", "xtbout.json", "xtbscan.xyz", "xtbscan.log", detailed_source}
        },
    }
    return result
