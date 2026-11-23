"""Read the trusted single-run xTB job record for private Task 04 scoring."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from tasks.task_04.private.evaluator.geometry import XYZGeometry, read_xyz


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


@dataclass(frozen=True)
class TrustedJob:
    job_id: str
    operation: str
    condition: str
    record: dict[str, object]
    workdir: Path
    input_geometry: XYZGeometry
    optimized_geometry: XYZGeometry | None
    total_energy_hartree: float | None
    gibbs_energy_hartree: float | None
    protocol_valid: bool
    termination_valid: bool
    optimization_converged: bool
    thermochemistry_valid: bool
    frequencies_cm1: tuple[float, ...]
    errors: tuple[str, ...]


def _last_float(text: str, label: str) -> float | None:
    matches = re.findall(rf"{label}\s+({FLOAT})", text, re.I)
    if not matches:
        return None
    try:
        return float(matches[-1].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _control_values(job_dir: Path, command: list[str]) -> tuple[float | None, float | None]:
    if "--input" not in command:
        return None, None
    index = command.index("--input")
    if index + 1 >= len(command):
        return None, None
    control = job_dir / Path(command[index + 1]).name
    if not control.is_file():
        return None, None
    text = control.read_text(encoding="utf-8", errors="replace")
    block = re.search(r"\$thermo\b(.*?)(?:\$end|\Z)", text, re.I | re.S)
    if block is None:
        return None, None
    temperature = re.search(rf"\btemp\s*=\s*({FLOAT})", block.group(1), re.I)
    rotor = re.search(rf"\bsthr\s*=\s*({FLOAT})", block.group(1), re.I)
    return (
        float(temperature.group(1)) if temperature else None,
        float(rotor.group(1)) if rotor else None,
    )


def _reported_thermo_values(stdout: str) -> tuple[float | None, float | None]:
    temperature_match = re.search(rf"\btemperature\s+({FLOAT})\s*K\b", stdout, re.I)
    if temperature_match is None:
        temperature_match = re.search(rf"^\s*({FLOAT})\s+VIB\b", stdout, re.I | re.M)
    rotor_match = re.search(rf"\brotor\s+cutoff\s+({FLOAT})\s*cm", stdout, re.I)
    return (
        float(temperature_match.group(1)) if temperature_match else None,
        float(rotor_match.group(1)) if rotor_match else None,
    )


def _frequencies(path: Path, atom_count: int) -> tuple[float, ...]:
    if not path.is_file():
        return ()
    raw = path.read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    in_table = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("$vibrational spectrum"):
            in_table = True
            continue
        if in_table and stripped.lower().startswith("$end"):
            break
        fields = stripped.split()
        if not fields or not fields[0].isdigit():
            continue
        frequency_field = 2 if in_table and len(fields) >= 3 else 1
        if len(fields) <= frequency_field:
            continue
        try:
            values.append(float(fields[frequency_field].replace("D", "E")))
        except ValueError:
            continue
    expected = max(0, 3 * atom_count - 6)
    if len(values) == 3 * atom_count:
        values = values[-expected:]
    return tuple(values)


def _condition(command: list[str]) -> str:
    if "--alpb" not in command:
        return "gas"
    index = command.index("--alpb")
    return "water" if index + 1 < len(command) and command[index + 1].lower() == "water" else "other"


def _flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _parse_job(run_root: Path, record: dict[str, object]) -> TrustedJob:
    errors: list[str] = []
    job_id = str(record.get("job_id") or "")
    operation = str(record.get("operation") or "")
    workdir = run_root / "jobs" / job_id
    command = [str(value) for value in record.get("command", [])]
    if record.get("output_directory") != f"jobs/{job_id}":
        errors.append("output_directory mismatch")
    if record.get("status") != "completed" or record.get("return_code") != 0:
        errors.append("job did not complete successfully")
    if not workdir.is_dir():
        errors.append("job directory missing")

    input_name = Path(str(record.get("input") or "input.xyz")).name
    input_path = workdir / input_name
    if not input_path.is_file():
        fallback = workdir / "input.xyz"
        input_path = fallback if fallback.is_file() else input_path
    try:
        input_geometry = read_xyz(input_path)
    except (OSError, ValueError):
        errors.append("input geometry missing or invalid")
        input_geometry = XYZGeometry(("H",), [[0.0, 0.0, 0.0]])

    stdout_path = workdir / "stdout.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    version_ok = bool(re.search(r"xtb\s+version\s+6\.7\.1", stdout, re.I))
    termination = bool(
        re.search(r"normal termination of xtb|^\s*\*\s*finished run on\b", stdout, re.I | re.M)
    )
    expected_flag = "--opt" if operation == "scan" else f"--{operation}"
    protocol = (
        operation in {"sp", "opt", "hess", "scan"}
        and expected_flag in command
        and _flag_value(command, "--chrg") == "0"
        and _flag_value(command, "--uhf") == "0"
        and _condition(command) in {"gas", "water"}
        and version_ok
    )
    if not protocol:
        errors.append("protocol mismatch")
    if not termination:
        errors.append("normal termination missing")

    optimized: XYZGeometry | None = None
    optimization_converged = False
    if operation == "opt":
        optimization_converged = bool(re.search(r"GEOMETRY OPTIMIZATION CONVERGED", stdout, re.I))
        try:
            optimized = read_xyz(workdir / "xtbopt.xyz")
        except (OSError, ValueError):
            errors.append("optimized geometry missing or invalid")
        if not optimization_converged:
            errors.append("optimization convergence missing")

    total_energy = _last_float(stdout, r"TOTAL\s+ENERGY")
    gibbs_energy = _last_float(stdout, r"TOTAL\s+FREE\s+ENERGY")
    frequencies = _frequencies(workdir / "vibspectrum", len(input_geometry.elements))
    thermochemistry_valid = False
    if operation == "hess":
        requested_temperature, requested_rotor = _control_values(workdir, command)
        reported_temperature, reported_rotor = _reported_thermo_values(stdout)
        requested_values_valid = bool(
            (requested_temperature is None or abs(requested_temperature - 298.15) <= 1e-6)
            and (requested_rotor is None or abs(requested_rotor - 50.0) <= 1e-6)
        )
        expected_modes = max(0, 3 * len(input_geometry.elements) - 6)
        thermochemistry_valid = bool(
            gibbs_energy is not None
            and requested_values_valid
            and reported_temperature is not None
            and abs(reported_temperature - 298.15) <= 1e-6
            and reported_rotor is not None
            and abs(reported_rotor - 50.0) <= 1e-6
            and len(frequencies) == expected_modes
            and not any(value < -30.0 for value in frequencies)
            and re.search(r"mRRHO|rotor cutoff", stdout, re.I)
        )
        if not thermochemistry_valid:
            errors.append("Hessian or thermochemistry evidence incomplete")

    return TrustedJob(
        job_id=job_id,
        operation=operation,
        condition=_condition(command),
        record=record,
        workdir=workdir,
        input_geometry=input_geometry,
        optimized_geometry=optimized,
        total_energy_hartree=total_energy if total_energy is not None and math.isfinite(total_energy) else None,
        gibbs_energy_hartree=gibbs_energy if gibbs_energy is not None and math.isfinite(gibbs_energy) else None,
        protocol_valid=protocol,
        termination_valid=termination,
        optimization_converged=optimization_converged,
        thermochemistry_valid=thermochemistry_valid,
        frequencies_cm1=frequencies,
        errors=tuple(errors),
    )


def load_trusted_jobs(run_root: Path) -> tuple[dict[str, TrustedJob], list[str]]:
    path = Path(run_root) / "jobs.json"
    if not path.is_file():
        return {}, ["jobs.json missing"]
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, ["jobs.json invalid"]
    if not isinstance(records, list):
        return {}, ["jobs.json must contain a list"]
    jobs: dict[str, TrustedJob] = {}
    errors: list[str] = []
    for raw in records:
        if not isinstance(raw, dict):
            errors.append("job record is not an object")
            continue
        job = _parse_job(Path(run_root), raw)
        if not job.job_id or job.job_id in jobs:
            errors.append("job IDs must be unique and non-empty")
            continue
        jobs[job.job_id] = job
        errors.extend(f"{job.job_id}: {message}" for message in job.errors)
    return jobs, errors
