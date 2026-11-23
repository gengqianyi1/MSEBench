"""Load and classify organizer-owned Task 18 xTB evidence."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from geometry import kabsch_rmsd_text, ordered_element_signature


ENERGY_PATTERN = re.compile(
    r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _flag(command: list[object], name: str) -> int | None:
    try:
        index = command.index(name)
        value = command[index + 1]
        if isinstance(value, bool):
            return None
        return int(value)
    except (ValueError, IndexError, TypeError):
        return None


def _total_energy(path: Path) -> float | None:
    if not path.is_file():
        return None
    matches = ENERGY_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))
    return float(matches[-1]) if matches else None


def parse_vibrational_spectrum(text: str, *, atom_count: int) -> dict[str, object]:
    frequencies: list[float] = []
    for raw in text.replace("\r", "").splitlines():
        fields = raw.split()
        if not fields or not fields[0].isdigit():
            continue
        wave_number: float | None = None
        for token in fields[1:3]:
            try:
                wave_number = float(token)
                break
            except ValueError:
                continue
        if wave_number is not None:
            frequencies.append(wave_number)
    imaginary = [value for value in frequencies if value < -20.0]
    expected = 3 * atom_count
    complete = len(frequencies) == expected
    return {
        "frequencies_cm1": frequencies,
        "imaginary_frequencies_cm1": imaginary,
        "expected_mode_count": expected,
        "parsed_mode_count": len(frequencies),
        "complete": complete,
        "minimum": complete and not imaginary,
    }


def load_trusted_jobs(run_root: Path) -> dict[str, dict[str, object]]:
    run_root = Path(run_root)
    records_path = run_root / "jobs.json"
    if not records_path.is_file():
        return {}
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(records, list):
        return {}
    trusted: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"job_\d{3}", job_id):
            continue
        directory = run_root / "jobs" / job_id
        command = record.get("command")
        input_value = record.get("input")
        if (
            record.get("status") != "completed"
            or record.get("return_code") != 0
            or not directory.is_dir()
            or not isinstance(command, list)
            or not isinstance(input_value, str)
        ):
            continue
        snapshot_value = record.get("input_snapshot")
        input_path = directory / (
            snapshot_value
            if isinstance(snapshot_value, str) and Path(snapshot_value).name == snapshot_value
            else Path(input_value).name
        )
        if not input_path.is_file():
            continue
        energy = _total_energy(directory / "stdout.txt")
        if energy is None:
            continue
        operation = record.get("operation")
        if operation not in {"sp", "opt", "hess", "opt_hess"}:
            continue
        input_bytes = input_path.read_bytes()
        input_sha256 = sha256_bytes(input_bytes)
        input_text = input_bytes.decode("utf-8", errors="strict")
        lineage_match = re.fullmatch(r"job_results/(job_\d{3})/xtbopt\.xyz", input_value)
        if not isinstance(snapshot_value, str) and lineage_match is not None:
            parent = trusted.get(lineage_match.group(1))
            parent_output = parent.get("output_text") if parent is not None else None
            parent_output_sha256 = parent.get("output_sha256") if parent is not None else None
            if not isinstance(parent_output, str) or not isinstance(parent_output_sha256, str):
                continue
            # Legacy jobs could run an input named xtbopt.xyz, which xTB then
            # overwrote.  The organizer-owned input path still binds the job
            # to the earlier authoritative output, so reconstruct that input
            # without trusting the model-visible result mirror.
            input_sha256 = parent_output_sha256
            input_text = parent_output
        output_path = directory / "xtbopt.xyz"
        output_text = output_path.read_text(encoding="utf-8", errors="strict") if output_path.is_file() else None
        vibration: dict[str, object] | None = None
        if (
            operation in {"hess", "opt_hess"}
            and (directory / "hessian").is_file()
            and (directory / "vibspectrum").is_file()
            and (operation != "opt_hess" or (directory / ".xtboptok").exists())
        ):
            try:
                atom_count = int(input_text.splitlines()[0].strip())
                vibration = parse_vibrational_spectrum(
                    (directory / "vibspectrum").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                    atom_count=atom_count,
                )
            except (IndexError, TypeError, ValueError):
                vibration = None
        hessian_complete = bool(vibration and vibration["complete"])
        trusted[job_id] = {
            "job_id": job_id,
            "operation": operation,
            "charge": _flag(command, "--chrg"),
            "uhf": _flag(command, "--uhf"),
            "energy_hartree": energy,
            "input_sha256": input_sha256,
            "input_text": input_text,
            "output_sha256": sha256_bytes(output_path.read_bytes()) if output_path.is_file() else None,
            "output_text": output_text,
            "hessian_complete": hessian_complete,
            "hessian_minimum": bool(vibration and vibration["minimum"]),
            "vibrations": vibration,
            "optimization_converged": bool(
                operation in {"opt", "opt_hess"}
                and output_path.is_file()
                and (directory / ".xtboptok").exists()
            ),
        }
    return trusted


def assign_candidates(
    jobs: dict[str, dict[str, object]],
    public_input: Path,
) -> dict[str, str]:
    seeds = {
        f"donor_{index:02d}": (Path(public_input) / f"donor_{index:02d}.xyz").read_text(encoding="utf-8")
        for index in range(1, 4)
    }
    signature_to_candidate = {
        ordered_element_signature(text): candidate_id
        for candidate_id, text in seeds.items()
    }
    if len(signature_to_candidate) != len(seeds):
        raise ValueError("Task 18 public candidates require unique ordered element signatures")
    sha_to_candidate = {
        sha256_bytes(text.encode("utf-8")): candidate_id
        for candidate_id, text in seeds.items()
    }
    assignment: dict[str, str] = {}
    for job_id in sorted(jobs):
        job = jobs[job_id]
        input_sha = str(job["input_sha256"])
        candidate_id = sha_to_candidate.get(input_sha)
        if candidate_id is None:
            try:
                candidate_id = signature_to_candidate.get(
                    ordered_element_signature(str(job["input_text"]))
                )
            except (TypeError, ValueError):
                candidate_id = None
        if candidate_id is None:
            continue
        assignment[job_id] = candidate_id
        output_sha = job.get("output_sha256")
        if isinstance(output_sha, str):
            sha_to_candidate[output_sha] = candidate_id
    return assignment


def _matching_hessian(
    members: list[dict[str, object]],
    minimum: dict[str, object],
    *,
    charge: int,
    uhf: int,
) -> dict[str, object] | None:
    if minimum["operation"] == "opt_hess" and minimum["hessian_complete"]:
        return minimum
    matches = [
        job
        for job in members
        if job["operation"] == "hess"
        and job["charge"] == charge
        and job["uhf"] == uhf
        and job["hessian_complete"]
        and job["input_sha256"] == minimum["output_sha256"]
    ]
    return min(matches, key=lambda job: str(job["job_id"])) if matches else None


def enumerate_coherent_cycles(
    jobs: dict[str, dict[str, object]],
    assignment: dict[str, str],
) -> dict[str, list[dict[str, object]]]:
    cycles: dict[str, list[dict[str, object]]] = {}
    for candidate_id in ("donor_01", "donor_02", "donor_03"):
        members = [job for job_id, job in jobs.items() if assignment.get(job_id) == candidate_id]
        neutral_minima = [
            job
            for job in members
            if job["operation"] in {"opt", "opt_hess"}
            and job["charge"] == 0
            and job["uhf"] == 0
            and job["optimization_converged"]
            and isinstance(job.get("output_sha256"), str)
        ]
        cation_minima = [
            job
            for job in members
            if job["operation"] in {"opt", "opt_hess"}
            and job["charge"] == 1
            and job["uhf"] == 1
            and job["optimization_converged"]
            and isinstance(job.get("output_sha256"), str)
        ]
        candidate_cycles: list[dict[str, object]] = []
        for neutral in neutral_minima:
            cation_at_neutral = [
                job
                for job in members
                if job["operation"] == "sp"
                and job["charge"] == 1
                and job["uhf"] == 1
                and job["input_sha256"] == neutral["output_sha256"]
            ]
            neutral_hessian = _matching_hessian(
                members, neutral, charge=0, uhf=0
            )
            for cation in cation_minima:
                neutral_at_cation = [
                    job
                    for job in members
                    if job["operation"] == "sp"
                    and job["charge"] == 0
                    and job["uhf"] == 0
                    and job["input_sha256"] == cation["output_sha256"]
                ]
                cation_hessian = _matching_hessian(
                    members, cation, charge=1, uhf=1
                )
                for cross_cation in cation_at_neutral:
                    for cross_neutral in neutral_at_cation:
                        try:
                            rmsd = kabsch_rmsd_text(
                                str(neutral["output_text"]),
                                str(cation["output_text"]),
                            )
                        except (ValueError, TypeError):
                            continue
                        role_jobs = [neutral, cation, cross_cation, cross_neutral]
                        for hessian in (neutral_hessian, cation_hessian):
                            if hessian is not None and hessian not in role_jobs:
                                role_jobs.append(hessian)
                        candidate_cycles.append(
                            {
                                "neutral_minimum": neutral,
                                "cation_minimum": cation,
                                "cation_at_neutral": cross_cation,
                                "neutral_at_cation": cross_neutral,
                                "neutral_hessian": neutral_hessian,
                                "cation_hessian": cation_hessian,
                                "neutral_cation_rmsd_angstrom": rmsd,
                                "hessian_validated": bool(
                                    neutral_hessian is not None
                                    and cation_hessian is not None
                                ),
                                "job_ids": [str(job["job_id"]) for job in role_jobs],
                            }
                        )
        if candidate_cycles:
            cycles[candidate_id] = sorted(
                candidate_cycles,
                key=lambda cycle: (
                    float(cycle["neutral_minimum"]["energy_hartree"]),
                    float(cycle["cation_minimum"]["energy_hartree"]),
                    tuple(cycle["job_ids"]),
                ),
            )
    return cycles


def coherent_cycles(
    jobs: dict[str, dict[str, object]],
    assignment: dict[str, str],
) -> dict[str, dict[str, object]]:
    return {
        candidate_id: candidate_cycles[0]
        for candidate_id, candidate_cycles in enumerate_coherent_cycles(
            jobs, assignment
        ).items()
    }
