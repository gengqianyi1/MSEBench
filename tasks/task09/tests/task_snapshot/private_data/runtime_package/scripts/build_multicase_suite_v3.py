"""Generate organizer-only torsion cases from real native xTB 6.7.1 runs.

Each grid point starts from the same RDKit reference for its molecule, applies
a constrained UFF preoptimization, and then executes a constrained GFN2-xTB
optimization. Raw attempt artifacts remain organizer-only. Model-visible case
materials contain no oracle values.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "organizer_private"
CASES_ROOT = PRIVATE / "cases"
HARTREE_TO_KCAL_MOL = 627.5094740631
GRID = tuple(range(0, 360, 5))
XTB_VERSION = "6.7.1"
FINAL_ANGLE_TOLERANCE_DEG = 2.0

CASE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "butane_central_torsion_v1": {
        "molecule": "n-butane",
        "mapped_smiles": "[CH3:1][CH2:2][CH2:3][CH3:4]",
        "target_maps": [1, 2, 3, 4],
        "reference_angle_deg": 180.0,
        "seed": 202608041,
        "expected_morphology": "threefold alkane anti/gauche profile",
    },
    "difluoroethane_fccf_torsion_v1": {
        "molecule": "1,2-difluoroethane",
        "mapped_smiles": "[F:1][CH2:2][CH2:3][F:4]",
        "target_maps": [1, 2, 3, 4],
        "reference_angle_deg": 60.0,
        "seed": 202608042,
        "expected_morphology": "gauche-effect profile",
    },
    "n_methylacetamide_cn_torsion_v1": {
        "molecule": "N-methylacetamide",
        "mapped_smiles": "[CH3:1][C:2](=[O:5])[NH:3][CH3:4]",
        "target_maps": [1, 2, 3, 4],
        "reference_angle_deg": 180.0,
        "seed": 202608043,
        "expected_morphology": "amide partial-double-bond planar barrier profile",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def locate_xtb() -> Path:
    candidates = [
        Path(sys.prefix) / "Library" / "bin" / "xtb.exe",
        Path(sys.prefix) / "Scripts" / "xtb.exe",
    ]
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.extend(
            [Path(prefix) / "Library" / "bin" / "xtb.exe", Path(prefix) / "Scripts" / "xtb.exe"]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise RuntimeError("xTB executable not found in the active environment")


def native_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    )
    return env


def xtb_identity(executable: Path) -> dict[str, Any]:
    version_run = subprocess.run(
        [str(executable), "--version"],
        cwd=str(executable.parent),
        env=native_env(),
        capture_output=True,
        timeout=30,
        shell=False,
    )
    help_run = subprocess.run(
        [str(executable), "--help"],
        cwd=str(executable.parent),
        env=native_env(),
        capture_output=True,
        timeout=30,
        shell=False,
    )
    version_bytes = version_run.stdout + version_run.stderr
    help_bytes = help_run.stdout + help_run.stderr
    version_text = version_bytes.decode("utf-8", errors="replace")
    help_text = help_bytes.decode("utf-8", errors="replace")
    match = re.search(r"xtb version\s+([0-9]+\.[0-9]+\.[0-9]+)", version_text, re.I)
    tokens = ["--opt", "--input", "--gfn", "--chrg", "--uhf"]
    if version_run.returncode != 0 or not match or match.group(1) != XTB_VERSION:
        raise RuntimeError("exact xTB 6.7.1 identity check failed")
    if any(token not in help_text for token in tokens):
        raise RuntimeError("required xTB capability token missing")
    return {
        "version": match.group(1),
        "version_exit_code": version_run.returncode,
        "required_help_tokens": tokens,
        "binary_sha256": sha256(executable),
        "version_output_sha256": hashlib.sha256(version_bytes).hexdigest().upper(),
        "help_output_sha256": hashlib.sha256(help_bytes).hexdigest().upper(),
        "identity_policy": "xtb_portable_identity/2.0",
    }


def base_molecule(definition: dict[str, Any]) -> Chem.Mol:
    source = Chem.MolFromSmiles(str(definition["mapped_smiles"]))
    if source is None:
        raise RuntimeError("invalid mapped SMILES")
    molecule = Chem.AddHs(source)
    maps = {atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
    if not set(definition["target_maps"]).issubset(maps):
        raise RuntimeError("target atom maps are absent")
    return molecule


def target_indices(molecule: Chem.Mol, definition: dict[str, Any]) -> tuple[int, int, int, int]:
    mapping = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum()
    }
    return tuple(mapping[int(value)] for value in definition["target_maps"])  # type: ignore[return-value]


def circular_distance(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def signed_angle(angle: float) -> float:
    value = float(angle) % 360.0
    return value - 360.0 if value > 180.0 else value


def constrain_uff(molecule: Chem.Mol, definition: dict[str, Any], angle: float) -> None:
    indices = target_indices(molecule, definition)
    target = signed_angle(angle)
    rdMolTransforms.SetDihedralDeg(molecule.GetConformer(), *indices, target)
    field = AllChem.UFFGetMoleculeForceField(molecule)
    if field is None:
        raise RuntimeError("UFF force field unavailable")
    field.UFFAddTorsionConstraint(*indices, False, target - 0.01, target + 0.01, 500.0)
    status = int(field.Minimize(maxIts=3000, forceTol=1.0e-6, energyTol=1.0e-8))
    if status != 0:
        raise RuntimeError("constrained UFF did not converge")
    actual = rdMolTransforms.GetDihedralDeg(molecule.GetConformer(), *indices)
    if circular_distance(actual, target) > 0.05:
        raise RuntimeError("constrained UFF angle tolerance failed")


def generate_reference(definition: dict[str, Any]) -> Chem.Mol:
    molecule = base_molecule(definition)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(definition["seed"])
    params.useRandomCoords = False
    params.enforceChirality = True
    if int(AllChem.EmbedMolecule(molecule, params)) != 0:
        raise RuntimeError("ETKDGv3 embedding failed")
    constrain_uff(molecule, definition, float(definition["reference_angle_deg"]))
    return molecule


def xyz_bytes(molecule: Chem.Mol, comment: str) -> bytes:
    conformer = molecule.GetConformer()
    lines = [str(molecule.GetNumAtoms()), comment]
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2s} {point.x: .12f} {point.y: .12f} {point.z: .12f}")
    return ("\n".join(lines) + "\n").encode("ascii")


def molecule_from_xyz(path: Path, template: Chem.Mol) -> Chem.Mol:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    if count != template.GetNumAtoms() or len(lines) < count + 2:
        raise RuntimeError("XYZ atom count mismatch")
    molecule = Chem.Mol(template)
    conformer = Chem.Conformer(count)
    for index, row in enumerate(lines[2 : 2 + count]):
        fields = row.split()
        if len(fields) < 4 or fields[0] != molecule.GetAtomWithIdx(index).GetSymbol():
            raise RuntimeError("XYZ atom order mismatch")
        coordinates = [float(value) for value in fields[1:4]]
        if not all(math.isfinite(value) for value in coordinates):
            raise RuntimeError("XYZ coordinate is non-finite")
        conformer.SetAtomPosition(index, Point3D(*coordinates))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def structure_signature(molecule: Chem.Mol, definition: dict[str, Any], reference: Path) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "molecule": definition["molecule"],
        "mapped_smiles": definition["mapped_smiles"],
        "atom_count": molecule.GetNumAtoms(),
        "atom_symbols": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "composition": dict(sorted(Counter(atom.GetSymbol() for atom in molecule.GetAtoms()).items())),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())),
        "bonds_zero_based": sorted(
            [
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            ]
            for bond in molecule.GetBonds()
        ),
        "target_dihedral_atom_maps": definition["target_maps"],
        "target_dihedral_indices_zero_based": list(target_indices(molecule, definition)),
        "target_dihedral_indices_one_based": [value + 1 for value in target_indices(molecule, definition)],
        "reference_seed": definition["seed"],
        "reference_angle_deg": definition["reference_angle_deg"],
        "reference_xyz_sha256": sha256(reference),
    }


def artifact_manifest(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "attempt_manifest.json"
    ]
    return {"schema_version": "1.0", "algorithm": "SHA-256", "files": files}


def run_angle(
    *,
    case_root: Path,
    definition: dict[str, Any],
    reference_molecule: Chem.Mol,
    reference_path: Path,
    angle: int,
    executable: Path,
    identity: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    attempt = case_root / "native_attempts" / f"angle_{angle:03d}"
    result_path = attempt / "result.json"
    if result_path.is_file() and (attempt / "attempt_manifest.json").is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if attempt.exists():
        raise FileExistsError(f"incomplete attempt already exists: {attempt}")
    attempt.mkdir(parents=True)
    molecule = Chem.Mol(reference_molecule)
    constrain_uff(molecule, definition, angle)
    input_path = attempt / "input.xyz"
    input_path.write_bytes(xyz_bytes(molecule, f"constrained UFF target={angle} deg"))
    one_based = [value + 1 for value in target_indices(molecule, definition)]
    control = (
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: {', '.join(str(value) for value in one_based)}, {signed_angle(angle):.1f}\n"
        "$end\n"
    )
    (attempt / "xcontrol.inp").write_text(control, encoding="ascii", newline="\n")
    argv = [
        str(executable), "input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0",
        "--opt", "tight", "--input", "xcontrol.inp",
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=str(attempt),
            env=native_env(),
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
        return_code = int(completed.returncode)
        stdout_bytes, stderr_bytes = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = -1
        stdout_bytes, stderr_bytes = exc.stdout or b"", exc.stderr or b""
    raw_stdout_sha256 = hashlib.sha256(stdout_bytes).hexdigest().upper()
    release_stdout = stdout_bytes.replace(str(executable).encode("utf-8"), b"xtb")
    (attempt / "stdout.log").write_bytes(release_stdout)
    (attempt / "stderr.log").write_bytes(stderr_bytes)
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    final_path = attempt / "xtbopt.xyz"
    energy: float | None = None
    final_angle: float | None = None
    parse_error: str | None = None
    try:
        matches = re.findall(r"\|\s*TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh\s*\|", stdout)
        if not matches or not final_path.is_file():
            raise RuntimeError("native energy or xtbopt.xyz missing")
        energy = float(matches[-1])
        final_molecule = molecule_from_xyz(final_path, reference_molecule)
        final_angle = rdMolTransforms.GetDihedralDeg(
            final_molecule.GetConformer(), *target_indices(final_molecule, definition)
        ) % 360.0
    except (OSError, ValueError, RuntimeError) as exc:
        parse_error = str(exc)
    angle_error = circular_distance(final_angle, angle) if final_angle is not None else None
    eligible = bool(
        return_code == 0
        and not timed_out
        and "GEOMETRY OPTIMIZATION CONVERGED" in stdout
        and "FAILED TO CONVERGE" not in stdout
        and energy is not None
        and math.isfinite(energy)
        and angle_error is not None
        and angle_error <= FINAL_ANGLE_TOLERANCE_DEG
    )
    result = {
        "schema_version": "3.0",
        "case_id": case_root.name,
        "angle_deg": angle,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 6),
        "converged": "GEOMETRY OPTIMIZATION CONVERGED" in stdout and "FAILED TO CONVERGE" not in stdout,
        "total_energy_hartree": energy,
        "final_angle_deg": final_angle,
        "angle_error_deg": angle_error,
        "profile_eligible": eligible,
        "parse_error": parse_error,
        "reference_xyz_sha256": sha256(reference_path),
        "input_xyz_sha256": sha256(input_path),
        "xcontrol_sha256": sha256(attempt / "xcontrol.inp"),
        "stdout_raw_sha256": raw_stdout_sha256,
        "stdout_sha256": sha256(attempt / "stdout.log"),
        "stdout_projection": "native-program-call-path-redaction/1.0",
        "stderr_sha256": sha256(attempt / "stderr.log"),
        "xtbopt_xyz_sha256": sha256(final_path) if final_path.is_file() else None,
        "native_argv": ["xtb", *argv[1:]],
        "xtb_identity": identity,
    }
    write_json(result_path, result)
    write_json(attempt / "attempt_manifest.json", artifact_manifest(attempt))
    return result


def build_case(case_id: str, angles: tuple[int, ...], timeout: int) -> dict[str, Any]:
    definition = CASE_DEFINITIONS[case_id]
    case_root = CASES_ROOT / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    reference_path = case_root / "reference.xyz"
    structure_path = case_root / "reference_structure.json"
    if reference_path.is_file():
        reference_molecule = molecule_from_xyz(reference_path, base_molecule(definition))
    else:
        reference_molecule = generate_reference(definition)
        reference_path.write_bytes(
            xyz_bytes(reference_molecule, f"ETKDGv3 seed={definition['seed']}; constrained UFF")
        )
        write_json(structure_path, structure_signature(reference_molecule, definition, reference_path))
    if not structure_path.is_file():
        write_json(structure_path, structure_signature(reference_molecule, definition, reference_path))
    executable = locate_xtb()
    identity = xtb_identity(executable)
    results = []
    for angle in angles:
        result = run_angle(
            case_root=case_root,
            definition=definition,
            reference_molecule=reference_molecule,
            reference_path=reference_path,
            angle=angle,
            executable=executable,
            identity=identity,
            timeout=timeout,
        )
        results.append(result)
        print(
            f"{case_id} angle={angle:03d} eligible={result['profile_eligible']} "
            f"energy={result['total_energy_hartree']}"
        )
    complete = angles == GRID
    if complete:
        invalid = [item["angle_deg"] for item in results if item["profile_eligible"] is not True]
        if invalid:
            raise RuntimeError(f"case {case_id} has invalid grid angles: {invalid}")
        energies = [float(item["total_energy_hartree"]) for item in results]
        minimum = min(energies)
        relative = [(energy - minimum) * HARTREE_TO_KCAL_MOL for energy in energies]
        oracle = {
            "schema_version": "3.0",
            "task_id": "adaptive_xtb_torsion_profile_agent_v3",
            "case_id": case_id,
            "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
            "method": "GFN2-xTB",
            "grid_angles_deg": list(GRID),
            "total_energies_hartree": energies,
            "relative_energies_kcal_mol": relative,
            "selected_attempt_ids": [f"angle_{angle:03d}" for angle in GRID],
        }
        write_json(case_root / "oracle_profile.json", oracle)
        provenance = {
            "schema_version": "3.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "native_attempt_count": len(results),
            "all_angles_eligible": True,
            "maximum_final_angle_error_deg": max(float(item["angle_error_deg"]) for item in results),
            "reference_xyz_sha256": sha256(reference_path),
            "oracle_profile_sha256": sha256(case_root / "oracle_profile.json"),
            "xtb_identity": identity,
            "scientific_invariants": {
                "method": "GFN2-xTB",
                "phase": "gas",
                "charge": 0,
                "uhf": 0,
                "threads": 1,
                "grid_step_deg": 5,
                "final_angle_tolerance_deg": FINAL_ANGLE_TOLERANCE_DEG,
            },
            "attempt_result_sha256": {
                f"angle_{item['angle_deg']:03d}": sha256(
                    case_root / "native_attempts" / f"angle_{item['angle_deg']:03d}" / "result.json"
                )
                for item in results
            },
        }
        write_json(case_root / "oracle_provenance.json", provenance)
        public_spec = {
            "schema_version": "3.0",
            "task_id": "adaptive_xtb_torsion_profile_agent_v3",
            "case_id": case_id,
            "molecule": definition["molecule"],
            "mapped_smiles": definition["mapped_smiles"],
            "target_dihedral_atom_maps": definition["target_maps"],
            "scientific_invariants": provenance["scientific_invariants"],
            "budgets": {
                "max_unique_scan_angles": 12,
                "min_unique_valid_scan_angles": 10,
                "max_scan_batches": 4,
                "min_feedback_rounds": 2,
            },
        }
        write_json(case_root / "case_spec.json", public_spec)
        write_json(case_root / "case_manifest.json", artifact_manifest(case_root))
        return provenance
    return {"case_id": case_id, "smoke_angles": list(angles), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", *CASE_DEFINITIONS], default="all")
    parser.add_argument("--angles", help="comma-separated 5-degree grid angles; omit for full grid")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.angles:
        angles = tuple(int(value) for value in args.angles.split(","))
        if any(value not in GRID for value in angles) or len(set(angles)) != len(angles):
            raise ValueError("angles must be unique members of the 5-degree grid")
    else:
        angles = GRID
    selected = list(CASE_DEFINITIONS) if args.case == "all" else [args.case]
    summaries = [build_case(case_id, angles, args.timeout) for case_id in selected]
    write_json(PRIVATE / "native_generation_summary.json", {"schema_version": "3.0", "cases": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
