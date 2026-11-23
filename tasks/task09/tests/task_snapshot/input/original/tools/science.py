"""Frozen RDKit/xTB scientific protocol for the torsion benchmark.

All native calculations start from the same immutable ``reference.xyz``.
The module never accepts a model-supplied path or executable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D


MAPPED_SMILES = (
    "[F:1][c:2]1[cH:3][cH:4][cH:5][cH:6][c:7]1-"
    "[c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1"
)
TARGET_DIHEDRAL_MAPS = (6, 7, 8, 9)
REFERENCE_SEED = 20260720
REFERENCE_TORSION_DEG = 90.0
ANGLE_GRID_DEG = 5
FINAL_ANGLE_TOLERANCE_DEG = 2.0
HARTREE_TO_KCAL_MOL = 627.5094740631
XTB_VERSION = "6.7.1"
XTB_BINARY_SHA256 = "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"


class ScientificProtocolError(RuntimeError):
    """A frozen scientific or native-runtime invariant was violated."""


class AngleValidationError(ValueError):
    """An angle was outside the frozen public domain."""


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(canonical_json_bytes(value))


def validate_angle(angle: Any) -> int:
    """Accept only integer multiples of five in the half-open [0, 360) domain."""

    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise AngleValidationError("angle_deg must be a numeric integer")
    value = float(angle)
    if not math.isfinite(value) or not value.is_integer():
        raise AngleValidationError("angle_deg must be a finite integer")
    integer = int(value)
    if integer < 0 or integer >= 360:
        raise AngleValidationError("angle_deg must be in [0, 360)")
    if integer % ANGLE_GRID_DEG:
        raise AngleValidationError("angle_deg must be a multiple of 5")
    return integer


def signed_angle(angle: float) -> float:
    value = float(angle) % 360.0
    return value - 360.0 if value > 180.0 else value


def circular_distance_deg(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def _base_molecule() -> Chem.Mol:
    mol = Chem.MolFromSmiles(MAPPED_SMILES)
    if mol is None:
        raise ScientificProtocolError("frozen mapped SMILES is invalid")
    mol = Chem.AddHs(mol)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()]
    if maps != list(range(1, 14)):
        raise ScientificProtocolError(f"unexpected mapped-heavy-atom order: {maps}")
    return mol


def map_to_index(mol: Chem.Mol) -> dict[int, int]:
    result: dict[int, int] = {}
    for atom in mol.GetAtoms():
        number = atom.GetAtomMapNum()
        if number:
            if number in result:
                raise ScientificProtocolError("duplicate atom-map number")
            result[number] = atom.GetIdx()
    if sorted(result) != list(range(1, 14)):
        raise ScientificProtocolError("atom maps 1..13 are not preserved")
    return result


def target_indices(mol: Chem.Mol, *, one_based: bool = False) -> tuple[int, int, int, int]:
    mapping = map_to_index(mol)
    offset = 1 if one_based else 0
    return tuple(mapping[value] + offset for value in TARGET_DIHEDRAL_MAPS)  # type: ignore[return-value]


def molecule_signature(mol: Chem.Mol) -> dict[str, Any]:
    bonds = sorted(
        [min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]
        for bond in mol.GetBonds()
    )
    return {
        "atom_count": mol.GetNumAtoms(),
        "atom_symbols": [atom.GetSymbol() for atom in mol.GetAtoms()],
        "atom_map_numbers": [atom.GetAtomMapNum() for atom in mol.GetAtoms()],
        "atomic_numbers": [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        "composition": dict(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).items())),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "bonds_zero_based": bonds,
        "target_dihedral_maps": list(TARGET_DIHEDRAL_MAPS),
        "target_dihedral_indices_zero_based": list(target_indices(mol)),
        "target_dihedral_indices_one_based": list(target_indices(mol, one_based=True)),
    }


def _set_constrained_uff(mol: Chem.Mol, angle_deg: float) -> None:
    conf = mol.GetConformer()
    indices = target_indices(mol)
    target = signed_angle(angle_deg)
    rdMolTransforms.SetDihedralDeg(conf, *indices, target)
    field = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
    if field is None:
        raise ScientificProtocolError("UFF force field unavailable")
    # A narrow absolute window makes the preoptimization deterministic while
    # avoiding a zero-width constraint edge case in older RDKit releases.
    field.UFFAddTorsionConstraint(*indices, False, target - 0.01, target + 0.01, 500.0)
    status = int(field.Minimize(maxIts=2000, forceTol=1.0e-6, energyTol=1.0e-8))
    if status != 0:
        raise ScientificProtocolError("constrained UFF preoptimization did not converge")
    actual = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *indices)
    if circular_distance_deg(actual, target) > 0.05:
        raise ScientificProtocolError("constrained UFF torsion tolerance was not met")


def generate_reference() -> Chem.Mol:
    mol = _base_molecule()
    params = AllChem.ETKDGv3()
    params.randomSeed = REFERENCE_SEED
    params.useRandomCoords = False
    params.enforceChirality = True
    result = int(AllChem.EmbedMolecule(mol, params))
    if result != 0:
        raise ScientificProtocolError(f"ETKDGv3 embedding failed with code {result}")
    _set_constrained_uff(mol, REFERENCE_TORSION_DEG)
    return mol


def xyz_bytes(mol: Chem.Mol, comment: str) -> bytes:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]
    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2s} {point.x: .12f} {point.y: .12f} {point.z: .12f}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def read_xyz(path: Path, template: Chem.Mol | None = None) -> Chem.Mol:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ScientificProtocolError("XYZ is truncated")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ScientificProtocolError("XYZ atom count is invalid") from exc
    rows = lines[2 : 2 + count]
    if len(rows) != count:
        raise ScientificProtocolError("XYZ atom rows are truncated")
    mol = Chem.Mol(template if template is not None else _base_molecule())
    if mol.GetNumAtoms() != count:
        raise ScientificProtocolError("XYZ atom count changed")
    conf = Chem.Conformer(count)
    for index, row in enumerate(rows):
        fields = row.split()
        if len(fields) < 4 or fields[0] != mol.GetAtomWithIdx(index).GetSymbol():
            raise ScientificProtocolError("XYZ atom order or symbol changed")
        try:
            x, y, z = (float(fields[1]), float(fields[2]), float(fields[3]))
        except ValueError as exc:
            raise ScientificProtocolError("XYZ coordinate is invalid") from exc
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ScientificProtocolError("XYZ coordinate is non-finite")
        conf.SetAtomPosition(index, Point3D(x, y, z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def write_reference(case_root: Path) -> dict[str, Any]:
    case_root.mkdir(parents=True, exist_ok=True)
    mol = generate_reference()
    reference_path = case_root / "reference.xyz"
    structure_path = case_root / "reference_structure.json"
    if reference_path.exists() or structure_path.exists():
        raise FileExistsError("frozen reference already exists")
    reference_path.write_bytes(
        xyz_bytes(
            mol,
            "RDKit ETKDGv3 seed=20260720; constrained UFF; torsion maps=6,7,8,9; angle=90deg",
        )
    )
    signature = molecule_signature(mol)
    signature.update(
        {
            "schema_version": "1.0",
            "mapped_smiles": MAPPED_SMILES,
            "reference_seed": REFERENCE_SEED,
            "embed_algorithm": "ETKDGv3",
            "preoptimization": "constrained UFF",
            "reference_torsion_deg": REFERENCE_TORSION_DEG,
            "reference_xyz_sha256": sha256_file(reference_path),
            "measured_reference_torsion_deg": rdMolTransforms.GetDihedralDeg(
                mol.GetConformer(), *target_indices(mol)
            )
            % 360.0,
        }
    )
    write_json(structure_path, signature, exclusive=True)
    return signature


def molecule_from_reference(reference_path: Path) -> Chem.Mol:
    return read_xyz(reference_path, _base_molecule())


def prepare_angle_input(reference_path: Path, angle_deg: int, output_path: Path) -> dict[str, Any]:
    angle = validate_angle(angle_deg)
    mol = molecule_from_reference(reference_path)
    _set_constrained_uff(mol, angle)
    output_path.write_bytes(
        xyz_bytes(mol, f"independent reference initialization; constrained UFF; target={angle} deg")
    )
    measured = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *target_indices(mol)) % 360.0
    return {
        "angle_deg": angle,
        "uff_final_angle_deg": measured,
        "uff_angle_error_deg": circular_distance_deg(measured, angle),
        "input_sha256": sha256_file(output_path),
        "reference_sha256": sha256_file(reference_path),
    }


def constraint_text(angle_deg: int) -> str:
    angle = validate_angle(angle_deg)
    atoms = ", ".join(str(value) for value in target_indices(_base_molecule(), one_based=True))
    return (
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: {atoms}, {signed_angle(angle):.1f}\n"
        "$end\n"
    )


def locate_xtb() -> Path:
    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.append(Path(conda_prefix))
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        prefixes.append(Path(user_profile) / ".conda" / "envs" / "xtb-benchmark")
    candidates = []
    for prefix in prefixes:
        candidates.extend(
            [prefix / "Library" / "bin" / "xtb.exe", prefix / "Scripts" / "xtb.exe"]
        )
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == XTB_BINARY_SHA256:
            return candidate.resolve(strict=True)
    raise ScientificProtocolError("allowlisted xTB 6.7.1 executable was not found")


def xtb_version_record(executable: Path | None = None) -> dict[str, Any]:
    exe = (executable or locate_xtb()).resolve(strict=True)
    completed = subprocess.run(
        [str(exe), "--version"],
        cwd=str(exe.parent),
        env=_native_environment(),
        capture_output=True,
        timeout=30,
        shell=False,
    )
    output = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    match = re.search(r"xtb version\s+([0-9]+\.[0-9]+\.[0-9]+)", output, re.IGNORECASE)
    version = match.group(1) if match else None
    if completed.returncode != 0 or version != XTB_VERSION:
        raise ScientificProtocolError("xTB executable failed exact-version validation")
    return {
        "executable": str(exe),
        "version": version,
        "binary_sha256": sha256_file(exe),
        "version_exit_code": completed.returncode,
    }


def _native_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return env


def parse_total_energy(stdout: str, final_xyz: Path) -> tuple[float, float]:
    matches = re.findall(
        r"\|\s*TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh\s*\|",
        stdout,
    )
    if not matches:
        raise ScientificProtocolError("xTB TOTAL ENERGY was not found")
    stdout_energy = float(matches[-1])
    if not math.isfinite(stdout_energy):
        raise ScientificProtocolError("xTB TOTAL ENERGY is non-finite")
    lines = final_xyz.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ScientificProtocolError("xtbopt.xyz is truncated")
    comment_match = re.search(r"\benergy:\s*([-+0-9.Ee]+)", lines[1])
    if not comment_match:
        raise ScientificProtocolError("xtbopt.xyz energy is missing")
    xyz_energy = float(comment_match.group(1))
    if not math.isfinite(xyz_energy) or abs(stdout_energy - xyz_energy) > 1.0e-9:
        raise ScientificProtocolError("stdout and xtbopt.xyz energies disagree")
    return stdout_energy, xyz_energy


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def validate_structure(final_mol: Chem.Mol, expected: Chem.Mol) -> dict[str, Any]:
    expected_signature = molecule_signature(expected)
    final_signature = molecule_signature(final_mol)
    order_ok = final_signature["atom_symbols"] == expected_signature["atom_symbols"]
    composition_ok = final_signature["composition"] == expected_signature["composition"]
    charge_ok = final_signature["formal_charge"] == 0

    radii = {"H": 0.31, "C": 0.76, "F": 0.57}
    conf = final_mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(final_mol.GetNumAtoms())]
    inferred: set[tuple[int, int]] = set()
    for i in range(final_mol.GetNumAtoms()):
        for j in range(i + 1, final_mol.GetNumAtoms()):
            si = final_mol.GetAtomWithIdx(i).GetSymbol()
            sj = final_mol.GetAtomWithIdx(j).GetSymbol()
            cutoff = 1.25 * (radii[si] + radii[sj])
            if _distance(coords[i], coords[j]) <= cutoff:
                inferred.add((i, j))
    expected_bonds = {tuple(pair) for pair in expected_signature["bonds_zero_based"]}
    connectivity_ok = inferred == expected_bonds
    return {
        "atom_count_ok": final_mol.GetNumAtoms() == expected.GetNumAtoms(),
        "atom_order_ok": order_ok,
        "composition_ok": composition_ok,
        "formal_charge_ok": charge_ok,
        "connectivity_ok": connectivity_ok,
        "expected_bond_count": len(expected_bonds),
        "inferred_bond_count": len(inferred),
        "missing_bonds": [list(pair) for pair in sorted(expected_bonds - inferred)],
        "unexpected_bonds": [list(pair) for pair in sorted(inferred - expected_bonds)],
    }


def _artifact_manifest(attempt_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(attempt_root.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "attempt_manifest.json":
            files.append(
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {"schema_version": "1.0", "algorithm": "SHA-256", "files": files}


def run_native_scan(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Run one immutable native xTB constrained optimization."""

    angle = validate_angle(angle_deg)
    if attempt_root.exists() or attempt_root.is_symlink():
        raise FileExistsError(f"attempt already exists: {attempt_root.name}")
    attempt_root.mkdir(parents=True, exist_ok=False)
    input_path = attempt_root / "input.xyz"
    control_path = attempt_root / "xcontrol.inp"
    stdout_path = attempt_root / "stdout.log"
    stderr_path = attempt_root / "stderr.log"
    prepared = prepare_angle_input(reference_path, angle, input_path)
    control_path.write_text(constraint_text(angle), encoding="ascii", newline="\n")
    executable = locate_xtb()
    argv = [
        str(executable),
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
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=str(attempt_root),
            env=_native_environment(),
            capture_output=True,
            timeout=int(timeout_seconds),
            shell=False,
        )
        return_code = int(completed.returncode)
        stdout_bytes = completed.stdout
        stderr_bytes = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = -1
        stdout_bytes = exc.stdout or b""
        stderr_bytes = exc.stderr or b""
    duration = time.monotonic() - started
    stdout_path.write_bytes(stdout_bytes)
    stderr_path.write_bytes(stderr_bytes)
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    final_path = attempt_root / "xtbopt.xyz"
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in stdout and "FAILED TO CONVERGE" not in stdout
    energy: float | None = None
    final_angle: float | None = None
    structure_validation: dict[str, Any] = {
        "atom_count_ok": False,
        "atom_order_ok": False,
        "composition_ok": False,
        "formal_charge_ok": False,
        "connectivity_ok": False,
    }
    parse_error: str | None = None
    if final_path.is_file():
        try:
            energy, _ = parse_total_energy(stdout, final_path)
            expected = molecule_from_reference(reference_path)
            final_mol = read_xyz(final_path, expected)
            final_angle = rdMolTransforms.GetDihedralDeg(
                final_mol.GetConformer(), *target_indices(final_mol)
            ) % 360.0
            structure_validation = validate_structure(final_mol, expected)
        except ScientificProtocolError as exc:
            parse_error = str(exc)
    angle_error = (
        circular_distance_deg(final_angle, angle) if final_angle is not None else None
    )
    profile_eligible = bool(
        return_code == 0
        and not timed_out
        and converged
        and energy is not None
        and final_angle is not None
        and angle_error is not None
        and angle_error <= FINAL_ANGLE_TOLERANCE_DEG
        and all(
            structure_validation.get(key) is True
            for key in (
                "atom_count_ok",
                "atom_order_ok",
                "composition_ok",
                "formal_charge_ok",
                "connectivity_ok",
            )
        )
    )
    record: dict[str, Any] = {
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
            "atom_maps": list(TARGET_DIHEDRAL_MAPS),
            "atom_indices_one_based": list(target_indices(_base_molecule(), one_based=True)),
            "target_deg": angle,
            "force_constant_au": 1.0,
        },
        "reference_sha256": sha256_file(reference_path),
        "prepared_input": prepared,
        "argv": argv,
        "shell": False,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "xtb": xtb_version_record(executable),
        "timeout_seconds": int(timeout_seconds),
        "timed_out": timed_out,
        "duration_seconds": round(duration, 6),
        "return_code": return_code,
        "converged": converged,
        "total_energy_hartree": energy,
        "final_angle_deg": final_angle,
        "angle_error_deg": angle_error,
        "requested_final_angle_tolerance_deg": FINAL_ANGLE_TOLERANCE_DEG,
        "structure_validation": structure_validation,
        "parse_error": parse_error,
        "profile_eligible": profile_eligible,
    }
    write_json(attempt_root / "result.json", record, exclusive=True)
    manifest = _artifact_manifest(attempt_root)
    write_json(attempt_root / "attempt_manifest.json", manifest, exclusive=True)
    record["attempt_manifest_sha256"] = sha256_file(attempt_root / "attempt_manifest.json")
    return record


def verify_attempt_manifest(attempt_root: Path) -> bool:
    manifest_path = attempt_root / "attempt_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest == _artifact_manifest(attempt_root)


def verify_case_manifest(case_root: Path) -> bool:
    path = case_root / "case_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected_files = []
    for name in ("case_spec.json", "reference.xyz", "reference_structure.json"):
        member = case_root / name
        if not member.is_file() or member.is_symlink():
            return False
        expected_files.append(
            {"path": name, "size_bytes": member.stat().st_size, "sha256": sha256_file(member)}
        )
    expected = {"schema_version": "1.0", "algorithm": "SHA-256", "files": expected_files}
    return manifest == expected


__all__ = [
    "ANGLE_GRID_DEG",
    "AngleValidationError",
    "FINAL_ANGLE_TOLERANCE_DEG",
    "HARTREE_TO_KCAL_MOL",
    "MAPPED_SMILES",
    "REFERENCE_SEED",
    "ScientificProtocolError",
    "TARGET_DIHEDRAL_MAPS",
    "XTB_VERSION",
    "canonical_json_bytes",
    "circular_distance_deg",
    "constraint_text",
    "generate_reference",
    "locate_xtb",
    "map_to_index",
    "molecule_from_reference",
    "molecule_signature",
    "prepare_angle_input",
    "run_native_scan",
    "sha256_file",
    "target_indices",
    "validate_angle",
    "verify_attempt_manifest",
    "verify_case_manifest",
    "write_json",
    "write_reference",
    "xtb_version_record",
]
