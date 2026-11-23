"""Standard external-chemistry interface for benchmark candidates.

Candidate scripts must import these functions instead of importing PySCF/Psi4
or launching xTB/Psi4 directly. All paths are relative to the current workspace.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

WORKSPACE = Path.cwd().resolve()
PROTOCOL_PATH = WORKSPACE / "protocol.yaml"
INSTANCE_PATH = WORKSPACE / "instance.json"

__all__ = [
    "run_xtb_microtest", "run_xtb_optimization", "run_xtb_single_point",
    "run_pyscf_microtest", "run_pyscf_formal_single_point",
    "run_psi4_microtest", "run_psi4_sapt0",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return data


def _protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    return _load_yaml(PROTOCOL_PATH)


def _instance() -> dict[str, Any]:
    if not INSTANCE_PATH.exists():
        raise FileNotFoundError(INSTANCE_PATH)
    data = json.loads(INSTANCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("instance.json must contain an object")
    return data


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (WORKSPACE / p).resolve()


def _ensure_within_workspace(path: Path) -> None:
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside benchmark workspace: {path}") from exc


def _read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")
    count = int(lines[0].strip())
    rows = lines[2:2 + count]
    if len(rows) != count:
        raise ValueError(f"Truncated XYZ file: {path}")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: {row}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coords


def _metadata_lines(mapping: dict[str, Any]) -> str:
    return "\n".join(f"BENCHMARK_{key.upper()}: {value}" for key, value in mapping.items()) + "\n"


def _run(cmd: list[str], *, cwd: Path, timeout_seconds: int, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    _ensure_within_workspace(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=int(timeout_seconds),
        env=env,
        check=False,
    )


def run_xtb_microtest(output_dir: str = "microtests/xtb", timeout_seconds: int = 300) -> dict[str, Any]:
    out = _path(output_dir)
    _ensure_within_workspace(out)
    out.mkdir(parents=True, exist_ok=True)
    xyz = out / "h2.xyz"
    xyz.write_text("2\nbenchmark microtest\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    proc = _run(["xtb", xyz.name, "--gfn", "2", "--sp"], cwd=out, timeout_seconds=timeout_seconds, env=env)
    log = out / "xtb_microtest.log"
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    return {"returncode": proc.returncode, "log_file": str(log.relative_to(WORKSPACE)), "passed": proc.returncode == 0}


def run_xtb_optimization(
    input_xyz: str,
    output_dir: str,
    *,
    solvent: str | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    protocol = _protocol()
    cfg = protocol["gas_conformer_search"] if solvent is None else protocol["endpoint_conformational_probe"]
    charge = int(cfg["charge"])
    uhf = int(cfg["uhf"])
    opt_level = str(cfg["optimization_level"])
    threads = int(cfg["resources"]["threads_per_job"])
    timeout_seconds = max(int(timeout_seconds), int(protocol["timeouts"]["xtb_production_floor_seconds"]))
    if solvent is not None:
        aliases = {item["xtb_alpb_name"] for item in _instance()["solvents"].values()}
        if solvent not in aliases:
            raise ValueError(f"Unsupported ALPB solvent alias: {solvent}")

    src = _path(input_xyz)
    out = _path(output_dir)
    _ensure_within_workspace(src)
    _ensure_within_workspace(out)
    if not src.exists():
        raise FileNotFoundError(src)
    out.mkdir(parents=True, exist_ok=True)
    local_xyz = out / "input.xyz"
    shutil.copy2(src, local_xyz)
    cmd = ["xtb", local_xyz.name, "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf), "--opt", opt_level]
    if solvent:
        cmd.extend(["--alpb", solvent])
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    })
    proc = _run(cmd, cwd=out, timeout_seconds=timeout_seconds, env=env)
    log = out / "xtb.log"
    metadata = {
        "program": "xTB",
        "method": "GFN2-xTB",
        "solvent": solvent or "gas",
        "charge": charge,
        "uhf": uhf,
        "geometry_file": str(src.relative_to(WORKSPACE)),
        "returncode": proc.returncode,
    }
    log.write_text(_metadata_lines(metadata) + proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    opt = out / "xtbopt.xyz"
    return {
        "returncode": proc.returncode,
        "log_file": str(log.relative_to(WORKSPACE)),
        "optimized_geometry": str(opt.relative_to(WORKSPACE)) if opt.exists() else None,
        "passed": proc.returncode == 0 and opt.exists(),
    }


def run_xtb_single_point(
    input_xyz: str,
    output_dir: str,
    *,
    solvent: str | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run a fixed-geometry GFN2-xTB single point for interaction/deformation bookkeeping."""
    protocol = _protocol()
    cfg = protocol["gas_conformer_search"] if solvent is None else protocol["endpoint_conformational_probe"]
    charge = int(cfg["charge"])
    uhf = int(cfg["uhf"])
    threads = int(cfg["resources"]["threads_per_job"])
    timeout_seconds = max(int(timeout_seconds), int(protocol["timeouts"]["xtb_production_floor_seconds"]))
    if solvent is not None:
        aliases = {item["xtb_alpb_name"] for item in _instance()["solvents"].values()}
        if solvent not in aliases:
            raise ValueError(f"Unsupported ALPB solvent alias: {solvent}")
    src = _path(input_xyz)
    out = _path(output_dir)
    _ensure_within_workspace(src); _ensure_within_workspace(out)
    if not src.exists():
        raise FileNotFoundError(src)
    out.mkdir(parents=True, exist_ok=True)
    local_xyz = out / "input.xyz"
    shutil.copy2(src, local_xyz)
    cmd = ["xtb", local_xyz.name, "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf), "--sp"]
    if solvent:
        cmd.extend(["--alpb", solvent])
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS":str(threads),"MKL_NUM_THREADS":str(threads),"OPENBLAS_NUM_THREADS":str(threads)})
    proc = _run(cmd, cwd=out, timeout_seconds=timeout_seconds, env=env)
    text = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    energy_matches = re.findall(r"TOTAL ENERGY\s+([-+0-9.eEdD]+)|total energy\s*[:=]\s*([-+0-9.eEdD]+)", text, flags=re.I)
    energy = None
    if energy_matches:
        token = next((x for x in energy_matches[-1] if x), None)
        if token:
            energy = float(token.replace("D","E").replace("d","e"))
    log = out / "xtb_sp.log"
    metadata = {"program":"xTB","method":"GFN2-xTB","job_type":"single_point","solvent":solvent or "gas","charge":charge,"uhf":uhf,"geometry_file":str(src.relative_to(WORKSPACE)),"total_energy_hartree":energy,"returncode":proc.returncode}
    log.write_text(_metadata_lines(metadata)+text, encoding="utf-8")
    return {"returncode":proc.returncode,"energy_hartree":energy,"log_file":str(log.relative_to(WORKSPACE)),"passed":proc.returncode==0 and energy is not None}


def _pyscf_molecule(xyz_file: Path, *, basis: str, charge: int, multiplicity: int, output_file: Path, memory_mb: int):
    from pyscf import gto

    symbols, coords = _read_xyz(xyz_file)
    atom_spec = "; ".join(
        f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for symbol, xyz in zip(symbols, coords)
    )
    return gto.M(
        atom=atom_spec,
        basis=basis,
        unit="Angstrom",
        charge=int(charge),
        spin=int(multiplicity) - 1,
        verbose=4,
        output=str(output_file),
        max_memory=int(memory_mb),
    )


def run_pyscf_microtest(output_dir: str = "microtests/pyscf") -> dict[str, Any]:
    out = _path(output_dir)
    _ensure_within_workspace(out)
    out.mkdir(parents=True, exist_ok=True)
    xyz = out / "h2.xyz"
    xyz.write_text("2\nbenchmark microtest\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    log = out / "pyscf_microtest.log"
    from pyscf import scf

    mol = _pyscf_molecule(xyz, basis="sto-3g", charge=0, multiplicity=1, output_file=log, memory_mb=1000)
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-9
    energy = float(mf.kernel())
    mol.stdout.close()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(_metadata_lines({"program": "PySCF", "total_energy_hartree": energy, "converged": bool(mf.converged)}))
    return {"energy_hartree": energy, "converged": bool(mf.converged), "log_file": str(log.relative_to(WORKSPACE)), "passed": bool(mf.converged)}


def run_pyscf_formal_single_point(
    geometry_file: str,
    solvent: str,
    log_file: str,
    *,
    sensitivity_mode: str | None = None,
) -> dict[str, Any]:
    protocol = _protocol()
    cfg = protocol["formal_single_point"]
    permitted_modes = {None, "second_retained_conformer", "tighter_integration_grid", "alternate_scf_initial_guess", "repeated_single_point_from_saved_density"}
    if sensitivity_mode not in permitted_modes:
        raise ValueError(f"Unsupported sensitivity mode: {sensitivity_mode}")
    valid_solvents = {"gas"} | {item["smd_name"] for item in _instance()["solvents"].values()}
    if solvent not in valid_solvents:
        raise ValueError(f"Unsupported SMD solvent name: {solvent}")

    functional = str(cfg["method"])
    basis = str(cfg["basis"])
    charge = int(cfg["charge"])
    multiplicity = int(cfg["multiplicity"])
    grid_level = int(cfg["dft_grid"]["level"])
    if sensitivity_mode == "tighter_integration_grid":
        grid_level += 1
    conv_tol = float(cfg["scf"]["conv_tol"])
    max_cycle = int(cfg["scf"]["max_cycle"])
    init_guess = "atom" if sensitivity_mode == "alternate_scf_initial_guess" else str(cfg["scf"]["init_guess"])
    threads = int(cfg["resources"]["threads"])
    memory_mb = int(cfg["resources"]["memory_mb"])

    geom = _path(geometry_file)
    log = _path(log_file)
    _ensure_within_workspace(geom)
    _ensure_within_workspace(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    from pyscf import dft

    mol = _pyscf_molecule(geom, basis=basis, charge=charge, multiplicity=multiplicity, output_file=log, memory_mb=memory_mb)
    mf = dft.RKS(mol)
    mf.xc = functional
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.init_guess = init_guess
    mf.diis_space = int(cfg["scf"]["diis_space"])
    mf.level_shift = float(cfg["scf"]["level_shift"])
    mf.grids.level = grid_level

    if solvent != "gas":
        from pyscf.solvent import smd
        solvent_obj = smd.SMD(mol, solvent=solvent)
        mf = smd.smd_for_scf(mf, solvent_obj=solvent_obj)

    energy = float(mf.kernel())
    mo_energy = list(map(float, mf.mo_energy))
    mo_occ = list(map(float, mf.mo_occ))
    occupied = [e for e, occ in zip(mo_energy, mo_occ) if occ > 0]
    virtual = [e for e, occ in zip(mo_energy, mo_occ) if occ <= 0]
    homo = max(occupied) if occupied else None
    lumo = min(virtual) if virtual else None
    try:
        dipole = [float(x) for x in mf.dip_moment(unit="Debye", verbose=0)]
    except Exception:
        dipole = None
    try:
        mol.stdout.close()
    except Exception:
        pass

    metadata = {
        "program": "PySCF",
        "functional": functional,
        "basis": basis,
        "reference": "RKS",
        "solvent": solvent,
        "charge": charge,
        "multiplicity": multiplicity,
        "geometry_file": str(geom.relative_to(WORKSPACE)),
        "total_energy_hartree": f"{energy:.15f}",
        "converged": str(bool(mf.converged)).lower(),
        "grid_level": grid_level,
        "scf_conv_tol": conv_tol,
        "scf_max_cycle": max_cycle,
        "sensitivity_mode": sensitivity_mode or "common_reference",
    }
    if homo is not None:
        metadata["homo_hartree"] = f"{homo:.15f}"
    if lumo is not None:
        metadata["lumo_hartree"] = f"{lumo:.15f}"
    if dipole is not None:
        metadata["dipole_debye_xyz"] = ",".join(f"{x:.10f}" for x in dipole)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n" + _metadata_lines(metadata))
    return {
        "energy_hartree": energy,
        "converged": bool(mf.converged),
        "homo_hartree": homo,
        "lumo_hartree": lumo,
        "dipole_debye_xyz": dipole,
        "log_file": str(log.relative_to(WORKSPACE)),
        "sensitivity_mode": sensitivity_mode,
    }


def run_psi4_microtest(output_dir: str = "microtests/psi4", timeout_seconds: int = 300) -> dict[str, Any]:
    out = _path(output_dir)
    _ensure_within_workspace(out)
    out.mkdir(parents=True, exist_ok=True)
    inp = out / "psi4_microtest.in"
    log = out / "psi4_microtest.log"
    inp.write_text(
        "memory 500 mb\n"
        "molecule h2 {\n0 1\nH 0 0 0\nH 0 0 0.74\nunits angstrom\n}\n"
        "set basis sto-3g\nenergy('scf')\n",
        encoding="utf-8",
    )
    proc = _run(["psi4", "-n", "1", inp.name, log.name], cwd=out, timeout_seconds=timeout_seconds)
    if proc.stdout or proc.stderr:
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n--- WRAPPER STDOUT ---\n" + proc.stdout + "\n--- WRAPPER STDERR ---\n" + proc.stderr)
    return {"returncode": proc.returncode, "log_file": str(log.relative_to(WORKSPACE)), "passed": proc.returncode == 0}


def _fragment_block(symbols: Iterable[str], coords: Iterable[Iterable[float]]) -> str:
    return "\n".join(
        f"{symbol} {float(x):.10f} {float(y):.10f} {float(z):.10f}"
        for symbol, (x, y, z) in zip(symbols, coords)
    )


def run_psi4_sapt0(
    monomer_a_xyz: str,
    monomer_b_xyz: str,
    output_dir: str,
    *,
    solvent_label: str,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    protocol = _protocol()
    cfg = protocol["sapt"]
    basis = str(cfg["basis"])
    threads = int(cfg["resources"]["threads"])
    memory_mb = int(cfg["resources"]["memory_mb"])
    timeout_seconds = max(int(timeout_seconds), int(protocol["timeouts"]["psi4_production_floor_seconds"]))
    if solvent_label not in set(cfg["endpoints"] if "endpoints" in cfg else _instance()["methods"]["explicit_complex"]["endpoints"]):
        raise ValueError(f"Unsupported SAPT endpoint label: {solvent_label}")
    a_path = _path(monomer_a_xyz)
    b_path = _path(monomer_b_xyz)
    out = _path(output_dir)
    for path in (a_path, b_path, out):
        _ensure_within_workspace(path)
    out.mkdir(parents=True, exist_ok=True)
    a_sym, a_xyz = _read_xyz(a_path)
    b_sym, b_xyz = _read_xyz(b_path)
    inp = out / "sapt0.in"
    log = out / "sapt0.log"
    complex_xyz = out / "complex.xyz"
    complex_rows = [f"{len(a_sym) + len(b_sym)}", "benchmark SAPT complex"]
    complex_rows.extend(
        f"{symbol} {float(x):.10f} {float(y):.10f} {float(z):.10f}"
        for symbol, (x, y, z) in zip(a_sym, a_xyz)
    )
    complex_rows.extend(
        f"{symbol} {float(x):.10f} {float(y):.10f} {float(z):.10f}"
        for symbol, (x, y, z) in zip(b_sym, b_xyz)
    )
    complex_xyz.write_text("\n".join(complex_rows) + "\n", encoding="utf-8")
    inp.write_text(
        f"memory {int(memory_mb)} mb\n"
        "molecule complex {\n"
        "0 1\n"
        + _fragment_block(a_sym, a_xyz)
        + "\n--\n0 1\n"
        + _fragment_block(b_sym, b_xyz)
        + "\nunits angstrom\nno_reorient\nno_com\n}\n"
        "set {\n"
        f"  basis {basis}\n"
        "  scf_type df\n"
        "  freeze_core true\n"
        "  e_convergence 1e-8\n"
        "  d_convergence 1e-8\n"
        "  maxiter 100\n"
        "}\n"
        "energy('sapt0')\n",
        encoding="utf-8",
    )
    proc = _run(["psi4", "-n", str(int(threads)), inp.name, log.name], cwd=out, timeout_seconds=timeout_seconds)
    if proc.stdout or proc.stderr:
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n--- WRAPPER STDOUT ---\n" + proc.stdout + "\n--- WRAPPER STDERR ---\n" + proc.stderr)
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    number = r"([-+0-9.eEdD]+)"
    patterns = {
        "elst": rf"^\s*Electrostatics\s+[-+0-9.eEdD]+\s+\[mEh\]\s+{number}\s+\[kcal/mol\]",
        "exch": rf"^\s*Exchange\s+[-+0-9.eEdD]+\s+\[mEh\]\s+{number}\s+\[kcal/mol\]",
        "ind": rf"^\s*Induction\s+[-+0-9.eEdD]+\s+\[mEh\]\s+{number}\s+\[kcal/mol\]",
        "disp": rf"^\s*Dispersion\s+[-+0-9.eEdD]+\s+\[mEh\]\s+{number}\s+\[kcal/mol\]",
        "total": rf"^\s*Total SAPT0\s+[-+0-9.eEdD]+\s+\[mEh\]\s+{number}\s+\[kcal/mol\]",
    }
    values: dict[str, float] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.I | re.M)
        if matches:
            values[key] = float(matches[-1].replace("D", "E").replace("d", "e"))
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n" + _metadata_lines({
            "program": "Psi4",
            "method": "SAPT0",
            "basis": basis,
            "solvent": solvent_label,
            "geometry_file": str(complex_xyz.relative_to(WORKSPACE)),
            "fragment_a_geometry": str(a_path.relative_to(WORKSPACE)),
            "fragment_b_geometry": str(b_path.relative_to(WORKSPACE)),
            "fragment_a_atoms": len(a_sym),
            "fragment_b_atoms": len(b_sym),
            "returncode": proc.returncode,
        }))
        for key, value in values.items():
            label = {"elst": "SAPT_ELST_KCAL_MOL", "exch": "SAPT_EXCH_KCAL_MOL", "ind": "SAPT_IND_KCAL_MOL", "disp": "SAPT_DISP_KCAL_MOL", "total": "SAPT_TOTAL_KCAL_MOL"}[key]
            fh.write(f"BENCHMARK_{label}: {value}\n")
    return {
        "returncode": proc.returncode,
        "log_file": str(log.relative_to(WORKSPACE)),
        "components_kcal_mol": values,
        "passed": proc.returncode == 0 and "total" in values,
    }
