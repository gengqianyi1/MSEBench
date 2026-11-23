from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
EMBED_SEED = 20260719
PROBE_SEQUENCE = (1, 2, 3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n"))


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("task config must be a mapping")
    return loaded


def _resolve_conda(config: dict[str, Any]) -> tuple[str, str]:
    conda = shutil.which("conda.exe") or shutil.which("conda")
    if not conda:
        raise RuntimeError("CONDA_NOT_FOUND")
    environment_name = str(config["runtime"]["xtb"]["conda_environment"])
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError("CONDA_ENV_LIST_FAILED")
    prefixes = [Path(value) for value in json.loads(result.stdout.decode("utf-8"))["envs"]]
    matches = [path for path in prefixes if path.name.casefold() == environment_name.casefold()]
    if len(matches) != 1:
        raise RuntimeError("XTB_CONDA_ENV_NOT_FOUND")
    relative = Path("Library/bin/xtb.exe") if os.name == "nt" else Path("bin/xtb")
    executable = matches[0] / relative
    if not executable.is_file():
        raise RuntimeError("XTB_EXECUTABLE_NOT_FOUND")
    return conda, environment_name


def _native_environment(config: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    system_root = os.environ.get("SystemRoot")
    if system_root:
        environment["PATH"] = (
            str(Path(system_root) / "System32")
            + os.pathsep
            + environment.get("PATH", "")
        )
    for key, value in config["runtime"]["xtb"]["environment"].items():
        environment[str(key)] = str(value)
    return environment


def _generate_geometry() -> tuple[str, dict[str, Any]]:
    molecule = Chem.MolFromSmiles(ASPIRIN_SMILES)
    if molecule is None:
        raise RuntimeError("RDKIT_SMILES_PARSE_FAILED")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = EMBED_SEED
    parameters.numThreads = 1
    parameters.useRandomCoords = False
    status = AllChem.EmbedMolecule(molecule, parameters)
    if status != 0:
        raise RuntimeError("RDKIT_EMBED_FAILED")
    if AllChem.UFFHasAllMoleculeParams(molecule):
        optimize_status = AllChem.UFFOptimizeMolecule(molecule, maxIters=1000)
        force_field = "UFF"
    else:
        optimize_status = AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000)
        force_field = "MMFF"
    if optimize_status not in (0, 1):
        raise RuntimeError("RDKIT_PREOPT_FAILED")
    conformer = molecule.GetConformer()
    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    lines = [
        str(molecule.GetNumAtoms()),
        (
            f"aspirin canonical_smiles={canonical} rdkit={rdBase.rdkitVersion} "
            f"seed={EMBED_SEED} preopt={force_field}"
        ),
    ]
    for index, element in enumerate(elements):
        position = conformer.GetAtomPosition(index)
        lines.append(
            f"{element:<2s} {position.x: .10f} {position.y: .10f} {position.z: .10f}"
        )
    xyz = "\n".join(lines) + "\n"
    metadata = {
        "canonical_smiles": canonical,
        "atom_count": molecule.GetNumAtoms(),
        "element_sequence": elements,
        "rdkit_version": rdBase.rdkitVersion,
        "embed_algorithm": "ETKDGv3",
        "embed_seed": EMBED_SEED,
        "preoptimization": force_field,
        "preoptimization_status": optimize_status,
    }
    return xyz, metadata


def _complete_xyz_frames(path: Path, expected_atoms: int) -> int:
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    index = 0
    frames = 0
    while index < len(lines):
        try:
            atom_count = int(lines[index].strip())
        except ValueError:
            index += 1
            continue
        if atom_count != expected_atoms or index + atom_count + 1 >= len(lines):
            index += 1
            continue
        coordinates = lines[index + 2 : index + 2 + atom_count]
        if len(coordinates) != atom_count:
            index += 1
            continue
        valid = True
        for coordinate in coordinates:
            fields = coordinate.split()
            if len(fields) < 4:
                valid = False
                break
            try:
                values = [float(value) for value in fields[1:4]]
            except ValueError:
                valid = False
                break
            if not all(value == value and abs(value) != float("inf") for value in values):
                valid = False
                break
        if valid:
            frames += 1
            index += atom_count + 2
        else:
            index += 1
    return frames


def _run_xtb(
    *,
    conda: str,
    environment_name: str,
    config: dict[str, Any],
    directory: Path,
    cycle_limit: int,
    input_xyz: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    _write_text(directory / "input.xyz", input_xyz)
    _write_text(directory / "xcontrol.inp", f"$opt\n  maxcycle={cycle_limit}\n$end\n")
    safe_argv = [
        "conda",
        "run",
        "-n",
        environment_name,
        "--no-capture-output",
        "xtb",
        "input.xyz",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
        "--opt",
        "normal",
        "--input",
        "xcontrol.inp",
        "--parallel",
        "1",
    ]
    actual_argv = [conda] + safe_argv[1:]
    started = _utc_now()
    monotonic_start = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            actual_argv,
            cwd=directory,
            env=_native_environment(config),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=int(config["budget"]["single_xtb_timeout_seconds"]),
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = bytes(exc.stdout or b"")
        stderr = bytes(exc.stderr or b"")
    ended = _utc_now()
    elapsed = round(time.monotonic() - monotonic_start, 6)
    (directory / "stdout.log").write_bytes(stdout.replace(b"\r\n", b"\n"))
    (directory / "stderr.log").write_bytes(stderr.replace(b"\r\n", b"\n"))
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    converged = bool(
        re.search(r"GEOMETRY\s+OPTIMIZATION\s+CONVERGED", combined, re.IGNORECASE)
    )
    frames = _complete_xyz_frames(directory / "xtbopt.log", 21)
    return {
        "safe_argv": safe_argv,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "duration_seconds": elapsed,
        "process_exit_code": exit_code,
        "timed_out": timed_out,
        "optimization_converged": converged,
        "complete_trajectory_frames": frames,
    }


def _manifest_entries(
    root: Path,
    *,
    native_names: Iterable[str],
    preexisting_names: Iterable[str],
) -> list[dict[str, Any]]:
    native = set(native_names)
    preexisting = set(preexisting_names)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_type": (
                    "NATIVE_XTB"
                    if relative in native
                    else "BENCHMARK_GENERATED_METADATA"
                ),
                "existed_before_native_run": relative in preexisting,
            }
        )
    return entries


def _copy_probe_to_initial(
    probe: Path,
    initial: Path,
    run: dict[str, Any],
    config: dict[str, Any],
    cycle_limit: int,
) -> None:
    initial.mkdir(parents=True, exist_ok=False)
    generated_before_metadata: list[str] = []
    for source in sorted(probe.iterdir(), key=lambda value: value.name):
        if source.is_file():
            shutil.copy2(source, initial / source.name)
            if source.name not in {"input.xyz", "xcontrol.inp"}:
                generated_before_metadata.append(source.name)
    _write_json(
        initial / "command.json",
        {
            "schema_version": "1.0",
            "source_type": "NATIVE_XTB",
            "phase": "geometry_optimization",
            "argv": run["safe_argv"],
            "shell": False,
            "optimization_cycle_limit": cycle_limit,
            "scientific_parameters_source": "../../job_spec.json",
        },
    )
    _write_json(
        initial / "environment.json",
        {
            "schema_version": "1.0",
            "source_type": "BENCHMARK_GENERATED_METADATA",
            "runtime": "conda://xtb-benchmark",
            "xtb_version": str(config["runtime"]["xtb"]["version"]),
            "xtb_identity_policy": str(config["runtime"]["xtb"]["identity_policy"]),
            "thread_environment": config["runtime"]["xtb"]["environment"],
            "operating_system": platform.system(),
            "architecture": platform.machine(),
            "working_directory_policy": "isolated attempt directory",
        },
    )
    _write_json(
        initial / "scheduler.json",
        {
            "schema_version": "1.0",
            "source_type": "LOCAL_WRAPPER",
            "state": "COMPLETED" if not run["timed_out"] else "TIMEOUT",
            "scope": "execution_layer_only",
            "scientific_success_inference_allowed": False,
        },
    )
    _write_json(
        initial / "execution_summary.json",
        {
            "schema_version": "1.0",
            "source_type": "NATIVE_XTB",
            "phase": "geometry_optimization",
            "started_at_utc": run["started_at_utc"],
            "ended_at_utc": run["ended_at_utc"],
            "duration_seconds": run["duration_seconds"],
            "process_exit_code": run["process_exit_code"],
            "process_completed": not run["timed_out"],
            "timed_out": run["timed_out"],
            "optimization_cycle_limit": cycle_limit,
            "optimization_converged": run["optimization_converged"],
            "complete_trajectory_frames": run["complete_trajectory_frames"],
            "scientific_status": "OPTIMIZATION_NOT_CONVERGED",
        },
    )
    entries = _manifest_entries(
        initial,
        native_names=generated_before_metadata,
        preexisting_names={"input.xyz", "xcontrol.inp"},
    )
    _write_json(
        initial / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "source_type": "NATIVE_XTB",
            "attempt_id": "attempt_000_initial_failure",
            "provenance_state": "CURRENT_INITIAL_FAILURE",
            "files": entries,
        },
    )


def _build_job_spec(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "molecule": {
            "name": "aspirin",
            "canonical_smiles": geometry["canonical_smiles"],
            "charge": 0,
            "uhf": 0,
            "atom_count": geometry["atom_count"],
            "element_sequence": geometry["element_sequence"],
        },
        "method": "GFN2-xTB",
        "solvation": None,
        "required_phases": ["geometry_optimization", "hessian"],
        "optimization": {"level": "normal", "require_convergence": True},
        "hessian": {"required": True},
        "allowed_mutations": [
            "optimization_cycle_limit",
            "safe_restart_source",
            "thread_count",
            "walltime",
        ],
        "forbidden_mutations": [
            "molecule_composition",
            "atom_order",
            "charge",
            "uhf",
            "method",
            "solvation",
            "required_phases",
        ],
        "geometry_generation": {
            "rdkit_version": geometry["rdkit_version"],
            "embed_algorithm": geometry["embed_algorithm"],
            "embed_seed": geometry["embed_seed"],
            "preoptimization": geometry["preoptimization"],
        },
    }


def _build_case_manifest(case_dir: Path, cycle_limit: int, config: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(case_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "case_manifest.json":
            continue
        relative = path.relative_to(case_dir).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "1.0",
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "source_type": "NATIVE_XTB",
        "initial_failure": {
            "attempt_id": "attempt_000_initial_failure",
            "optimization_cycle_limit": cycle_limit,
            "verified_not_converged": True,
        },
        "legacy": {
            "provenance_state": "LEGACY_INDEPENDENT_SUCCESS",
            "allowed_as_current_success": False,
            "allowed_as_restart_source": False,
        },
        "xtb": {
            "version": str(config["runtime"]["xtb"]["version"]),
            "identity_policy": str(config["runtime"]["xtb"]["identity_policy"]),
            "required_help_tokens": list(config["runtime"]["xtb"]["required_help_tokens"]),
        },
        "files": files,
    }


def build_case(config_path: Path, case_dir: Path, development_root: Path) -> dict[str, Any]:
    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError("REFUSING_TO_OVERWRITE_EXISTING_CASE")
    if development_root.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_CASE_BUILD_EVIDENCE")
    development_root.mkdir(parents=True, exist_ok=False)
    config = _load_config(config_path)
    expected_rdkit = str(config["runtime"]["geometry_generation"]["rdkit_version"])
    if rdBase.rdkitVersion != expected_rdkit:
        raise RuntimeError("RDKIT_VERSION_MISMATCH")
    conda, environment_name = _resolve_conda(config)
    xyz, geometry = _generate_geometry()
    _write_text(development_root / "start.xyz", xyz)
    _write_json(development_root / "geometry_metadata.json", geometry)

    selected_cycle: int | None = None
    selected_probe: Path | None = None
    selected_run: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []
    for cycle_limit in PROBE_SEQUENCE:
        probe = development_root / f"probe_cycle_{cycle_limit}"
        run = _run_xtb(
            conda=conda,
            environment_name=environment_name,
            config=config,
            directory=probe,
            cycle_limit=cycle_limit,
            input_xyz=xyz,
        )
        probes.append(
            {
                "cycle_limit": cycle_limit,
                "process_exit_code": run["process_exit_code"],
                "timed_out": run["timed_out"],
                "optimization_converged": run["optimization_converged"],
                "complete_trajectory_frames": run["complete_trajectory_frames"],
                "stdout_sha256": _sha256(probe / "stdout.log"),
                "stderr_sha256": _sha256(probe / "stderr.log"),
            }
        )
        if (
            not run["timed_out"]
            and not run["optimization_converged"]
            and run["complete_trajectory_frames"] >= 1
            and (probe / "xtbopt.log").is_file()
        ):
            selected_cycle = cycle_limit
            selected_probe = probe
            selected_run = run
            break
    if selected_cycle is None or selected_probe is None or selected_run is None:
        _write_json(
            development_root / "build_summary.json",
            {"status": "BLOCKED_INITIAL_FIXTURE", "probes": probes},
        )
        raise RuntimeError("BLOCKED_INITIAL_FIXTURE")

    legacy_dir = development_root / "legacy_independent_success"
    legacy_run = _run_xtb(
        conda=conda,
        environment_name=environment_name,
        config=config,
        directory=legacy_dir,
        cycle_limit=int(config["runtime"]["xtb"]["restart_cycle_limit_default"]),
        input_xyz=xyz,
    )
    legacy_xyz = legacy_dir / "xtbopt.xyz"
    if (
        legacy_run["timed_out"]
        or not legacy_run["optimization_converged"]
        or not legacy_xyz.is_file()
        or _complete_xyz_frames(legacy_xyz, int(geometry["atom_count"])) != 1
    ):
        _write_json(
            development_root / "build_summary.json",
            {
                "status": "BLOCKED_INITIAL_FIXTURE",
                "reason": "LEGACY_NATIVE_SUCCESS_NOT_VERIFIED",
                "probes": probes,
                "legacy": legacy_run,
            },
        )
        raise RuntimeError("LEGACY_NATIVE_SUCCESS_NOT_VERIFIED")

    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "job_spec.json", _build_job_spec(geometry))
    _write_text(case_dir / "input" / "start.xyz", xyz)
    _copy_probe_to_initial(
        selected_probe,
        case_dir / "attempts" / "attempt_000_initial_failure",
        selected_run,
        config,
        selected_cycle,
    )
    legacy_target = case_dir / "legacy"
    legacy_target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(legacy_xyz, legacy_target / "xtbopt.xyz")
    _write_json(
        legacy_target / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "source_type": "NATIVE_XTB",
            "provenance_state": "LEGACY_INDEPENDENT_SUCCESS",
            "origin": "independent_native_optimization",
            "allowed_as_current_success": False,
            "allowed_as_restart_source": False,
            "files": [
                {
                    "path": "xtbopt.xyz",
                    "size_bytes": (legacy_target / "xtbopt.xyz").stat().st_size,
                    "sha256": _sha256(legacy_target / "xtbopt.xyz"),
                    "atom_count": geometry["atom_count"],
                    "element_sequence": geometry["element_sequence"],
                }
            ],
            "native_run_receipt": {
                "xtb_version": str(config["runtime"]["xtb"]["version"]),
                "xtb_identity_policy": str(config["runtime"]["xtb"]["identity_policy"]),
                "stdout_sha256": _sha256(legacy_dir / "stdout.log"),
                "stderr_sha256": _sha256(legacy_dir / "stderr.log"),
                "optimization_converged": True,
            },
        },
    )
    _write_json(
        case_dir / "case_manifest.json",
        _build_case_manifest(case_dir, selected_cycle, config),
    )
    summary = {
        "status": "BUILT",
        "initial_failure_cycle_limit": selected_cycle,
        "initial_complete_trajectory_frames": selected_run["complete_trajectory_frames"],
        "initial_process_exit_code": selected_run["process_exit_code"],
        "legacy_optimization_converged": True,
        "start_xyz_sha256": _sha256(case_dir / "input" / "start.xyz"),
        "job_spec_sha256": _sha256(case_dir / "job_spec.json"),
        "case_manifest_sha256": _sha256(case_dir / "case_manifest.json"),
        "probes": probes,
    }
    _write_json(development_root / "build_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen native xTB benchmark case")
    parser.add_argument("--config", default="configs/task.yaml")
    parser.add_argument("--case-dir", default="benchmark_case")
    parser.add_argument("--development-root", default="runs_failed/case_build_native")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    case_dir = (ROOT / args.case_dir).resolve()
    development_root = (ROOT / args.development_root).resolve()
    for path in (config_path, case_dir, development_root):
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("all paths must remain below project root") from exc
    if case_dir.is_dir() and any(case_dir.iterdir()):
        from validate_config import validate_configuration

        validation = validate_configuration(ROOT)
        result = {
            "status": "EXISTING_CASE_PRESERVED",
            "passed": bool(validation["passed"]),
            "config_status": validation["status"],
            "errors": validation["errors"],
        }
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    try:
        summary = build_case(config_path, case_dir, development_root)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": str(exc)}, ensure_ascii=True, sort_keys=True))
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
