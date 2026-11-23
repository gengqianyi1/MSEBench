from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from geometry import read_xyz, write_xyz, make_slipped_cofacial_dimer, write_xcontrol_split


class XTBError(RuntimeError):
    pass


def _run_command(cmd, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run external command and capture stdout/stderr as bytes.

    On Windows, subprocess text=True decodes output using the system locale
    such as GBK. xTB output can contain non-GBK bytes, causing
    UnicodeDecodeError in reader threads. Capturing bytes and decoding with
    errors="replace" avoids this and preserves logs.
    """
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=False,
        timeout=timeout,
    )


def _combined_output_text(proc: subprocess.CompletedProcess) -> str:
    stdout = proc.stdout or b""
    stderr = proc.stderr or b""

    if isinstance(stdout, str):
        stdout_text = stdout
    else:
        stdout_text = stdout.decode("utf-8", errors="replace")

    if isinstance(stderr, str):
        stderr_text = stderr
    else:
        stderr_text = stderr.decode("utf-8", errors="replace")

    return stdout_text + "\n" + stderr_text


def _copy_input_xyz(input_xyz: Path, workdir: Path, name: str = "input.xyz") -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / name
    shutil.copy2(input_xyz, dst)
    return dst


def parse_xtb_energy_hartree(output_text: str) -> Optional[float]:
    """Parse xTB total energy in Hartree from stdout.

    xTB output formats vary, so this parser tries several common patterns.
    """
    patterns = [
        r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh",
        r"\|\s*TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh",
        r"total energy\s+([-+]?\d+\.\d+)",
        r":: total energy\s+([-+]?\d+\.\d+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, output_text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def parse_dipro_hole_coupling_ev(output_text: str) -> Optional[float]:
    """Parse only the occupied-MO hole-transport summary from xTB DIPRO.

    There is deliberately no keyword or numeric fallback.  In particular,
    command echoes such as ``--dipro 0.1``, the DIPRO energy threshold, and
    charge/electron/unoccupied-MO transport rows are not hole couplings.
    Missing or unrecognised summary output therefore returns ``None``.
    """
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    patterns = (
        # xTB 6.6/6.7 production summary format.
        re.compile(
            rf"total\s+\|J\(AB\s*,\s*eff\)\|\s+for\s+hole\s+transport\s*"
            rf"\(\s*occ\.\s*MOs\s*\)\s*:\s*{number}\s*(meV|eV)\b",
            re.IGNORECASE,
        ),
        # Accepted labelled fixture format; still requires occupied-MO hole transport.
        re.compile(
            rf"hole\s+transport\s*\(\s*occ\.\s*MOs\s*\)\s*:\s*"
            rf"(?:\d+\s*->\s*\d+\s+)?(?:V\s*=\s*)?{number}\s*(meV|eV)\b",
            re.IGNORECASE,
        ),
    )
    matches = []
    for line in output_text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                value = abs(float(match.group(1)))
                unit = match.group(2).lower()
                matches.append(value / 1000.0 if unit == "mev" else value)
                break
    return matches[-1] if matches else None


def _float_list(value: Any, default: float) -> List[float]:
    if value is None:
        return [float(default)]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def run_xtb_singlepoint(
    input_xyz: Path,
    workdir: Path,
    *,
    executable: str,
    gfn: int,
    charge: int,
    uhf: int,
    timeout: int,
) -> Dict[str, Any]:
    xyz = _copy_input_xyz(input_xyz, workdir)
    cmd = [executable, xyz.name, "--gfn", str(gfn), "--chrg", str(charge), "--uhf", str(uhf)]
    proc = _run_command(cmd, workdir, timeout)
    out = _combined_output_text(proc)
    (workdir / "xtb.out").write_text(out, encoding="utf-8", errors="replace")
    energy = parse_xtb_energy_hartree(out)
    ok = proc.returncode == 0 and energy is not None
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "cmd": cmd,
        "energy_hartree": energy,
        "stdout_stderr_file": str(workdir / "xtb.out"),
        "error": None if ok else "xTB single-point failed or energy could not be parsed.",
    }


def search_lowest_energy_packing(
    neutral_opt_xyz: Path,
    workdir: Path,
    packing_cfg: Dict[str, Any],
    xtb_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    mono = read_xyz(neutral_opt_xyz)
    search_cfg = packing_cfg.get("search", {})

    dx_values = _float_list(search_cfg.get("dx_values_A"), packing_cfg["dx_A"])
    dy_values = _float_list(search_cfg.get("dy_values_A"), packing_cfg["dy_A"])
    dz_values = _float_list(search_cfg.get("dz_values_A"), packing_cfg["dz_A"])

    exe = xtb_cfg.get("executable", "xtb")
    gfn = int(xtb_cfg.get("gfn", 2))
    timeout = int(search_cfg.get("timeout_seconds", xtb_cfg.get("timeout_seconds", 1800)))

    trials = []
    best = None
    trial_idx = 0
    for dx in dx_values:
        for dy in dy_values:
            for dz in dz_values:
                trial_idx += 1
                trial_dir = workdir / f"trial_{trial_idx:03d}_dx_{dx:g}_dy_{dy:g}_dz_{dz:g}"
                dimer = make_slipped_cofacial_dimer(mono, dx=dx, dy=dy, dz=dz)
                dimer_path = trial_dir / "dimer.xyz"
                trial_dir.mkdir(parents=True, exist_ok=True)
                write_xyz(dimer_path, dimer)
                result = run_xtb_singlepoint(
                    dimer_path,
                    trial_dir,
                    executable=exe,
                    gfn=gfn,
                    charge=0,
                    uhf=0,
                    timeout=timeout,
                )
                trial = {
                    "dx_A": dx,
                    "dy_A": dy,
                    "dz_A": dz,
                    "ok": result["ok"],
                    "energy_hartree": result["energy_hartree"],
                    "dimer_xyz": str(dimer_path),
                    "details": result,
                }
                trials.append(trial)
                if trial["ok"] and (best is None or trial["energy_hartree"] < best["energy_hartree"]):
                    best = trial

    if best is None:
        return {
            "ok": False,
            "trials": trials,
            "error": "No packing search trial completed successfully.",
        }

    best_dimer = make_slipped_cofacial_dimer(mono, dx=best["dx_A"], dy=best["dy_A"], dz=best["dz_A"])
    best_dimer_path = workdir / "dimer_lowest_energy.xyz"
    write_xyz(best_dimer_path, best_dimer)
    return {
        "ok": True,
        "selected_dx_A": best["dx_A"],
        "selected_dy_A": best["dy_A"],
        "selected_dz_A": best["dz_A"],
        "selected_energy_hartree": best["energy_hartree"],
        "selected_dimer_xyz": str(best_dimer_path),
        "n_trials": len(trials),
        "n_successful_trials": sum(1 for trial in trials if trial["ok"]),
        "trials": trials,
        "error": None,
    }


def run_xtb_opt(
    input_xyz: Path,
    workdir: Path,
    *,
    executable: str,
    gfn: int,
    charge: int,
    uhf: int,
    opt_level: str,
    timeout: int,
) -> Dict[str, Any]:
    xyz = _copy_input_xyz(input_xyz, workdir)
    cmd = [
        executable,
        xyz.name,
        "--gfn", str(gfn),
        "--opt", str(opt_level),
        "--chrg", str(charge),
        "--uhf", str(uhf),
    ]
    proc = _run_command(cmd, workdir, timeout)
    out = _combined_output_text(proc)
    (workdir / "xtb.out").write_text(out, encoding="utf-8", errors="replace")
    energy = parse_xtb_energy_hartree(out)
    opt_xyz = workdir / "xtbopt.xyz"
    ok = proc.returncode == 0 and energy is not None and opt_xyz.exists()
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "cmd": cmd,
        "energy_hartree": energy,
        "opt_xyz": str(opt_xyz) if opt_xyz.exists() else None,
        "stdout_stderr_file": str(workdir / "xtb.out"),
        "error": None if ok else "xTB optimization failed, xtbopt.xyz missing, or energy could not be parsed.",
    }


def compute_reorganization_energy(
    initial_xyz: Path,
    workdir: Path,
    xtb_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    exe = xtb_cfg.get("executable", "xtb")
    gfn = int(xtb_cfg.get("gfn", 2))
    opt_level = xtb_cfg.get("opt_level", "tight")
    timeout = int(xtb_cfg.get("timeout_seconds", 1800))
    hartree_to_ev = float(xtb_cfg.get("hartree_to_ev", 27.211386245988))

    neutral_opt = run_xtb_opt(
        initial_xyz,
        workdir / "neutral_opt",
        executable=exe,
        gfn=gfn,
        charge=int(xtb_cfg.get("neutral_charge", 0)),
        uhf=int(xtb_cfg.get("neutral_uhf", 0)),
        opt_level=opt_level,
        timeout=timeout,
    )
    if not neutral_opt["ok"]:
        return {"ok": False, "stage": "neutral_opt", "error": neutral_opt["error"], "details": neutral_opt}

    cation_opt = run_xtb_opt(
        Path(neutral_opt["opt_xyz"]),
        workdir / "cation_opt",
        executable=exe,
        gfn=gfn,
        charge=int(xtb_cfg.get("cation_charge", 1)),
        uhf=int(xtb_cfg.get("cation_uhf", 1)),
        opt_level=opt_level,
        timeout=timeout,
    )
    if not cation_opt["ok"]:
        return {"ok": False, "stage": "cation_opt", "error": cation_opt["error"], "details": cation_opt}

    e0_q0 = neutral_opt["energy_hartree"]
    ep_qp = cation_opt["energy_hartree"]

    ep_q0 = run_xtb_singlepoint(
        Path(neutral_opt["opt_xyz"]),
        workdir / "sp_cation_at_neutral",
        executable=exe,
        gfn=gfn,
        charge=int(xtb_cfg.get("cation_charge", 1)),
        uhf=int(xtb_cfg.get("cation_uhf", 1)),
        timeout=timeout,
    )
    if not ep_q0["ok"]:
        return {"ok": False, "stage": "sp_cation_at_neutral", "error": ep_q0["error"], "details": ep_q0}

    e0_qp = run_xtb_singlepoint(
        Path(cation_opt["opt_xyz"]),
        workdir / "sp_neutral_at_cation",
        executable=exe,
        gfn=gfn,
        charge=int(xtb_cfg.get("neutral_charge", 0)),
        uhf=int(xtb_cfg.get("neutral_uhf", 0)),
        timeout=timeout,
    )
    if not e0_qp["ok"]:
        return {"ok": False, "stage": "sp_neutral_at_cation", "error": e0_qp["error"], "details": e0_qp}

    lambda_h = ((ep_q0["energy_hartree"] - ep_qp) + (e0_qp["energy_hartree"] - e0_q0)) * hartree_to_ev

    return {
        "ok": lambda_h > 0,
        "stage": "reorganization_energy",
        "lambda_h_eV": lambda_h,
        "neutral_opt_xyz": neutral_opt["opt_xyz"],
        "cation_opt_xyz": cation_opt["opt_xyz"],
        "energies_hartree": {
            "E_neutral_Q0": e0_q0,
            "E_plus_Qplus": ep_qp,
            "E_plus_Q0": ep_q0["energy_hartree"],
            "E_neutral_Qplus": e0_qp["energy_hartree"],
        },
        "details": {
            "neutral_opt": neutral_opt,
            "cation_opt": cation_opt,
            "sp_cation_at_neutral": ep_q0,
            "sp_neutral_at_cation": e0_qp,
        },
        "error": None if lambda_h > 0 else "Computed lambda_h_eV <= 0.",
    }


def run_dipro(
    neutral_opt_xyz: Path,
    workdir: Path,
    packing_cfg: Dict[str, Any],
    dipro_cfg: Dict[str, Any],
    xtb_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    mono = read_xyz(neutral_opt_xyz)
    packing_search = None
    if bool(packing_cfg.get("search", {}).get("enabled", False)):
        packing_search = search_lowest_energy_packing(
            neutral_opt_xyz,
            workdir / "packing_search",
            packing_cfg,
            xtb_cfg,
        )
        if not packing_search["ok"]:
            return {
                "ok": False,
                "returncode": None,
                "cmd": None,
                "dimer_xyz": None,
                "xcontrol": None,
                "Vh_DIPRO_eV": None,
                "Vh_DIPRO_meV": None,
                "stdout_stderr_file": None,
                "packing_search": packing_search,
                "error": packing_search["error"],
            }
        dimer = make_slipped_cofacial_dimer(
            mono,
            dx=float(packing_search["selected_dx_A"]),
            dy=float(packing_search["selected_dy_A"]),
            dz=float(packing_search["selected_dz_A"]),
        )
        dimer_path = workdir / "dimer_lowest_energy_packing.xyz"
        write_xyz(dimer_path, dimer)
    else:
        dimer = make_slipped_cofacial_dimer(
            mono,
            dx=float(packing_cfg["dx_A"]),
            dy=float(packing_cfg["dy_A"]),
            dz=float(packing_cfg["dz_A"]),
        )
        dimer_path = workdir / "dimer_standardized.xyz"
        write_xyz(dimer_path, dimer)

    xcontrol = workdir / "xcontrol"
    write_xcontrol_split(xcontrol, len(mono.elements))

    exe = xtb_cfg.get("executable", "xtb")
    gfn = int(xtb_cfg.get("gfn", 2))
    timeout = int(dipro_cfg.get("timeout_seconds", xtb_cfg.get("timeout_seconds", 1800)))
    dipro_param = str(dipro_cfg.get("dipro_parameter", 0.1))
    cmd = [
        exe,
        dimer_path.name,
        "--gfn", str(gfn),
        "--dipro", dipro_param,
        "--chrg", "0",
        "--uhf", "0",
        "-I", xcontrol.name,
    ]
    proc = _run_command(cmd, workdir, timeout)
    out = _combined_output_text(proc)
    (workdir / "dipro.out").write_text(out, encoding="utf-8", errors="replace")

    vh = parse_dipro_hole_coupling_ev(out)
    ok = proc.returncode == 0 and vh is not None and vh > 0
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "cmd": cmd,
        "dimer_xyz": str(dimer_path),
        "xcontrol": str(xcontrol),
        "packing_search": packing_search,
        "Vh_DIPRO_eV": vh,
        "Vh_DIPRO_meV": None if vh is None else vh * 1000.0,
        "stdout_stderr_file": str(workdir / "dipro.out"),
        "error": None if ok else "DIPRO failed or hole coupling could not be parsed.",
    }


def marcus_hole_rate(lambda_h_eV: float, Vh_eV: float, marcus_cfg: Dict[str, Any]) -> float:
    T = float(marcus_cfg.get("temperature_K", 300.0))
    kB = float(marcus_cfg.get("kB_eV_per_K", 8.617333262145e-5))
    hbar = float(marcus_cfg.get("hbar_eV_s", 6.582119569e-16))
    lam = float(lambda_h_eV)
    V = float(Vh_eV)
    if lam <= 0 or V <= 0:
        raise ValueError("lambda_h_eV and Vh_eV must be positive.")
    return (2.0 * math.pi / hbar) * (V ** 2) / math.sqrt(4.0 * math.pi * lam * kB * T) * math.exp(-lam / (4.0 * kB * T))
