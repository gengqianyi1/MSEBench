from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 60) -> dict:
    path = shutil.which(cmd[0])
    out = {"cmd": cmd, "path": path}
    if path is None:
        out["ok"] = False
        out["error"] = f"executable not found: {cmd[0]}"
        return out
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        out.update(
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout[-4000:],
            stderr=proc.stderr[-4000:],
        )
    except Exception as exc:
        out.update(ok=False, error=repr(exc))
    return out


def check_imports() -> dict:
    mods = ["rdkit", "numpy", "pandas", "pyscf", "psi4", "openai", "yaml"]
    result = {}
    for name in mods:
        try:
            mod = __import__(name)
            result[name] = {
                "ok": True,
                "version": getattr(mod, "__version__", None),
                "file": getattr(mod, "__file__", None),
            }
        except Exception as exc:
            result[name] = {"ok": False, "error": repr(exc)}
    return result


def check_pyscf_smd() -> dict:
    try:
        from pyscf import dft, gto
        from pyscf.solvent import smd

        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
            basis="sto-3g",
            charge=0,
            spin=0,
            verbose=0,
        )
        mf0 = dft.RKS(mol)
        mf0.xc = "M05-2X"
        mf0.grids.level = 1
        try:
            mf = mf0.SMD()
            mf.with_solvent.solvent = "water"
        except Exception:
            smd_obj = smd.SMD(mol)
            smd_obj.solvent = "water"
            mf = mf0.SMD(smd_obj)
        energy = mf.kernel()
        return {
            "ok": bool(mf.converged),
            "converged": bool(mf.converged),
            "energy_hartree": float(energy),
            "libsolvent_enabled": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "libsolvent_enabled": False,
        }


def check_required_solvent_names() -> dict:
    smd_required = ["n-hexane", "1,4-dioxane", "chloroform", "water"]
    alpb_required = ["hexane", "dioxane", "chcl3", "water"]
    result = {
        "smd_required": smd_required,
        "alpb_required": alpb_required,
        "smd_missing": [],
        "alpb_missing": [],
    }
    try:
        from pyscf.solvent import smd

        result["smd_missing"] = [name for name in smd_required if name not in smd.solvent_db]
    except Exception as exc:
        result["smd_error"] = repr(exc)
        result["smd_missing"] = list(smd_required)

    try:
        proc = subprocess.run(["xtb", "--help"], text=True, capture_output=True, timeout=60)
        help_text = (proc.stdout + "\n" + proc.stderr).lower()
        result["alpb_missing"] = [name for name in alpb_required if name not in help_text]
        result["xtb_help_returncode"] = proc.returncode
    except Exception as exc:
        result["alpb_error"] = repr(exc)
        result["alpb_missing"] = list(alpb_required)

    result["ok"] = not result["smd_missing"] and not result["alpb_missing"]
    return result


def check_psi4_sapt() -> dict:
    script = r'''
import psi4
psi4.core.set_output_file("psi4_sapt_smoke.log", False)
psi4.set_memory("500 MB")
psi4.set_num_threads(1)
mol = psi4.geometry("""
0 1
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.586000
H  0.000000 -0.757000  0.586000
--
0 1
O  0.000000  0.000000  2.900000
H  0.000000  0.757000  3.486000
H  0.000000 -0.757000  3.486000
units angstrom
no_reorient
no_com
""")
e = psi4.energy("sapt0/jun-cc-pvdz", molecule=mol)
print(e)
'''
    path = Path("_psi4_sapt_smoke.py")
    path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, str(path)], text=True, capture_output=True, timeout=240)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "output_file": "psi4_sapt_smoke.log",
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip the Psi4 SAPT smoke test.")
    args = parser.parse_args()

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "imports": check_imports(),
        "executables": {
            "xtb": run_cmd(["xtb", "--version"]),
            "psi4": run_cmd(["psi4", "--version"]),
        },
        "pyscf_smd_smoke": check_pyscf_smd(),
        "required_solvent_names": check_required_solvent_names(),
    }
    if not args.quick:
        report["psi4_sapt_smoke"] = check_psi4_sapt()

    print(json.dumps(report, ensure_ascii=False, indent=2))

    required = [
        report["imports"].get("rdkit", {}).get("ok"),
        report["imports"].get("pyscf", {}).get("ok"),
        report["imports"].get("psi4", {}).get("ok"),
        report["executables"]["xtb"].get("ok"),
        report["executables"]["psi4"].get("ok"),
        report["pyscf_smd_smoke"].get("ok"),
        report["required_solvent_names"].get("ok"),
    ]
    if not all(required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
