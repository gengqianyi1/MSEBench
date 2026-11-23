from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib.metadata import version


EXPECTED = {
    "numpy": "2.2.6",
    "openai": "2.43.0",
    "PyYAML": "6.0.3",
    "rdkit": "2026.3.1",
}


def main() -> None:
    packages = {}
    ok = True
    for name, expected in EXPECTED.items():
        try:
            actual = version(name)
        except Exception as exc:
            packages[name] = {"ok": False, "error": repr(exc)}
            ok = False
            continue
        matches = actual == expected
        packages[name] = {"ok": matches, "version": actual, "expected": expected}
        ok = ok and matches

    xtb_path = shutil.which("xtb")
    xtb = {"ok": False, "path": xtb_path}
    if xtb_path:
        proc = subprocess.run(["xtb", "--version"], capture_output=True, text=True, timeout=60)
        output = proc.stdout + proc.stderr
        xtb.update(
            ok=proc.returncode == 0 and "6.7.1" in output,
            returncode=proc.returncode,
            output=output[-2000:],
        )
    ok = ok and bool(xtb["ok"])
    report = {"python": sys.version, "packages": packages, "xtb": xtb, "ok": ok}
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
