from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment(config: dict[str, Any]) -> dict[str, str]:
    result = os.environ.copy()
    system_root = os.environ.get("SystemRoot")
    if system_root:
        result["PATH"] = str(Path(system_root) / "System32") + os.pathsep + result.get("PATH", "")
    for key, value in config["runtime"]["xtb"]["environment"].items():
        result[str(key)] = str(value)
    result["PYTHONNOUSERSITE"] = "1"
    result.pop("PYTHONPATH", None)
    return result


def _conda() -> str:
    resolved = shutil.which("conda.exe") or shutil.which("conda")
    if not resolved:
        raise RuntimeError("CONDA_NOT_FOUND")
    return resolved


def run_smoke(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_NATIVE_SMOKE")
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    job = json.loads((ROOT / "benchmark_case" / "job_spec.json").read_text(encoding="utf-8"))
    legacy_manifest = json.loads(
        (ROOT / "benchmark_case" / "legacy" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if legacy_manifest.get("native_run_receipt", {}).get("optimization_converged") is not True:
        raise RuntimeError("LEGACY_OPTIMIZATION_NOT_VERIFIED")
    source = ROOT / "benchmark_case" / "legacy" / "xtbopt.xyz"
    expected_hash = legacy_manifest["files"][0]["sha256"]
    if _sha256(source) != expected_hash:
        raise RuntimeError("LEGACY_HASH_MISMATCH")
    shutil.copy2(source, output / "input.xyz")

    environment_name = str(config["runtime"]["xtb"]["conda_environment"])
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
        str(job["molecule"]["charge"]),
        "--uhf",
        str(job["molecule"]["uhf"]),
        "--hess",
        "--parallel",
        str(config["runtime"]["xtb"]["thread_count_default"]),
    ]
    started = _utc_now()
    timer = time.monotonic()
    completed = subprocess.run(
        [_conda()] + safe_argv[1:],
        cwd=output,
        env=_environment(config),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=int(config["budget"]["single_xtb_timeout_seconds"]),
    )
    ended = _utc_now()
    (output / "stdout.log").write_bytes(completed.stdout.replace(b"\r\n", b"\n"))
    (output / "stderr.log").write_bytes(completed.stderr.replace(b"\r\n", b"\n"))
    combined = (completed.stdout + b"\n" + completed.stderr).decode("utf-8", errors="replace").casefold()
    hessian = output / "hessian"
    vibspectrum = output / "vibspectrum"
    hessian_complete = bool(
        completed.returncode == 0
        and "normal termination of xtb" in combined
        and hessian.is_file()
        and hessian.stat().st_size > 0
        and vibspectrum.is_file()
        and vibspectrum.stat().st_size > 0
    )
    files = []
    for path in sorted(output.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name not in {"artifact_manifest.json", "execution_summary.json"}:
            files.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "source_type": "NATIVE_XTB" if path.name != "input.xyz" else "VERIFIED_LEGACY_SOURCE",
                }
            )
    summary = {
        "schema_version": "1.0",
        "source_type": "NATIVE_XTB",
        "phase": "hessian",
        "argv": safe_argv,
        "shell": False,
        "started_at_utc": started,
        "ended_at_utc": ended,
        "duration_seconds": round(time.monotonic() - timer, 6),
        "process_exit_code": completed.returncode,
        "normal_termination": "normal termination of xtb" in combined,
        "hessian_artifact_present": hessian.is_file() and hessian.stat().st_size > 0,
        "vibspectrum_artifact_present": vibspectrum.is_file() and vibspectrum.stat().st_size > 0,
        "hessian_completed": hessian_complete,
        "scientific_parameters": {
            "method": job["method"],
            "charge": job["molecule"]["charge"],
            "uhf": job["molecule"]["uhf"],
            "solvation": job["solvation"],
        },
    }
    _write_json(output / "execution_summary.json", summary)
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "source_type": "NATIVE_XTB",
            "purpose": "native_hessian_smoke",
            "files": files,
        },
    )
    return {
        "passed": hessian_complete,
        "optimization_source_verified": True,
        "hessian_completed": hessian_complete,
        "process_exit_code": completed.returncode,
        "artifact_count": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one native xTB Hessian smoke test")
    parser.add_argument("--output", default="runs_failed/native_hessian_smoke")
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve()
    try:
        output.relative_to(ROOT)
        result = run_smoke(output)
    except (ValueError, FileExistsError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=True, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
