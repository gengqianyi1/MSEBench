from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_initial_case import (
    ROOT,
    _load_config,
    _resolve_conda,
    _run_xtb,
    _sha256,
    _write_json,
)


def verify(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_REGENERATION_CHECK")
    config = _load_config(ROOT / "configs" / "task.yaml")
    conda, environment_name = _resolve_conda(config)
    input_path = ROOT / "benchmark_case" / "input" / "start.xyz"
    input_xyz = input_path.read_text(encoding="utf-8")
    cycle_limit = int(config["runtime"]["xtb"]["initial_failure_cycle_limit"])
    run = _run_xtb(
        conda=conda,
        environment_name=environment_name,
        config=config,
        directory=output,
        cycle_limit=cycle_limit,
        input_xyz=input_xyz,
    )
    stdout = (output / "stdout.log").read_text(encoding="utf-8")
    explicit_cycle = f"max. optcycles                     {cycle_limit}" in stdout
    explicit_failure = (
        f"FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN {cycle_limit} ITERATIONS" in stdout
    )
    passed = bool(
        not run["timed_out"]
        and not run["optimization_converged"]
        and run["complete_trajectory_frames"] >= 1
        and explicit_cycle
        and explicit_failure
    )
    result: dict[str, object] = {
        "schema_version": "1.0",
        "source_type": "NATIVE_XTB",
        "passed": passed,
        "initial_failure_cycle_limit": cycle_limit,
        "process_exit_code": run["process_exit_code"],
        "timed_out": run["timed_out"],
        "optimization_converged": run["optimization_converged"],
        "complete_trajectory_frames": run["complete_trajectory_frames"],
        "explicit_cycle_evidence": explicit_cycle,
        "explicit_failure_evidence": explicit_failure,
        "input_xyz_sha256": _sha256(input_path),
        "stdout_sha256": _sha256(output / "stdout.log"),
        "stderr_sha256": _sha256(output / "stderr.log"),
    }
    _write_json(output / "regeneration_check.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate and verify the frozen initial failure")
    parser.add_argument(
        "--output",
        default="runs_failed/initial_failure_regeneration_check",
    )
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve()
    try:
        output.relative_to(ROOT)
        report = verify(output)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=True, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
