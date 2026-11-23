"""Trusted in-process state for one disposable harness run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Operation = Literal["sp", "opt", "hess", "scan"]


def count_requested_scan_points(control_text: str) -> int:
    """Count xTB scan schedule points without interpreting the scientific coordinate."""

    inside_scan = False
    total = 0
    for raw_line in control_text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered == "$scan":
            inside_scan = True
            continue
        if inside_scan and lowered == "$end":
            inside_scan = False
            continue
        if not inside_scan or not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"\d+\s*:\s*[-+0-9.eE]+\s*,\s*[-+0-9.eE]+\s*,\s*(\d+)",
            line,
        )
        if match is None:
            raise ValueError(f"unsupported xTB scan schedule line: {line}")
        points = int(match.group(1))
        if points < 1:
            raise ValueError("xTB scan point count must be positive")
        total += points
    return total


def resolve_executable(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise RuntimeError(f"Executable not found: {candidate}")
        return candidate.resolve()

    resolved = shutil.which(str(value))
    if resolved is None:
        raise RuntimeError(f"Executable not found on PATH: {value}")
    return Path(resolved).resolve()


@dataclass(frozen=True)
class JobRequest:
    operation: Operation
    input_path: str
    charge: int = 0
    uhf: int = 0
    solvent: str | None = None
    control_path: str | None = None


def build_xtb_command(
    executable: Path,
    input_name: str,
    request: JobRequest,
) -> list[str]:
    command = [
        str(executable),
        input_name,
        "--chrg",
        str(request.charge),
        "--uhf",
        str(request.uhf),
    ]
    if request.solvent is not None:
        command.extend(("--alpb", request.solvent))
    if request.control_path is not None:
        command.extend(("--input", Path(request.control_path).name))
    command.append("--opt" if request.operation == "scan" else f"--{request.operation}")
    return command


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class RunState:
    def __init__(
        self,
        run_dir: Path,
        main_budget: int,
        recovery_budget: int,
        xtb_executable: Path,
        timeout_seconds: int,
        *,
        operation_costs: dict[str, int] | None = None,
        allowed_operations: frozenset[str] | None = None,
        max_scan_points: int | None = None,
    ) -> None:
        if min(main_budget, recovery_budget) < 0:
            raise ValueError("budgets must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        costs = operation_costs or {"sp": 1, "opt": 1, "hess": 4, "scan": 1}
        if not costs or any(cost <= 0 for cost in costs.values()):
            raise ValueError("operation costs must be positive")
        resolved_executable = resolve_executable(xtb_executable)
        self.run_dir = Path(run_dir).resolve()
        self.xtb_executable = resolved_executable
        self.timeout_seconds = timeout_seconds
        self.operation_costs = dict(costs)
        self.allowed_operations = allowed_operations or frozenset(costs)
        if max_scan_points is not None and max_scan_points < 0:
            raise ValueError("scan-point budget must be non-negative")
        self.max_scan_points = max_scan_points
        self.scan_points_used = 0
        self.main_budget = main_budget
        self.recovery_budget = recovery_budget
        self.main_used = 0
        self.recovery_used = 0
        self._next_job = 1
        self._records: list[dict[str, object]] = []
        self._ended = False
        (self.run_dir / "jobs").mkdir(exist_ok=True)

    @classmethod
    def from_existing(
        cls,
        run_dir: Path,
        main_budget: int,
        recovery_budget: int,
        xtb_executable: Path,
        timeout_seconds: int,
        *,
        operation_costs: dict[str, int] | None = None,
        allowed_operations: frozenset[str] | None = None,
        max_scan_points: int | None = None,
    ) -> RunState:
        state = cls(
            run_dir,
            main_budget,
            recovery_budget,
            xtb_executable,
            timeout_seconds,
            operation_costs=operation_costs,
            allowed_operations=allowed_operations,
            max_scan_points=max_scan_points,
        )
        jobs_path = state.run_dir / "jobs.json"
        records = json.loads(jobs_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("jobs.json must contain an array")
        used = 0
        restored: list[dict[str, object]] = []
        scan_points_used = 0
        for number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError("jobs.json contains a non-object record")
            expected_id = f"job_{number:03d}"
            if record.get("job_id") != expected_id:
                raise ValueError("jobs.json job IDs must be sequential")
            operation = record.get("operation")
            if operation not in state.allowed_operations or operation not in state.operation_costs:
                raise ValueError(f"jobs.json contains unsupported operation: {operation}")
            if not (state.run_dir / "jobs" / expected_id).is_dir():
                raise FileNotFoundError(f"missing authoritative job directory: {expected_id}")
            used += state.operation_costs[str(operation)]
            scan_points_used += int(record.get("scan_point_count", 0) or 0)
            restored.append(dict(record))
        if used > state.main_budget:
            raise ValueError("existing jobs exceed the configured main budget")
        if state.max_scan_points is not None and scan_points_used > state.max_scan_points:
            raise ValueError("existing jobs exceed the configured scan-point budget")
        state.main_used = used
        state.scan_points_used = scan_points_used
        state._next_job = len(restored) + 1
        state._records = restored
        return state

    def budget_status(self) -> dict[str, int]:
        result = {
            "main_budget": self.main_budget,
            "main_used": self.main_used,
            "main_remaining": self.main_budget - self.main_used,
            "recovery_budget": self.recovery_budget,
            "recovery_used": self.recovery_used,
            "recovery_remaining": self.recovery_budget - self.recovery_used,
        }
        if self.max_scan_points is not None:
            result.update(
                {
                    "scan_points_budget": self.max_scan_points,
                    "scan_points_used": self.scan_points_used,
                    "scan_points_remaining": self.max_scan_points - self.scan_points_used,
                }
            )
        return result

    def _cost(self, operation: Operation) -> int:
        return self.operation_costs[operation]

    def _visible_path(self, root: Path, relative: str) -> Path:
        supplied = Path(relative)
        if supplied.is_absolute() or ".." in supplied.parts:
            raise ValueError("path must stay inside its visible directory")
        resolved_root = root.resolve()
        resolved = (resolved_root / supplied).resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError("path must stay inside its visible directory")
        return resolved

    def run_xtb(self, request: JobRequest) -> dict[str, object]:
        if request.operation not in self.allowed_operations:
            raise ValueError(f"operation {request.operation!r} is not allowed by this task")
        if self._ended:
            raise RuntimeError("run has ended")
        control_source: Path | None = None
        if request.control_path is not None:
            control_source = self._visible_path(
                self.run_dir / "model" / "workspace",
                request.control_path,
            )
            if not control_source.is_file():
                raise FileNotFoundError(control_source)
            if "scan" not in self.allowed_operations and any(
                line.lstrip().lower().startswith(("$scan", "scan:"))
                for line in control_source.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            ):
                raise ValueError("scan directives are not allowed by this task")

        source = self._visible_path(
            self.run_dir / "model" / "workspace",
            request.input_path,
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size == 0:
            raise ValueError("xTB input geometry is empty")
        if control_source is not None:
            if control_source.name == source.name:
                raise ValueError("control and geometry inputs must have different names")

        scan_point_count = 0
        if request.operation == "scan":
            if control_source is None:
                if self.max_scan_points is not None:
                    raise ValueError("scan operation requires control_path")
            else:
                scan_point_count = count_requested_scan_points(
                    control_source.read_text(encoding="utf-8", errors="replace")
                )
                if self.max_scan_points is not None and scan_point_count == 0:
                    raise ValueError("scan control contains no $scan schedule")
            if (
                self.max_scan_points is not None
                and self.scan_points_used + scan_point_count > self.max_scan_points
            ):
                raise RuntimeError("scan-point budget exceeded")

        if not sys.platform.startswith("linux"):
            raise RuntimeError("actual execution requires Linux or WSL")

        cost = self._cost(request.operation)
        if self.main_used + cost > self.main_budget:
            raise RuntimeError("main budget exceeded")

        self.main_used += cost
        self.scan_points_used += scan_point_count
        job_id = f"job_{self._next_job:03d}"
        self._next_job += 1
        job_dir = self.run_dir / "jobs" / job_id
        job_dir.mkdir()
        copied_input = job_dir / source.name
        shutil.copy2(source, copied_input)
        if control_source is not None:
            shutil.copy2(control_source, job_dir / control_source.name)
        command = build_xtb_command(self.xtb_executable, copied_input.name, request)

        stdout = ""
        stderr = ""
        return_code: int | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=job_dir,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            return_code = completed.returncode
            status = "completed" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            stdout = _text(exc.stdout)
            stderr = _text(exc.stderr)
            status = "timeout"
        except OSError as exc:
            stderr = str(exc)
            status = "failed"

        (job_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (job_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        record: dict[str, object] = {
            "job_id": job_id,
            "operation": request.operation,
            "input": request.input_path,
            "command": command,
            "status": status,
            "return_code": return_code,
            "output_directory": f"jobs/{job_id}",
            "result_directory": f"job_results/{job_id}",
        }
        if self.max_scan_points is not None:
            record["scan_point_count"] = scan_point_count
        self._records.append(record)
        (self.run_dir / "jobs.json").write_text(
            json.dumps(self._records, indent=2) + "\n",
            encoding="utf-8",
        )
        mirror = self.run_dir / "model" / "workspace" / "job_results" / job_id
        mirror.parent.mkdir(exist_ok=True)
        shutil.copytree(job_dir, mirror)
        return record

    def submit(
        self,
        required_paths: Sequence[str],
        *,
        allow_incomplete: bool = False,
    ) -> dict[str, object]:
        if self._ended:
            raise RuntimeError("run has ended")
        submission = self.run_dir / "model" / "submission"
        missing: list[str] = []
        for relative in required_paths:
            required = self._visible_path(submission, relative)
            if not required.exists():
                missing.append(relative)
        if missing and not allow_incomplete:
            raise FileNotFoundError(missing[0])

        accepted = self.run_dir / "accepted_submission"
        shutil.copytree(submission, accepted)
        digest = hashlib.sha256()
        for path in sorted(item for item in accepted.rglob("*") if item.is_file()):
            relative = path.relative_to(accepted).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        result: dict[str, object] = {
            "accepted_submission": "accepted_submission",
            "tree_sha256": digest.hexdigest(),
        }
        if missing:
            result["incomplete_required_outputs"] = True
            result["missing_required_outputs"] = missing
        (self.run_dir / "submission.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        self._ended = True
        return result
