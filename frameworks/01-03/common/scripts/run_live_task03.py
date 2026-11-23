#!/usr/bin/env python3
"""Run the Task 03 five-round design protocol with a live OpenAI-compatible model."""
from __future__ import annotations
import asyncio, json, os, shutil, subprocess, tempfile, traceback
from pathlib import Path
from types import SimpleNamespace

from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parent / "tasks/task03/tasks/task-03-five-ring-xtb-dipro-marcus"


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _rounds_written(workspace: Path) -> int:
    return sum(1 for number in range(1, 6) if (workspace / f"rounds/round-{number}/candidate.json").is_file())


def _checkpoint(workspace: Path, *, status: str, current_round: int | None, phase: str, **extra: object) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "current_round": current_round,
        "phase": phase,
        "rounds_written": _rounds_written(workspace),
        **extra,
    }
    _atomic_json(workspace / "task03_runner_state.json", payload)


def _failure_result(workspace: Path, metadata: dict, *, failure: str, current_round: int | None, phase: str, error: BaseException | None = None) -> dict:
    result = {
        "metadata": metadata,
        "failure": failure,
        "round": current_round,
        "phase": phase,
        "rounds_written": _rounds_written(workspace),
        "workspace": str(workspace),
    }
    if error is not None:
        result.update({
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
    _atomic_json(workspace / "run_result.json", result)
    _checkpoint(workspace, status="failed", current_round=current_round, phase=phase, failure=failure, error_type=result.get("error_type"), error=result.get("error"))
    return result


async def main() -> int:
    output = ROOT / os.environ.get("CHEMBENCH_WORKSPACE", "parity/task03_model_workspace_v2")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copytree(TASK / "environment/public", output)
    environment = FakeEnvironment(output)
    agent = object.__new__(MSEBenchAgent)
    metadata = {}
    for round_number in range(1, 6):
        phase = "agent"
        candidate_written = False
        try:
            _checkpoint(output, status="running", current_round=round_number, phase=phase)
            context = SimpleNamespace(metadata={})
            await agent.run(f"Submit candidate R{round_number} for round {round_number}.", environment, context)
            metadata[f"R{round_number}"] = getattr(context, "metadata", {})
            phase = "candidate_write"
            candidate_written = (output / f"rounds/round-{round_number}/candidate.json").is_file()
            if not candidate_written:
                result = _failure_result(output, metadata, failure="candidate_missing", current_round=round_number, phase=phase)
                print(json.dumps({"status": "FAIL", **result}, ensure_ascii=False))
                return 1
        except TimeoutError as exc:
            failure = "agent_timeout"
            result = _failure_result(output, metadata, failure=failure, current_round=round_number, phase=phase, error=exc)
            print(json.dumps({"status": "FAIL", **result}, ensure_ascii=False))
            return 1
        except BaseException as exc:
            failure = "agent_exception"
            result = _failure_result(output, metadata, failure=failure, current_round=round_number, phase=phase, error=exc)
            print(json.dumps({"status": "FAIL", **result}, ensure_ascii=False))
            return 1
        finally:
            # This is intentionally per-round: a killed/failed run leaves the last
            # completed phase and candidate state rather than a silent gap after Rn.
            if not (output / "run_result.json").is_file():
                _checkpoint(
                    output,
                    status="running",
                    current_round=round_number,
                    phase=f"{phase}:finally",
                    candidate_written=candidate_written,
                )
    context = SimpleNamespace(metadata={})
    _checkpoint(output, status="running", current_round=None, phase="final_recommendation")
    try:
        await agent.run("Return final recommended_candidate_id from R1-R5.", environment, context)
    except BaseException as exc:
        result = _failure_result(output, metadata, failure="final_recommendation_exception", current_round=None, phase="final_recommendation", error=exc)
        print(json.dumps({"status": "FAIL", **result}, ensure_ascii=False))
        return 1
    metadata["final"] = getattr(context, "metadata", {})
    final_answer_path = output / "final_answer.json"
    if not final_answer_path.is_file():
        result = _failure_result(output, metadata, failure="final_answer_missing", current_round=None, phase="final_recommendation")
        print(json.dumps({"status": "FAIL", **result}, ensure_ascii=False))
        return 1
    final = json.loads(final_answer_path.read_text(encoding="utf-8"))
    result = {"metadata": metadata, "final_answer": final, "workspace": str(output)}
    _atomic_json(output / "run_result.json", result)
    _checkpoint(output, status="completed", current_round=None, phase="complete", exit_code=0)
    print(json.dumps({
        "status": "PASS" if final.get("recommended_candidate_id") in {"R1","R2","R3","R4","R5"} else "FAIL",
        "recommended_candidate_id": final.get("recommended_candidate_id"),
        "rounds_written": sum(1 for n in range(1,6) if (output / f"rounds/round-{n}/candidate.json").exists()),
        "final_written": (output / "final_answer.json").exists(),
        "workspace": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    workspace = ROOT / os.environ.get("CHEMBENCH_WORKSPACE", "parity/task03_model_workspace_v2")
    try:
        exit_code = asyncio.run(main())
    except BaseException as exc:
        if workspace.is_dir():
            _failure_result(workspace, {}, failure="runner_unhandled_exception", current_round=None, phase="runner", error=exc)
        raise
    if workspace.is_dir():
        _checkpoint(workspace, status="completed" if exit_code == 0 else "failed", current_round=None, phase="process_exit", exit_code=exit_code)
    raise SystemExit(exit_code)
