"""Disposable end-to-end smoke run for Task 01/02 with a scripted LLM.

The scripted model only emits a public-tool run_python call and a normal
submit_candidate call. The sidecar and Docker verifier remain real.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

def posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def wait_health(port: int) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"sidecar health timeout on {port}")


def task01_script() -> str:
    return """import gzip,json,shutil
from pathlib import Path
root=Path('.')
shutil.copy2(Path('input/original/examples/example_result.json'),root/'result.json')
(root/'final_answer.json').write_text(json.dumps({'final_status':'completed','rationale':'E2E protocol test submission.'}))
(root/'native_final_report.json').write_text(json.dumps({'status':'completed'}))
(root/'predictions.csv').write_text('sample_id,predicted_exfoliation_energy_mev_atom\\n')
(root/'candidate_library.json.gz').write_bytes(gzip.compress(b'[]'))
(root/'candidate_summary.json').write_text('{}')
(root/'model.joblib').write_bytes(b'mock-model')
(root/'feature_names.json').write_text('[]')
(root/'training_log.json').write_text('{}')
(root/'data').mkdir(exist_ok=True)
(root/'generator').mkdir(exist_ok=True)
(root/'generator/generate.py').write_text('')
(root/'generator/config.json').write_text('{}')
(root/'generator/generation_manifest.json').write_text('{}')
"""


def task02_script() -> str:
    return """import json
from pathlib import Path
root=Path('.')
for name,value in [('result.json',{}),('stage_A_audit.json',{}),('plan.json',{}),('stage_B_bulk.json',{}),('stage_C_local.json',{}),('revisions.json',[])]:
    (root/name).write_text(json.dumps(value))
(root/'final_answer.json').write_text(json.dumps({'schema_version':'chembench-task02-final-answer/1.0','task_id':'task_02_oligothiophene_smd_sapt','status':'completed','summary':'End-to-end protocol test submission with minimal public-contract artifacts.','result_file':'result.json'}))
(root/'native_final_report.json').write_text(json.dumps({'schema_version':'chembench-task02-native-report/1.0','task_id':'task_02_oligothiophene_smd_sapt','status':'completed','summary':'End-to-end protocol test native report with minimal public-contract artifacts.','artifacts':{'result':'result.json','stage_a':'stage_A_audit.json','plan':'plan.json','stage_b':'stage_B_bulk.json','stage_c':'stage_C_local.json','revisions':'revisions.json'},'scientific_scope':{'continuum_solvation_scope':'Protocol smoke test only.','orbital_gap_scope':'Protocol smoke test only.','conformational_sampling_scope':'Protocol smoke test only.','geometry_specific_sapt_scope':'Protocol smoke test only.'},'provenance':{'protocol_file':'protocol.yaml','instance_file':'instance.json'}}))
"""


async def run_task(key: str, task_dir: str, image: str, port: int, script: str) -> dict:
    task = ROOT / "tasks" / task_dir
    base = ROOT / "eval_runs" / f"e2e_{key}_20260901f"
    # A previous interrupted run can leave protocol_state.json with a session
    # token but without its public case.  Each E2E invocation is independent;
    # never reuse protocol state or evidence from an earlier attempt.
    shutil.rmtree(base, ignore_errors=True)
    workspace = base / "agent_workspace"
    evidence = base / "evidence"
    logs = base / "verifier_logs"
    run_dir = base / "verifier_run"
    for path in (workspace, evidence, logs, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task / "environment/public", workspace, dirs_exist_ok=True)
    container = f"chembench-{key}-e2e"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    subprocess.run([
        "docker", "run", "-d", "--rm", "--user", "0:0", "--name", container,
        "-p", f"127.0.0.1:{port}:{port}",
        "-v", f"{posix(ROOT / 'templates/sidecar/hidden_suite_server.py')}:/sidecar.py:ro",
        "-v", f"{posix(task / 'tests/private_suite_adapter.py')}:/tests/private_suite_adapter.py:ro",
        "-v", f"{posix(task / 'tests/task_snapshot')}:/task:ro",
        "-v", f"{posix(evidence)}:/organizer_evidence",
        image, "micromamba", "run", "-n", "chembench", "python", "/sidecar.py",
        "--adapter", "/tests/private_suite_adapter.py", "--snapshot", "/task",
        "--evidence-dir", "/organizer_evidence", "--host", "0.0.0.0", "--port", str(port),
    ], check=True, capture_output=True, text=True)
    try:
        os.environ[f"CHEMBENCH_{key.upper()}_SIDECAR_URL"] = f"http://127.0.0.1:{port}"
        wait_health(port)
        agent = object.__new__(MSEBenchAgent)
        phases = ["tool", "submit"]
        if key == "task01":
            phases = ["tool", "chgnet", "submit"]
        calls = iter(phases)

        def mocked_native(*_args, **_kwargs):
            phase = next(calls)
            if phase == "tool":
                args = {"script": script, "timeout_seconds": 30}
                name = "run_python"
            elif phase == "chgnet":
                args = {"candidate_ids": [f"CAND-{i:06d}" for i in range(5)]}
                name = "relax_selected_candidates_chgnet"
            else:
                args = {"final_status": "completed", "rationale": f"E2E {key} protocol test submission."}
                name = "submit_candidate"
            raw = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"mock-{name}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]}
            return name, args, "", raw

        agent._call_tool_action = mocked_native
        context = SimpleNamespace(metadata={})
        await agent.run(f"Solve protected {key} case 1.", FakeEnvironment(workspace), context)
        case_path = workspace / "cases/case-1/case_result.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        submission = case.get("submission") or {}
        row = {"task": key, "metadata": context.metadata, "submission": submission}
        evidence_workspace = evidence.joinpath(*submission["evidence_id"].split("/"), "workspace")
        row["evidence_workspace_exists"] = evidence_workspace.is_dir()
        verifier = [
            "docker", "run", "--rm", "-v", f"{posix(evidence_workspace)}:/workspace",
            "-v", f"{posix(task / 'tests/task_snapshot')}:/task:ro",
            # Use the checked-out wrapper even when the local verifier image is
            # cached; this E2E test is specifically a regression test for its
            # evidence-boundary wiring.
            "-v", f"{posix(task / 'tests/run_legacy_grader.py')}:/tests/run_legacy_grader.py:ro",
            "-v", f"{posix(logs)}:/logs", "-v", f"{posix(run_dir)}:/run", image,
            "micromamba", "run", "-n", "chembench", "python", "/tests/run_legacy_grader.py",
        ]
        result = subprocess.run(verifier, capture_output=True, text=True, timeout=3600)
        row["verifier_returncode"] = result.returncode
        row["verifier_stderr"] = result.stderr[-4000:]
        details = logs / "verifier/details.json"
        reward = logs / "verifier/reward.json"
        row["details"] = json.loads(details.read_text()) if details.exists() else None
        row["reward"] = json.loads(reward.read_text()) if reward.exists() else None
        return row
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


async def main() -> None:
    print(json.dumps(await run_task("task01", "task-01-jarvis-2d-exfoliation-candidate", "chembench/std-task01-verifier:2.1.0", 18720, task01_script()), ensure_ascii=False, indent=2, default=str))
    print(json.dumps(await run_task("task02", "task-02-oligothiophene-smd-sapt", "chembench/std-task02-verifier:2.1.1", 18721, task02_script()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
