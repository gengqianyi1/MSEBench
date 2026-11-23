"""Author-side parity check for this new Harbor-native task."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_parity(output_root):
    adapter_module = _module("task20_parity_adapter", HERE / "private_suite_adapter.py")
    grader = _module("task20_parity_grader", HERE / "protocol_grader.py")
    adapter = adapter_module.create_adapter(HERE / "task_snapshot")
    reference = json.loads((HERE / "task_snapshot/private_data/reference.json").read_text())
    oracle = json.loads((HERE / "fixtures/oracle.json").read_text())
    requests = [
        ("tpo", {"sample": "spent"}),
        ("bet", {"sample": "spent"}),
        ("xps", {"region": "S2p"}),
        ("particle_size_test", {"mode": "tem"}),
    ]
    state = adapter.start_session(Path(output_root), "parity")
    events = []
    for tool, arguments in requests:
        response = adapter.call_tool(0, tool, arguments, state)
        events.append({"tool": tool, "arguments": arguments, "response": response})
    adapter.submit(0, oracle, events, state)
    sidecar = json.loads((Path(output_root) / "parity/case-1-evidence.json").read_text())
    direct = {"case_id": "PDH-DEACT-H01", "candidate": oracle, "tool_events": events}
    sidecar_result = grader.grade_evidence(sidecar, reference)
    direct_result = grader.grade_evidence(direct, reference)
    return {"equal": sidecar_result == direct_result, "sidecar_score": sidecar_result["score"], "direct_score": direct_result["score"]}
