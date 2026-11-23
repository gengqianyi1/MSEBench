#!/usr/bin/env python3
"""Audit the harness coverage against one canonical registry, and gate release.

Bridges the runtime ``protocol_mode`` vocabulary (``agents/chembench_agent.py``)
with the declarative ``harness_contract.yaml`` ``protocol.type`` taxonomy, then
checks three things per task:

1. every ``release_eligible=true`` task is actually implemented by the common agent;
2. each contract's ``protocol.type`` is consistent with its ``protocol_mode``;
3. each contract's ``implementation_status`` matches the agent's real coverage.

Exit code 0 means the coverage gate passes; 1 means gaps were found.

With ``--fix-contracts``, the script surgically rewrites only the
``protocol.implementation_status`` field (both contract copies) to the derived
value, leaving every other field untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_coverage as hc  # noqa: E402


def audit(root: Path, fix_contracts: bool) -> dict:
    supported = hc.supported_protocol_modes()
    records = []
    gaps: list[str] = []
    for task_dir in sorted((root / "tasks").glob("task-*")):
        if not (task_dir / "task.toml").is_file():
            continue
        mode = hc.task_protocol_mode(task_dir)
        release = hc.task_release_eligible(task_dir)
        implemented = hc.agent_implements(mode, supported)
        contract = hc.load_contract(task_dir)
        ctype = hc.contract_type(contract)
        declared = hc.contract_implementation_status(contract)
        derived = hc.derived_implementation_status(mode, supported)

        allowed = hc.CONTRACT_TYPE_TO_MODES.get(ctype, set())
        type_ok = (ctype == "") or (mode in allowed)
        status_ok = (declared == "") or (declared == derived)

        issues: list[str] = []
        if release and not implemented:
            issues.append(f"release_eligible=true but protocol_mode '{mode}' is not implemented by the agent")
        if not type_ok:
            issues.append(f"contract protocol.type '{ctype}' is inconsistent with protocol_mode '{mode}'")
        if not status_ok:
            issues.append(f"implementation_status '{declared}' != derived '{derived}'")

        records.append({
            "task": task_dir.name,
            "protocol_mode": mode,
            "contract_type": ctype,
            "agent_implemented": implemented,
            "release_eligible": release,
            "contract_implementation_status": declared,
            "derived_implementation_status": derived,
            "issues": issues,
        })

        if fix_contracts and not status_ok and contract:
            for path in hc.contract_paths(task_dir):
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8")
                # replace the single implementation_status line, preserving indentation
                new_text, n = re.subn(
                    r"^(\s*implementation_status:\s*)\S+(\s*)$",
                    rf"\g<1>{derived}\g<2>",
                    text,
                    count=1,
                    flags=re.M,
                )
                if n:
                    path.write_text(new_text, encoding="utf-8", newline="\n")

        for issue in issues:
            gaps.append(f"{task_dir.name}: {issue}")

    report = {
        "schema_version": "chembench-harness-coverage/1.0",
        "agent": str(hc.AGENT.relative_to(root)),
        "supported_protocol_modes": sorted(supported),
        "contract_type_to_modes": {k: sorted(v) for k, v in hc.CONTRACT_TYPE_TO_MODES.items()},
        "tasks": records,
        "gate_pass": not gaps,
        "gaps": gaps,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="benchmark root")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    parser.add_argument("--fix-contracts", action="store_true", help="rewrite implementation_status to derived value")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = audit(root, args.fix_contracts)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
