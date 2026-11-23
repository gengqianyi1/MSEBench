#!/usr/bin/env python3
"""Create deterministic per-task harness contracts from the reviewed migration matrix."""
from __future__ import annotations

from pathlib import Path
import sys
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import harness_coverage as hc  # noqa: E402


PROTOCOLS = {
    "01": ("controlled_tool", {"actions": 30, "reserved_repair_actions": 2, "agent_turns": 34}, [
        "Confirm run_python sandbox semantics and one-shot CHGNet tool accounting.",
        "Provide an Oracle that exercises both controlled tools.",
    ]),
    "02": ("staged_campaign", {"actions": 60, "agent_turns": 60}, [
        "Specify the exact A/B/C/D stage transition and timeout policy.",
        "Define transferable verifier evidence for the private replay grader.",
    ]),
    "03": ("multi_round", {"actions": 5}, [
        "Specify all five observations and candidate action schemas.",
        "Replace trusted native-result score with independent scientific recomputation.",
    ]),
    "07": ("controlled_suite", {
        "actions": 12,
        "agent_turns": 12,
        "unique_evaluated_candidates": 6,
        "provider_retries": 2,
        "provider_request_timeout_seconds": 180,
        "total_timeout_seconds": 900,
        "minimum_successful_actions": 3,
        "minimum_unique_evaluated_candidates": 2,
        "minimum_feedback_rounds": 2,
    }, [
        "Specify hidden-suite case sequencing and per-case interaction.",
        "Define a protected tool/service boundary for hidden cases.",
    ]),
    "08": ("controlled_suite", {
        "actions": 24,
        "agent_turns": 14,
        "execution_attempts": 3,
        "provider_retries": 5,
        "total_timeout_seconds": 900,
        "single_xtb_timeout_seconds": 240,
        "minimum_successful_actions": 6,
        "text_read_lines_per_action": 160,
        "text_read_bytes_per_action": 65536,
    }, [
        "Specify structured xTB diagnostic actions and execution-attempt state.",
        "Confirm hidden-case feedback redaction.",
    ]),
    "09": ("adaptive_campaign", {
        "actions": 12,
        "scan_batches": 4,
        "unique_scan_angles": 12,
        "minimum_valid_scan_angles": 10,
        "minimum_feedback_rounds": 2,
        "attempts_per_angle": 2,
        "single_xtb_timeout_seconds": 180,
        "total_native_timeout_seconds": 900,
        "output_tokens_per_turn": 384,
    }, [
        "Specify adaptive angle selection, state transitions, and xTB call accounting.",
        "Provide deterministic campaign replay fixtures.",
    ]),
    "10": ("single_step", {"actions": 20}, ["Confirm generic-agent parity tolerance and one valid Oracle."]),
    "11": ("single_step", {"actions": 12}, ["Confirm generic-agent parity tolerance and one valid Oracle."]),
    "12": ("single_step", {"actions": 12}, ["Confirm generic-agent parity tolerance and one valid Oracle."]),
    "13": ("single_step", {"actions": 15}, ["Confirm generic-agent parity tolerance and one valid Oracle."]),
    "14": ("multi_round", {"actions": 5}, [
        "Confirm five-round public feedback fields and edit constraints.",
        "Approve scientific validity of the non-optimal synthetic lifecycle Oracle.",
    ]),
    "15": ("multi_round", {"actions": 5}, [
        "Confirm five-round candidate/feedback semantics and edit-budget accounting.",
        "Provide a scientifically valid Oracle.",
    ]),
    "16": ("multi_round", {"actions": 5}, [
        "Confirm private response-model outputs that may appear in public feedback.",
        "Provide a scientifically valid Oracle.",
    ]),
    "17": ("multi_round", {"actions": 5}, [
        "Confirm private agglomeration/response-model public feedback boundary.",
        "Provide a scientifically valid Oracle.",
    ]),
}

# Protocol rounds are state-machine turns, not task.toml suite steps.  Task 09
# uses four scan-batch turns plus one terminal submit turn.
PROTOCOL_MAX_ROUNDS = {"07": 12, "08": 14, "09": 5}

IMMUTABLE = [
    "schema_version",
    "protocol.state_file",
    "ownership",
    "release_gates",
    "msebench_harness core",
]
EDITABLE = [
    "protocol.type",
    "protocol.max_rounds",
    "budgets",
    "task-local plugin",
    "public feedback fields",
    "private grader scientific implementation",
]
COMMON_REQUIRED = [
    "Declare the exact original action/observation sequence.",
    "Declare tool budgets, retry accounting, timeouts, and failure behavior.",
    "Identify every public and private field/file.",
    "Provide one valid Oracle and at least three invalid submissions.",
    "Define same-model legacy-versus-Harbor parity tolerances.",
]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    selected = set(sys.argv[2:])
    for task_dir in sorted((root / "tasks").glob("task-*")):
        if selected and task_dir.name not in selected:
            continue
        number = task_dir.name.split("-")[1]
        protocol_type, budgets, required = PROTOCOLS[number]
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        metadata = config.get("metadata", {})
        already_decoupled = bool(metadata.get("protocol_decoupled", False))
        # implementation_status must reflect whether the common agent actually
        # implements the task's runtime protocol_mode, not author metadata.
        supported = hc.supported_protocol_modes()
        mode = hc.task_protocol_mode(task_dir)
        status = hc.derived_implementation_status(mode, supported)
        contract = {
            "schema_version": "chembench-harness-contract/1.0",
            "task_id": task_dir.name,
            "protocol": {
                "type": protocol_type,
                "implementation_status": status,
                "state_file": "/workspace/protocol_state.json",
                "max_rounds": PROTOCOL_MAX_ROUNDS.get(
                    number, 5 if protocol_type == "multi_round" else 1
                ),
            },
            "budgets": dict(budgets),
            "ownership": {
                "immutable": IMMUTABLE,
                "author_editable": EDITABLE,
                "author_required": COMMON_REQUIRED + required,
            },
            "author_confirmations": {
                "protocol_equivalent": "confirmed" if metadata.get("protocol_validated") else "pending",
                "scoring_equivalent": "confirmed" if metadata.get("scoring_validated") else "pending",
                "private_boundary_correct": "confirmed" if metadata.get("private_boundary_validated") else "pending",
                "oracle_scientifically_valid": "confirmed" if metadata.get("oracle_scientifically_validated") else "pending",
            },
            "release_gates": {
                "schema": bool(metadata.get("schema_validated", False)),
                "build": bool(metadata.get("build_validated", False)),
                "oracle": metadata.get("oracle_status") in {"verified_offline", "verified_harbor"},
                "protocol_decoupled": already_decoupled,
                "parity": bool(metadata.get("parity_validated", False)),
                "leakage": bool(metadata.get("leakage_validated", False)),
                "release_eligible": bool(metadata.get("release_eligible", False)),
            },
        }
        if number == "15":
            contract["budgets"].update({"tool_calls_per_round": 6, "tool_messages_per_round": 2})
        if number == "16":
            contract["budgets"].update({"measurements": 5, "tool_calls_per_round": 4, "tool_messages_per_round": 2})
        if number == "17":
            contract["budgets"].update({"measurements": 5, "tool_calls_per_round": 8, "tool_messages_per_round": 2})
        output = yaml.safe_dump(contract, sort_keys=False, allow_unicode=True)
        (task_dir / "harness_contract.yaml").write_text(output, encoding="utf-8", newline="\n")
        public_contract = task_dir / "environment" / "public" / "harness_contract.yaml"
        public_contract.parent.mkdir(parents=True, exist_ok=True)
        public_contract.write_text(output, encoding="utf-8", newline="\n")
        print(task_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
