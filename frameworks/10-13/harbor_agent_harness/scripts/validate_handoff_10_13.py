#!/usr/bin/env python3
"""Static author-handoff checks for Harbor Tasks 10--13."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import yaml


TASKS = {
    "task-10-matbench-bandgap-pipeline": ("predictions.csv", "pipeline.json", "final_answer.json"),
    "task-11-lithium-electrolyte-design": ("final_answer.json",),
    "task-12-pybamm-fast-charge": ("final_answer.json",),
    "task-13-retrosynthesis-route": ("final_answer.json",),
}
FORBIDDEN_PUBLIC_PARTS = {
    "private_data",
    "private_grader",
    "organizer_evidence",
    "organizer_private",
    "solution",
    "tests",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    rows: dict[str, dict] = {}

    for task_name, required in TASKS.items():
        task = root / "tasks" / task_name
        contract = yaml.safe_load((task / "harness_contract.yaml").read_text(encoding="utf-8"))
        public_contract = yaml.safe_load(
            (task / "environment" / "public" / "harness_contract.yaml").read_text(encoding="utf-8")
        )
        agent = json.loads(
            (task / "environment" / "public" / ".chembench_agent.json").read_text(encoding="utf-8")
        )
        snapshot = yaml.safe_load(
            (task / "tests" / "task_snapshot" / "task.yaml").read_text(encoding="utf-8")
        )
        harbor = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        instruction = (task / "instruction.md").read_text(encoding="utf-8")
        environment_dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
        verifier_dockerfile = (task / "tests" / "Dockerfile").read_text(encoding="utf-8")

        if contract != public_contract:
            errors.append(f"{task_name}: public contract copy differs")
        confirmations = contract.get("author_confirmations", {})
        if not confirmations or set(confirmations.values()) != {"confirmed"}:
            errors.append(f"{task_name}: author confirmations are incomplete")
        if contract["release_gates"]["release_eligible"] is not harbor.get("metadata", {}).get("release_eligible"):
            errors.append(f"{task_name}: task.toml and the contract disagree on release eligibility")

        budgets = contract["budgets"]
        comparisons = {
            "model_iterations": (agent["max_iterations"], snapshot["max_iterations"]),
            "actions": (agent["max_tool_calls"], snapshot["max_tool_calls"]),
            "timeout_per_action_sec": (agent["timeout_per_tool_sec"], snapshot["timeout_per_iteration"]),
            "total_timeout_sec": (harbor["agent"]["timeout_sec"], snapshot["total_timeout_seconds"]),
        }
        for budget_name, values in comparisons.items():
            if any(float(value) != float(budgets[budget_name]) for value in values):
                errors.append(f"{task_name}: inconsistent {budget_name}: {values} vs {budgets[budget_name]}")

        missing_names = [name for name in required if name not in instruction]
        if missing_names:
            errors.append(f"{task_name}: instruction omits required artifacts {missing_names}")
        if sorted(snapshot["required_artifacts"]) != sorted(required):
            errors.append(f"{task_name}: snapshot required_artifacts mismatch")
        if float(snapshot["pass_score"]) <= 0:
            errors.append(f"{task_name}: pass_score must be positive")

        public_root = task / "environment" / "public"
        leaked_paths = [
            path.relative_to(public_root).as_posix()
            for path in public_root.rglob("*")
            if path.is_file() and FORBIDDEN_PUBLIC_PARTS.intersection(path.relative_to(public_root).parts)
        ]
        if leaked_paths:
            errors.append(f"{task_name}: private path leaked into public tree: {leaked_paths}")
        if "COPY --chown=$MAMBA_USER:$MAMBA_USER public/ /workspace/" not in environment_dockerfile:
            errors.append(f"{task_name}: participant image does not copy only public/ to /workspace")
        if "task_snapshot/ /task/" not in verifier_dockerfile or "protocol_grader.py" not in verifier_dockerfile:
            errors.append(f"{task_name}: verifier image wiring is incomplete")
        if list(task.rglob("native_runner.py")):
            errors.append(f"{task_name}: native_runner.py is present")
        if not (task / "tests" / "private_suite_adapter.py").is_file():
            errors.append(f"{task_name}: private_suite_adapter.py is missing")
        if not (task / "tests" / "protocol_grader.py").is_file():
            errors.append(f"{task_name}: protocol_grader.py is missing")

        rows[task_name] = {
            "contract_copy_identical": contract == public_contract,
            "protocol_type": contract["protocol"]["type"],
            "budgets": budgets,
            "automatic_retries": budgets["automatic_retries"],
            "required_artifacts": list(required),
            "pass_score": snapshot["pass_score"],
            "public_private_path_leaks": leaked_paths,
            "release_eligible": contract["release_gates"]["release_eligible"],
        }

    report = {
        "schema_version": "chembench-author-handoff-validation/1.0",
        "tasks": rows,
        "error_count": len(errors),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
