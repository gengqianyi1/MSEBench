from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tomllib
from typing import Any

import yaml


@dataclass
class ValidationReport:
    root: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> str:
        data = asdict(self)
        data["ok"] = self.ok
        return json.dumps(data, ensure_ascii=False, indent=2)


def validate_repository(root: Path) -> ValidationReport:
    root = root.resolve()
    report = ValidationReport(str(root))
    expected = json.loads((root / "TASK_INVENTORY.json").read_text(encoding="utf-8"))
    expected_ids = {key.replace("task-", "") for key, value in expected["expected_tasks"].items() if value == "present"}
    task_dirs = sorted((root / "tasks").glob("task-*"))
    actual_ids = {path.name.split("-")[1] for path in task_dirs}
    if actual_ids != expected_ids:
        report.errors.append(f"task inventory mismatch: expected {sorted(expected_ids)}, got {sorted(actual_ids)}")
    for task_dir in task_dirs:
        task_id = task_dir.name
        row: dict[str, Any] = {}
        report.tasks[task_id] = row
        try:
            config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
            row["schema_version"] = config.get("schema_version")
            row["migration_status"] = config.get("metadata", {}).get("migration_status")
        except Exception as exc:
            report.errors.append(f"{task_id}: task.toml: {exc}")
            continue
        for rel in ("environment/Dockerfile", "environment/environment.lock.yml", "tests/test.sh"):
            if not (task_dir / rel).is_file():
                report.errors.append(f"{task_id}: missing {rel}")
        contract = task_dir / "harness_contract.yaml"
        if not contract.is_file():
            report.errors.append(f"{task_id}: missing harness_contract.yaml")
        else:
            try:
                spec = yaml.safe_load(contract.read_text(encoding="utf-8"))
                if spec.get("schema_version") != "chembench-harness-contract/1.0":
                    report.errors.append(f"{task_id}: invalid harness contract version")
                ownership = spec.get("ownership", {})
                for key in ("immutable", "author_editable", "author_required"):
                    if key not in ownership or not isinstance(ownership[key], list):
                        report.errors.append(f"{task_id}: ownership.{key} must be a list")
                confirmations = spec.get("author_confirmations", {})
                pending = sorted(key for key, value in confirmations.items() if value != "confirmed")
                gates = spec.get("release_gates", {})
                row["protocol_type"] = spec.get("protocol", {}).get("type")
                row["implementation_status"] = spec.get("protocol", {}).get("implementation_status")
                row["pending_author_confirmations"] = pending
                if pending:
                    report.warnings.append(f"{task_id}: author confirmation pending: {', '.join(pending)}")
                if gates.get("release_eligible") and (pending or not all(
                    gates.get(name, False) for name in ("schema", "build", "oracle", "protocol_decoupled", "parity", "leakage")
                )):
                    report.errors.append(f"{task_id}: release_eligible set before all release gates")
            except Exception as exc:
                report.errors.append(f"{task_id}: harness contract: {exc}")
        public_contract = task_dir / "environment/public/harness_contract.yaml"
        if not public_contract.is_file():
            report.errors.append(f"{task_id}: missing public harness contract")
        elif contract.is_file() and public_contract.read_bytes() != contract.read_bytes():
            report.errors.append(f"{task_id}: public harness contract is out of sync")
    for manifest in sorted((root / "datasets").glob("*/dataset.toml")):
        try:
            value = tomllib.loads(manifest.read_text(encoding="utf-8"))
            dataset = value.get("dataset")
            if not isinstance(dataset, dict) or "/" not in str(dataset.get("name", "")):
                report.errors.append(f"{manifest.relative_to(root)}: invalid [dataset].name")
            for index, task in enumerate(value.get("tasks", [])):
                digest = str(task.get("digest", ""))
                if not digest.startswith("sha256:") or len(digest) != 71:
                    report.errors.append(f"{manifest.relative_to(root)}: tasks[{index}] invalid digest")
        except Exception as exc:
            report.errors.append(f"{manifest.relative_to(root)}: {exc}")
    return report
