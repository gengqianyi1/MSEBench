from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass
class ProtocolState:
    task_id: str
    protocol_type: str
    current_round: int = 0
    status: str = "running"
    budgets: dict[str, dict[str, int]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    task_state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "chembench-protocol-state/1.0"

    def validate(self) -> None:
        if not self.task_id or not self.protocol_type:
            raise ValueError("task_id and protocol_type are required")
        if self.current_round < 0:
            raise ValueError("current_round must be non-negative")
        if self.status not in {"running", "complete", "failed"}:
            raise ValueError(f"invalid protocol status: {self.status}")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProtocolState":
        state = cls(**value)
        state.validate()
        return state


class StateStore:
    """Atomic JSON state persistence restricted to a declared workspace root."""

    def __init__(self, workspace: Path | str, relative_path: str = "protocol_state.json"):
        self.workspace = Path(workspace).resolve()
        candidate = (self.workspace / relative_path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError("state path escapes workspace")
        self.path = candidate

    def load(self) -> ProtocolState | None:
        if not self.path.exists():
            return None
        return ProtocolState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, state: ProtocolState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.as_dict(), ensure_ascii=False, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
