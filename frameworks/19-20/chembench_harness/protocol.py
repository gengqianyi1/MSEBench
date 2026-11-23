from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any

from .budget import BudgetManager
from .state import ProtocolState, StateStore


class ProtocolError(RuntimeError):
    pass


def resolve_case_number(instruction: str, case_count: int) -> int:
    """Resolve an explicit suite case number without silently selecting a case.

    A one-case suite may omit the number.  A multi-case suite must state it in
    the Harbor step instruction so that an agent cannot accidentally submit
    evidence for case 1 while it is being evaluated on another case.
    """
    match = re.search(r"case\s+(\d+)", instruction, re.I)
    if match:
        number = int(match.group(1))
        if 1 <= number <= case_count:
            return number
        raise ProtocolError("hidden-suite case number is out of range")
    if case_count == 1:
        return 1
    raise ProtocolError("multi-case hidden-suite step lacks case number")


class TaskPlugin(ABC):
    """Only task-specific extension point. Core lifecycle methods are intentionally stable."""

    @abstractmethod
    def initialize(self) -> dict[str, Any]: ...

    @abstractmethod
    def validate_action(self, action: dict[str, Any], state: ProtocolState) -> None: ...

    @abstractmethod
    def execute_action(
        self, action: dict[str, Any], state: ProtocolState, workspace: Path
    ) -> dict[str, Any]: ...

    @abstractmethod
    def public_feedback(self, result: dict[str, Any], state: ProtocolState) -> dict[str, Any]: ...

    @abstractmethod
    def is_terminal(self, state: ProtocolState) -> bool: ...

    @abstractmethod
    def materialize_submission(self, state: ProtocolState, workspace: Path) -> None: ...


class ProtocolEngine:
    def __init__(
        self,
        *,
        task_id: str,
        protocol_type: str,
        workspace: Path | str,
        plugin: TaskPlugin,
        budget_limits: dict[str, int],
    ):
        self.workspace = Path(workspace).resolve()
        self.plugin = plugin
        self.store = StateStore(self.workspace)
        self.task_id = task_id
        self.protocol_type = protocol_type
        self.budget_limits = budget_limits

    def start(self) -> ProtocolState:
        if self.store.load() is not None:
            raise ProtocolError("protocol already initialized")
        budget = BudgetManager(self.budget_limits)
        state = ProtocolState(
            task_id=self.task_id,
            protocol_type=self.protocol_type,
            budgets=budget.as_dict(),
            task_state=self.plugin.initialize(),
        )
        self.store.save(state)
        return state

    def submit(self, action: dict[str, Any]) -> dict[str, Any]:
        state = self.store.load()
        if state is None:
            raise ProtocolError("protocol is not initialized")
        if state.status != "running":
            raise ProtocolError(f"protocol is {state.status}")
        self.plugin.validate_action(action, state)
        limits = {name: row["limit"] for name, row in state.budgets.items()}
        used = {name: row["used"] for name, row in state.budgets.items()}
        budget = BudgetManager(limits, used)
        budget.consume("actions")
        # Persist the reservation before task code runs. A crash/timeout must not grant a free retry.
        state.budgets = budget.as_dict()
        self.store.save(state)
        try:
            result = self.plugin.execute_action(action, state, self.workspace)
        except Exception as exc:
            state.history.append({
                "round": state.current_round + 1,
                "action": action,
                "status": "execution_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            self.store.save(state)
            raise
        state.current_round += 1
        feedback = self.plugin.public_feedback(result, state)
        state.history.append({"round": state.current_round, "action": action, "feedback": feedback})
        if self.plugin.is_terminal(state):
            state.status = "complete"
            self.plugin.materialize_submission(state, self.workspace)
        self.store.save(state)
        return feedback
