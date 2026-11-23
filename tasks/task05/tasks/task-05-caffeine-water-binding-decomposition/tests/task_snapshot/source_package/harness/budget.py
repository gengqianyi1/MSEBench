"""Append-only cumulative budget accounting for Agent and recovery activity."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .schema_validation import validate_budget_ledger


COUNTER_FIELDS = (
    "agent_turns",
    "api_attempts",
    "tool_calls",
    "xtb_jobs",
    "xtb_scan_points",
    "wall_seconds",
    "requested_tokens",
)

# Frozen spec 5.3 ledger counter names (canonical).
SPEC53_FIELDS = (
    "agent_turns_used",
    "api_requests_used",
    "empty_response_retries",
    "recovery_attempts",
    "tool_calls_used",
    "xtb_jobs_reserved",
    "xtb_jobs_completed",
    "wall_time_seconds",
)

# historical internal counter -> frozen spec 5.3 counter.
_HISTORICAL_TO_SPEC53 = {
    "agent_turns": "agent_turns_used",
    "api_attempts": "api_requests_used",
    "tool_calls": "tool_calls_used",
    "xtb_jobs": "xtb_jobs_reserved",
    "wall_seconds": "wall_time_seconds",
}

FINALIZATION_NOTICE = (
    "Active work budget exhausted. The Harness will state the exact trigger and remaining "
    "submission-only finalization turns. No new xTB jobs are available; immediately materialize "
    "an honest complete or partial submission from existing evidence."
)
FINALIZATION_TOOLS = frozenset({"read_file", "run_python", "write_file", "xtb_parse"})

TASK04_COMPUTE_CLASSES = frozenset(
    {"other", "single_point", "optimization", "hessian", "optimization_hessian", "scan"}
)


def classify_xtb_compute(arguments: Sequence[str]) -> str:
    """Classify one validated xTB argument vector for weighted accounting."""

    normalized = {str(value).strip().lower() for value in arguments}
    if "--scan" in normalized:
        return "scan"
    has_optimization = "--opt" in normalized or "--ohess" in normalized
    has_hessian = "--hess" in normalized or "--ohess" in normalized
    if has_optimization and has_hessian:
        return "optimization_hessian"
    if has_hessian:
        return "hessian"
    if has_optimization:
        return "optimization"
    if normalized & {"--help", "-h", "--version"}:
        return "other"
    return "single_point"


def task04_compute_cost(compute_class: str, compute_budget: Mapping[str, object]) -> int:
    """Return the frozen Task 04 credit cost for a normalized class."""

    if compute_class not in TASK04_COMPUTE_CLASSES:
        raise ValueError(f"unknown xTB compute class: {compute_class}")
    if compute_class == "scan":
        return int((compute_budget.get("costs") or {}).get("other", 1))  # type: ignore[union-attr]
    costs = compute_budget.get("costs")
    if not isinstance(costs, Mapping) or compute_class not in costs:
        raise ValueError(f"compute budget has no cost for class: {compute_class}")
    value = costs[compute_class]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid compute cost for class: {compute_class}")
    return int(value)


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str | None = None
    enters_finalization: bool = False


class BudgetExceededError(RuntimeError):
    """Raised before an operation would exceed a frozen logical limit."""


class BudgetLedger:
    """Mutable runtime view whose serialized representation is append-only."""

    def __init__(
        self,
        limits: Mapping[str, int | float],
        *,
        compute_budget: Mapping[str, object] | None = None,
        schema_dir: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = dict(limits)
        self.limits.setdefault("finalization_turns", 2)
        self.limits.setdefault("finalization_wall_seconds", 600)
        self.schema_dir = Path(schema_dir)
        self.clock = clock
        self.started = float(clock())
        self.events: list[dict] = []
        self.totals = {field: 0 for field in COUNTER_FIELDS}
        self.totals["wall_seconds"] = 0.0
        self._begun_turns: set[int] = set()
        self.phase = "active"
        self.exhaustion_trigger: str | None = None
        self.rejected_requests = 0
        self.finalization_turns_used = 0
        self.finalization_tool_calls = 0
        self.phase_history = [{"phase": "active", "trigger": "run_started"}]
        self._reserved_job_ids: set[str] = set()
        # Spec 5.3 counters without a historical counterpart.
        self._spec53_extra = {
            "empty_response_retries": 0,
            "recovery_attempts": 0,
            "xtb_jobs_completed": 0,
        }
        self.compute_budget = deepcopy(dict(compute_budget)) if compute_budget is not None else None
        self.compute_events: list[dict] = []
        self.compute_totals = {
            "main_credits_used": 0,
            "recovery_credits_used": 0,
            "total_credits_used": 0,
            "main_credits_remaining": int((self.compute_budget or {}).get("main_credits", 0)),
            "recovery_credits_remaining": int((self.compute_budget or {}).get("recovery_credits", 0)),
            "admitted_jobs": 0,
        }
        self._compute_jobs: dict[str, dict] = {}
        self._recovery_chain_credits: dict[str, int] = {}

    def _append_compute_event(
        self,
        event_type: str,
        *,
        job_id: str,
        compute_class: str,
        credits: int,
        pool: str | None,
        candidate_id: str | None,
        condition: str | None,
        parent_job_id: str | None,
        recovery_of_job_id: str | None,
        reason: str | None = None,
    ) -> None:
        self.compute_events.append(
            {
                "sequence": len(self.compute_events) + 1,
                "event_type": event_type,
                "job_id": str(job_id),
                "compute_class": str(compute_class),
                "credits": int(credits),
                "pool": pool,
                "candidate_id": candidate_id,
                "condition": condition,
                "parent_job_id": parent_job_id,
                "recovery_of_job_id": recovery_of_job_id,
                "reason": reason,
                "cumulative": deepcopy(self.compute_totals),
            }
        )

    def _elapsed(self) -> float:
        return max(float(self.totals["wall_seconds"]), max(0.0, float(self.clock()) - self.started))

    def _assert_limits(self, counters: tuple[str, ...]) -> None:
        exceeded: list[str] = []
        for counter in counters:
            if self.totals[counter] > self.limits[counter]:
                exceeded.append(counter)
        if self._elapsed() > float(self.limits["wall_seconds"]):
            exceeded.append("wall_seconds")
        if exceeded:
            raise BudgetExceededError(
                "; ".join(f"budget limit exceeded: {counter}" for counter in exceeded)
            )

    def _append(
        self,
        event_type: str,
        logical_turn: int,
        *,
        reason: str | None = None,
        enforce_limits: bool = True,
        limit_fields: tuple[str, ...] = (),
    ) -> None:
        self.totals["wall_seconds"] = self._elapsed()
        if enforce_limits:
            self._assert_limits(limit_fields)
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "logical_turn": int(logical_turn),
                "api_attempt": int(self.totals["api_attempts"]),
                "tool_calls": int(self.totals["tool_calls"]),
                "xtb_jobs": int(self.totals["xtb_jobs"]),
                "xtb_scan_points": int(self.totals["xtb_scan_points"]),
                "elapsed_wall_seconds": float(self.totals["wall_seconds"]),
                "retry_or_resume_reason": reason,
                "cumulative": deepcopy(self.totals),
            }
        )

    def begin_turn(self, logical_turn: int) -> None:
        logical_turn = int(logical_turn)
        if logical_turn in self._begun_turns:
            return
        if self.phase != "active":
            raise BudgetExceededError("scientific phase has ended")
        if int(self.totals["agent_turns"]) >= int(self.limits["agent_turns"]):
            decision = self._deny("agent_turns", logical_turn=logical_turn)
            raise BudgetExceededError(decision.reason or "agent turn budget exceeded")
        self._begun_turns.add(logical_turn)
        self.totals["agent_turns"] += 1
        self._append("agent_turn", logical_turn, enforce_limits=False)

    def enter_finalization(self, trigger: str) -> None:
        """Enter the non-compute submission phase exactly once."""
        if self.phase == "finalizing":
            return
        self.phase = "finalizing"
        self.exhaustion_trigger = str(trigger)
        self.phase_history.append({"phase": "finalizing", "trigger": str(trigger)})
        logical_turn = int(self.totals["agent_turns"])
        self._append("finalization_started", logical_turn, reason=str(trigger), enforce_limits=False)

    def _deny(self, reason: str, *, logical_turn: int | None = None) -> BudgetDecision:
        entered = self.phase != "finalizing"
        self.rejected_requests += 1
        self.enter_finalization(reason)
        self._append(
            "budget_rejected",
            int(self.totals["agent_turns"] if logical_turn is None else logical_turn),
            reason=f"budget limit exceeded: {reason}",
            enforce_limits=False,
        )
        return BudgetDecision(False, f"budget limit exceeded: {reason}", entered)

    def record_agent_turn(self, phase: str) -> BudgetDecision:
        """Reserve one active or finalization model turn without overshooting."""
        if phase == "active":
            if self.phase != "active":
                return BudgetDecision(False, "scientific phase has ended", False)
            if int(self.totals["agent_turns"]) >= int(self.limits["agent_turns"]):
                return self._deny("agent_turns")
            self.totals["agent_turns"] += 1
            logical_turn = int(self.totals["agent_turns"])
            self._begun_turns.add(logical_turn)
            self._append("agent_turn", logical_turn, enforce_limits=False)
            return BudgetDecision(True)
        if phase != "finalization":
            raise ValueError("phase must be active or finalization")
        if self.phase != "finalizing":
            raise ValueError("finalization turn requested before finalization")
        limit = int(self.limits.get("finalization_turns", 0))
        if self.finalization_turns_used >= limit:
            self.rejected_requests += 1
            self._append(
                "budget_rejected",
                int(self.totals["agent_turns"]),
                reason="budget limit exceeded: finalization_turns",
                enforce_limits=False,
            )
            return BudgetDecision(False, "budget limit exceeded: finalization_turns", False)
        self.finalization_turns_used += 1
        # RobustModelClient retains the legacy ``begin_turn`` hook.  Mark the
        # phase-aware reservation with the same next logical-turn identity so
        # that hook observes an existing reservation instead of rejecting the
        # finalization request as a new scientific turn.
        logical_turn = max(self._begun_turns, default=0) + 1
        self._begun_turns.add(logical_turn)
        # The schema's logical_turn counter tracks scientific turns only;
        # finalization has its own counter and therefore records the last
        # scientific turn while retaining the private reservation identity.
        self._append("finalization_turn", int(self.totals["agent_turns"]), enforce_limits=False)
        return BudgetDecision(True)

    def record_api_attempt(
        self,
        logical_turn: int,
        requested_tokens: int,
        attempt_class: str,
        reason: str | None = None,
    ) -> None:
        self.totals["api_attempts"] += 1
        self.totals["requested_tokens"] += int(requested_tokens)
        # The adapter uses a private next-turn identity for finalization so
        # RobustModelClient can recognize the phase-aware reservation.  The
        # serialized ledger, however, intentionally counts scientific turns
        # only; keep finalization API events attached to the last cumulative
        # scientific turn so the formal invariant remains true.
        ledger_logical_turn = (
            int(self.totals["agent_turns"])
            if self.phase == "finalizing"
            else int(logical_turn)
        )
        event_type = {
            "initial": "api_attempt",
            "resumed_initial": "api_attempt",
            "transport_retry": "transport_retry",
            "resumed_transport_retry": "transport_retry",
            "semantic_retry": "semantic_retry",
            "resumed_semantic_retry": "semantic_retry",
            "key_fallback": "key_fallback",
            "resumed_key_fallback": "key_fallback",
        }.get(attempt_class, "api_attempt")
        try:
            self._append(event_type, ledger_logical_turn, reason=reason, limit_fields=("api_attempts",))
        except BudgetExceededError as exc:
            self._append("budget_rejected", ledger_logical_turn, reason=str(exc), enforce_limits=False)
            raise

    def record_tool_call(self, tool_name: str | int, phase: str | None = None) -> BudgetDecision | None:
        """Reserve a tool call; legacy integer calls retain exception semantics."""
        if isinstance(tool_name, int):
            logical_turn = int(tool_name)
            if self.phase != "active" or int(self.totals["tool_calls"]) >= int(self.limits["tool_calls"]):
                decision = self._deny("tool_calls", logical_turn=logical_turn)
                raise BudgetExceededError(decision.reason or "tool budget exceeded")
            self.totals["tool_calls"] += 1
            self._append("tool_call", logical_turn, enforce_limits=False)
            return None
        selected_phase = self.phase if phase is None else phase
        if selected_phase == "finalization":
            if self.phase != "finalizing":
                raise ValueError("finalization tool requested before finalization")
            if tool_name not in FINALIZATION_TOOLS:
                self.rejected_requests += 1
                self._append(
                    "tool_rejected_in_finalization",
                    int(self.totals["agent_turns"]),
                    reason=f"tool unavailable during finalization: {tool_name}",
                    enforce_limits=False,
                )
                return BudgetDecision(False, f"tool unavailable during finalization: {tool_name}", False)
            self.finalization_tool_calls += 1
            self._append(
                "finalization_tool_call",
                int(self.totals["agent_turns"]),
                reason=tool_name,
                enforce_limits=False,
            )
            return BudgetDecision(True)
        if selected_phase != "active":
            raise ValueError("phase must be active or finalization")
        if self.phase != "active":
            return BudgetDecision(False, "scientific phase has ended", False)
        if int(self.totals["tool_calls"]) >= int(self.limits["tool_calls"]):
            return self._deny("tool_calls")
        self.totals["tool_calls"] += 1
        self._append("tool_call", int(self.totals["agent_turns"]), reason=tool_name, enforce_limits=False)
        return BudgetDecision(True)

    def reserve_xtb_job(self, job_id: str, *, scan_points: int = 0) -> BudgetDecision:
        """Atomically reserve one xTB job and all declared scan points."""
        if isinstance(scan_points, bool) or not isinstance(scan_points, int) or scan_points < 0:
            raise ValueError("scan_points must be a non-negative integer")
        if self.phase != "active":
            return BudgetDecision(False, "scientific phase has ended", False)
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if job_id in self._reserved_job_ids:
            raise ValueError(f"job_id already reserved: {job_id}")
        exceeded: list[str] = []
        if int(self.totals["xtb_jobs"]) + 1 > int(self.limits["xtb_jobs"]):
            exceeded.append("xtb_jobs")
        if int(self.totals["xtb_scan_points"]) + scan_points > int(self.limits["xtb_scan_points"]):
            exceeded.append("xtb_scan_points")
        if exceeded:
            return self._deny(",".join(exceeded))
        self._reserved_job_ids.add(job_id)
        self.totals["xtb_jobs"] += 1
        self.totals["xtb_scan_points"] += scan_points
        self._append("xtb_job", int(self.totals["agent_turns"]), reason=job_id, enforce_limits=False)
        return BudgetDecision(True)

    def reserve_xtb_compute(
        self,
        job_id: str,
        *,
        compute_class: str,
        credits: int,
        candidate_id: str | None,
        condition: str | None,
        parent_job_id: str | None,
        recovery_of_job_id: str | None,
    ) -> BudgetDecision:
        """Atomically admit one weighted Task 04 xTB request."""

        if self.compute_budget is None:
            raise ValueError("weighted compute budget is not configured")
        if self.phase != "active":
            return BudgetDecision(False, "scientific phase has ended", False)
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if job_id in self._reserved_job_ids:
            raise ValueError(f"job_id already reserved: {job_id}")
        if compute_class not in TASK04_COMPUTE_CLASSES:
            raise ValueError(f"unknown xTB compute class: {compute_class}")
        expected_credits = task04_compute_cost(compute_class, self.compute_budget)
        if isinstance(credits, bool) or not isinstance(credits, int) or credits != expected_credits:
            raise ValueError(f"compute credits for {compute_class} must equal {expected_credits}")
        if compute_class == "scan" and self.compute_budget.get("scan_allowed") is not True:
            reason = "scan is not allowed for Task 04"
            self._append_compute_event(
                "compute_rejected", job_id=job_id, compute_class=compute_class, credits=credits,
                pool=None, candidate_id=candidate_id, condition=condition,
                parent_job_id=parent_job_id, recovery_of_job_id=recovery_of_job_id, reason=reason,
            )
            return BudgetDecision(False, reason, False)
        if int(self.totals["xtb_jobs"]) + 1 > int(self.compute_budget["max_jobs"]):
            reason = "budget limit exceeded: max_jobs"
            self._append_compute_event(
                "compute_rejected", job_id=job_id, compute_class=compute_class, credits=credits,
                pool=None, candidate_id=candidate_id, condition=condition,
                parent_job_id=parent_job_id, recovery_of_job_id=recovery_of_job_id, reason=reason,
            )
            return BudgetDecision(False, reason, False)

        pool = "main"
        chain_root: str | None = None
        if recovery_of_job_id is not None:
            source = self._compute_jobs.get(str(recovery_of_job_id))
            reason = None
            if source is None:
                reason = "recovery source job is not admitted"
            elif source.get("status") not in {"failed", "timeout"}:
                reason = "recovery source job is not failed or timeout"
            elif source.get("candidate_id") != candidate_id or source.get("condition") != condition:
                reason = "recovery must keep the same candidate and condition lineage"
            if reason is not None:
                self._append_compute_event(
                    "compute_rejected", job_id=job_id, compute_class=compute_class, credits=credits,
                    pool="recovery", candidate_id=candidate_id, condition=condition,
                    parent_job_id=parent_job_id, recovery_of_job_id=recovery_of_job_id, reason=reason,
                )
                return BudgetDecision(False, reason, False)
            pool = "recovery"
            chain_root = str(source.get("chain_root") or recovery_of_job_id)
            chain_is_new = chain_root not in self._recovery_chain_credits
            if chain_is_new and len(self._recovery_chain_credits) >= int(self.compute_budget["max_recovery_chains"]):
                reason = "recovery chain limit exceeded"
            elif self._recovery_chain_credits.get(chain_root, 0) + credits > int(
                self.compute_budget["max_recovery_credits_per_chain"]
            ):
                reason = "recovery credits per chain exceeded"
            elif self.compute_totals["recovery_credits_used"] + credits > int(self.compute_budget["recovery_credits"]):
                reason = "recovery credit pool exhausted"
            else:
                reason = None
            if reason is not None:
                self._append_compute_event(
                    "compute_rejected", job_id=job_id, compute_class=compute_class, credits=credits,
                    pool=pool, candidate_id=candidate_id, condition=condition,
                    parent_job_id=parent_job_id, recovery_of_job_id=recovery_of_job_id, reason=reason,
                )
                return BudgetDecision(False, reason, False)
        elif self.compute_totals["main_credits_used"] + credits > int(self.compute_budget["main_credits"]):
            reason = "main compute credit pool exhausted"
            self._append_compute_event(
                "compute_rejected", job_id=job_id, compute_class=compute_class, credits=credits,
                pool=pool, candidate_id=candidate_id, condition=condition,
                parent_job_id=parent_job_id, recovery_of_job_id=None, reason=reason,
            )
            return BudgetDecision(False, reason, False)

        self._reserved_job_ids.add(job_id)
        self.totals["xtb_jobs"] += 1
        self.compute_totals[f"{pool}_credits_used"] += credits
        self.compute_totals["total_credits_used"] += credits
        self.compute_totals["main_credits_remaining"] = int(self.compute_budget["main_credits"]) - int(
            self.compute_totals["main_credits_used"]
        )
        self.compute_totals["recovery_credits_remaining"] = int(self.compute_budget["recovery_credits"]) - int(
            self.compute_totals["recovery_credits_used"]
        )
        self.compute_totals["admitted_jobs"] += 1
        if pool == "recovery" and chain_root is not None:
            self._recovery_chain_credits[chain_root] = self._recovery_chain_credits.get(chain_root, 0) + credits
        self._compute_jobs[job_id] = {
            "candidate_id": candidate_id,
            "condition": condition,
            "status": "reserved",
            "chain_root": chain_root,
        }
        self._append("xtb_job", int(self.totals["agent_turns"]), reason=job_id, enforce_limits=False)
        self._append_compute_event(
            "compute_admitted", job_id=job_id, compute_class=compute_class, credits=credits,
            pool=pool, candidate_id=candidate_id, condition=condition,
            parent_job_id=parent_job_id, recovery_of_job_id=recovery_of_job_id,
        )
        return BudgetDecision(True)

    def record_xtb_job_outcome(self, job_id: str, status: str) -> None:
        """Record the terminal state used to authorize a later recovery."""

        if job_id not in self._compute_jobs:
            raise ValueError(f"job is not admitted: {job_id}")
        if status not in {"completed", "failed", "timeout"}:
            raise ValueError("xTB job status must be completed, failed or timeout")
        previous = self._compute_jobs[job_id]["status"]
        if previous in {"completed", "failed", "timeout"}:
            raise ValueError(f"xTB job already has a terminal outcome: {job_id}")
        self._compute_jobs[job_id]["status"] = status
        if status == "completed":
            self.record_xtb_job_completed(int(self.totals["agent_turns"]))

    def reserve_scan_points(self, job_id: str, point_count: int) -> BudgetDecision:
        """Reserve a scan job and its points through the same atomic boundary."""
        return self.reserve_xtb_job(job_id, scan_points=point_count)

    def record_xtb_job(self, logical_turn: int, *, scan_points: int = 0) -> None:
        job_id = f"legacy_job_{int(self.totals['xtb_jobs']) + 1:06d}"
        decision = self.reserve_xtb_job(job_id, scan_points=scan_points)
        if not decision.allowed:
            raise BudgetExceededError(decision.reason or "xTB budget exceeded")

    def record_empty_response_retry(self, logical_turn: int) -> None:
        """Record an empty-response semantic retry (spec 5.3)."""
        self._spec53_extra["empty_response_retries"] += 1
        self._spec53_extra["recovery_attempts"] += 1

    def record_xtb_job_completed(self, logical_turn: int, *, count: int = 1) -> None:
        """Record completed xTB jobs (spec 5.3)."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        self._spec53_extra["xtb_jobs_completed"] += count

    def record_process_resume(self, logical_turn: int, reason: str) -> None:
        self._spec53_extra["recovery_attempts"] += 1
        self._append("process_resume", logical_turn, reason=reason, enforce_limits=False)

    def _spec53_totals(self) -> dict:
        result = {}
        for historical, spec53 in _HISTORICAL_TO_SPEC53.items():
            result[spec53] = self.totals[historical]
        result["empty_response_retries"] = self._spec53_extra["empty_response_retries"]
        result["recovery_attempts"] = self._spec53_extra["recovery_attempts"]
        result["xtb_jobs_completed"] = self._spec53_extra["xtb_jobs_completed"]
        return result

    def to_document(self) -> dict:
        # Freeze the final elapsed value in an append-only event so serialized
        # totals exactly match the final cumulative snapshot.  Updating only
        # ``totals`` here would make an otherwise valid ledger unverifiable.
        logical_turn = max((int(event["logical_turn"]) for event in self.events), default=0)
        self._append("finalize", logical_turn, enforce_limits=False)
        document = {
            "schema_version": "3.0.0" if self.compute_budget is not None else "2.0.0",
            # Frozen spec 5.3 canonical counters at top level.
            **self._spec53_totals(),
            # Historical compatibility aliases.
            "limits": deepcopy(self.limits),
            "events": deepcopy(self.events),
            "totals": deepcopy(self.totals),
            "phase": self.phase,
            "exhaustion_trigger": self.exhaustion_trigger,
            "rejected_requests": self.rejected_requests,
            "finalization_turns_used": self.finalization_turns_used,
            "finalization_tool_calls": self.finalization_tool_calls,
            "phase_history": deepcopy(self.phase_history),
            "finalization_notice": FINALIZATION_NOTICE,
        }
        if self.compute_budget is not None:
            document["compute_budget"] = deepcopy(self.compute_budget)
            document["compute_totals"] = deepcopy(self.compute_totals)
            document["compute_events"] = deepcopy(self.compute_events)
        validate_budget_ledger(document, self.schema_dir)
        return document

    @classmethod
    def from_document(
        cls,
        document: Mapping,
        *,
        schema_dir: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> "BudgetLedger":
        validate_budget_ledger(document, schema_dir)
        ledger = cls(
            document["limits"],
            compute_budget=document.get("compute_budget"),
            schema_dir=schema_dir,
            clock=clock,
        )
        ledger.events = deepcopy(list(document["events"]))
        ledger.totals = deepcopy(dict(document["totals"]))
        ledger.phase = str(document.get("phase", "active"))
        ledger.exhaustion_trigger = document.get("exhaustion_trigger")
        ledger.rejected_requests = int(document.get("rejected_requests", 0))
        ledger.finalization_turns_used = int(document.get("finalization_turns_used", 0))
        ledger.finalization_tool_calls = int(document.get("finalization_tool_calls", 0))
        ledger.phase_history = deepcopy(list(document.get("phase_history", [{"phase": "active", "trigger": "run_started"}])))
        ledger._reserved_job_ids = {
            str(event.get("retry_or_resume_reason"))
            for event in ledger.events
            if event.get("event_type") == "xtb_job" and event.get("retry_or_resume_reason")
        }
        for field in ("empty_response_retries", "recovery_attempts", "xtb_jobs_completed"):
            ledger._spec53_extra[field] = int(document.get(field, 0))
        if ledger.compute_budget is not None:
            ledger.compute_events = deepcopy(list(document.get("compute_events", [])))
            ledger.compute_totals = deepcopy(dict(document.get("compute_totals", {})))
            ledger._compute_jobs = {}
            ledger._recovery_chain_credits = {}
            for event in ledger.compute_events:
                if event.get("event_type") != "compute_admitted":
                    continue
                job_id = str(event["job_id"])
                recovery_of = event.get("recovery_of_job_id")
                chain_root = None
                if recovery_of is not None:
                    source = ledger._compute_jobs.get(str(recovery_of), {})
                    chain_root = str(source.get("chain_root") or recovery_of)
                    ledger._recovery_chain_credits[chain_root] = (
                        ledger._recovery_chain_credits.get(chain_root, 0) + int(event["credits"])
                    )
                ledger._compute_jobs[job_id] = {
                    "candidate_id": event.get("candidate_id"),
                    "condition": event.get("condition"),
                    "status": "reserved",
                    "chain_root": chain_root,
                }
        ledger._begun_turns = {
            int(event["logical_turn"])
            for event in ledger.events
            if event["event_type"] == "agent_turn"
        }
        ledger.started = float(clock()) - float(ledger.totals["wall_seconds"])
        return ledger
