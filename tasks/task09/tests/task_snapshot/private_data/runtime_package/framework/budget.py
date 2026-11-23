"""Runtime guards and independent event-order budget reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence


ANGLE_GRID_DEG = 5
ANGLE_MIN_DEG = 0
ANGLE_MAX_EXCLUSIVE_DEG = 360
MAX_FIRST_TURN_NEW_ANGLES = 6


class BudgetError(ValueError):
    """A budget configuration or accounting operation is invalid."""


class BudgetExceededError(BudgetError):
    """A runner attempted an operation after its configured limit."""


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BudgetError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_scan_angle(value: Any) -> int:
    """Return a canonical angle, accepting only integer multiples of five."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError("scan angle must be a numeric integer")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise BudgetError("scan angle must be a finite integer")
    angle = int(value)
    if not ANGLE_MIN_DEG <= angle < ANGLE_MAX_EXCLUSIVE_DEG:
        raise BudgetError("scan angle must be in [0, 360)")
    if angle % ANGLE_GRID_DEG:
        raise BudgetError("scan angle must be an integer multiple of 5 degrees")
    return angle


@dataclass(frozen=True)
class BudgetLimits:
    max_tool_calls: int = 30
    max_agent_turns: int = 16
    max_unique_scan_angles: int = 16
    min_unique_valid_scan_angles: int = 8
    min_scan_feedback_rounds: int = 3
    min_profile_fit_rounds: int = 2
    max_profile_fits: int = 4
    max_attempts_per_angle: int = 2
    total_timeout_seconds: int = 1200
    single_xtb_timeout_seconds: int = 180
    unique_angle_warning_at: int = 14

    def __post_init__(self) -> None:
        for field_name in (
            "max_tool_calls",
            "max_agent_turns",
            "max_unique_scan_angles",
            "min_unique_valid_scan_angles",
            "min_scan_feedback_rounds",
            "min_profile_fit_rounds",
            "max_profile_fits",
            "max_attempts_per_angle",
            "total_timeout_seconds",
            "single_xtb_timeout_seconds",
            "unique_angle_warning_at",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.min_unique_valid_scan_angles > self.max_unique_scan_angles:
            raise BudgetError("minimum valid angles exceed the unique-angle budget")
        if self.min_profile_fit_rounds > self.max_profile_fits:
            raise BudgetError("minimum profile fits exceed the profile-fit budget")
        if self.unique_angle_warning_at >= self.max_unique_scan_angles:
            raise BudgetError("unique-angle warning must occur before finalization")
        if self.single_xtb_timeout_seconds > self.total_timeout_seconds:
            raise BudgetError("single xTB timeout exceeds total timeout")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetLimits":
        if not isinstance(value, Mapping):
            raise BudgetError("budget configuration must be an object")
        raw: Mapping[str, Any] = value
        if "budgets" in value:
            nested = value["budgets"]
            if not isinstance(nested, Mapping):
                raise BudgetError("budgets must be an object")
            raw = nested
        elif "budget" in value:
            nested = value["budget"]
            if not isinstance(nested, Mapping):
                raise BudgetError("budget must be an object")
            raw = nested
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise BudgetError("unknown budget field(s): " + ", ".join(sorted(unknown)))
        return cls(**{name: raw[name] for name in known if name in raw})

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ScanReservation:
    accepted: bool
    angle_deg: int | None
    attempt_number: int | None
    new_unique_angle: bool
    warning: bool
    finalization_only: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "angle_deg": self.angle_deg,
            "attempt_number": self.attempt_number,
            "new_unique_angle": self.new_unique_angle,
            "warning": self.warning,
            "finalization_only": self.finalization_only,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BudgetSnapshot:
    tool_calls: int
    agent_turns: int
    unique_scan_angles: tuple[int, ...]
    valid_unique_scan_angles: tuple[int, ...]
    scan_attempts_by_angle: tuple[tuple[int, int], ...]
    profile_fits: int
    valid_profile_fits: int
    feedback_rounds: int
    feedback_round_turns: tuple[int, ...]
    first_turn_unique_scan_angles: int
    warning: bool
    warning_message: str | None
    finalization_only: bool
    violations: tuple[str, ...] = ()
    new_unique_angles_this_round: int = 0
    cumulative_unique_angles: int = 0
    unique_angles_remaining: int = 16
    scan_feedback_rounds: int = 0
    profile_fit_rounds: int = 0
    current_sampling_phase: str = "initial"
    phase_new_unique_angle_limit: int = 6
    phase_new_unique_angles_remaining: int = 6
    phase_fit_pending_analysis: bool = False
    phase_fit_profile_id: str | None = None
    initial_sampling_complete: bool = False
    initial_sampling_new_unique_angles: int = 0
    third_feedback_round_reachable: bool = True
    eligibility_gaps: Mapping[str, int] | None = None
    completed_sampling_rounds: tuple[Mapping[str, Any], ...] = ()
    sampling_turns: tuple[int, ...] = ()

    @property
    def unique_scan_angle_count(self) -> int:
        return len(self.unique_scan_angles)

    @property
    def valid_unique_scan_angle_count(self) -> int:
        return len(self.valid_unique_scan_angles)

    @property
    def eligible_minimum_process(self) -> bool:
        return (
            not self.violations
            and self.valid_unique_scan_angle_count >= 8
            and self.feedback_rounds >= 3
            and self.valid_profile_fits >= 2
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "agent_turns": self.agent_turns,
            "unique_scan_angles": list(self.unique_scan_angles),
            "unique_scan_angle_count": self.unique_scan_angle_count,
            "valid_unique_scan_angles": list(self.valid_unique_scan_angles),
            "valid_unique_scan_angle_count": self.valid_unique_scan_angle_count,
            "scan_attempts_by_angle": {
                str(angle): count for angle, count in self.scan_attempts_by_angle
            },
            "profile_fits": self.profile_fits,
            "valid_profile_fits": self.valid_profile_fits,
            "feedback_rounds": self.feedback_rounds,
            "feedback_round_turns": list(self.feedback_round_turns),
            "first_turn_unique_scan_angles": self.first_turn_unique_scan_angles,
            "first_sampling_turn_new_angles": self.first_turn_unique_scan_angles,
            "warning": self.warning,
            "warning_message": self.warning_message,
            "finalization_only": self.finalization_only,
            "violations": list(self.violations),
            "new_unique_angles_this_round": self.new_unique_angles_this_round,
            "cumulative_unique_angles": self.cumulative_unique_angles,
            "unique_angles_remaining": self.unique_angles_remaining,
            "scan_feedback_rounds": self.scan_feedback_rounds,
            "profile_fit_rounds": self.profile_fit_rounds,
            "current_sampling_phase": self.current_sampling_phase,
            "phase_new_unique_angle_limit": self.phase_new_unique_angle_limit,
            "phase_new_unique_angles_remaining": self.phase_new_unique_angles_remaining,
            "phase_fit_pending_analysis": self.phase_fit_pending_analysis,
            "phase_fit_profile_id": self.phase_fit_profile_id,
            "initial_sampling_complete": self.initial_sampling_complete,
            "initial_sampling_new_unique_angles": self.initial_sampling_new_unique_angles,
            "third_feedback_round_reachable": self.third_feedback_round_reachable,
            "eligibility_gaps": dict(self.eligibility_gaps or {}),
            "completed_sampling_rounds": [
                dict(item) for item in self.completed_sampling_rounds
            ],
            "sampling_turns": list(self.sampling_turns),
        }


class BudgetLedger:
    """Fail-closed online accounting used by the trusted runner.

    ``consume_tool_call`` is intentionally separate from tool-specific guards:
    malformed and illegal-angle calls still consume the global tool-call budget.
    """

    def __init__(self, limits: BudgetLimits | Mapping[str, Any] | None = None) -> None:
        if limits is None:
            limits = BudgetLimits()
        elif isinstance(limits, Mapping):
            limits = BudgetLimits.from_mapping(limits)
        if not isinstance(limits, BudgetLimits):
            raise BudgetError("limits must be BudgetLimits or a mapping")
        self.limits = limits
        self.tool_calls = 0
        self.agent_turns = 0
        self._unique_angles: list[int] = []
        self._unique_angle_set: set[int] = set()
        self._valid_angles: set[int] = set()
        self._attempts: dict[int, int] = {}
        self._profile_fits = 0
        self._profile_fit_results: dict[int, bool] = {}
        self._first_turn_unique_angles = 0

    @property
    def warning(self) -> bool:
        return len(self._unique_angles) >= self.limits.unique_angle_warning_at

    @property
    def finalization_only(self) -> bool:
        return len(self._unique_angles) >= self.limits.max_unique_scan_angles

    @property
    def warning_message(self) -> str | None:
        if not self.warning:
            return None
        return (
            f"unique scan-angle budget warning: {len(self._unique_angles)}/"
            f"{self.limits.max_unique_scan_angles} consumed"
        )

    def begin_agent_turn(self) -> int:
        if self.agent_turns >= self.limits.max_agent_turns:
            raise BudgetExceededError("agent-turn budget exhausted")
        self.agent_turns += 1
        return self.agent_turns

    def consume_tool_call(self) -> int:
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceededError("tool-call budget exhausted")
        self.tool_calls += 1
        return self.tool_calls

    def reserve_scan(self, angle: Any, *, assistant_turn: int | None = None) -> ScanReservation:
        """Reserve one accepted native attempt before starting xTB.

        Invalid angles never reserve an attempt or unique-angle slot.  A newly
        accepted angle is recorded immediately, so a later xTB failure still
        consumes that slot.  Once 16/16 is reached no further native scan,
        including a retry, may start.
        """

        try:
            canonical = normalize_scan_angle(angle)
        except BudgetError as exc:
            return ScanReservation(
                accepted=False,
                angle_deg=None,
                attempt_number=None,
                new_unique_angle=False,
                warning=self.warning,
                finalization_only=self.finalization_only,
                reason=str(exc),
            )
        if self.finalization_only:
            return ScanReservation(
                accepted=False,
                angle_deg=canonical,
                attempt_number=None,
                new_unique_angle=False,
                warning=True,
                finalization_only=True,
                reason="finalization_only forbids further native scans",
            )
        attempts = self._attempts.get(canonical, 0)
        if attempts >= self.limits.max_attempts_per_angle:
            return ScanReservation(
                accepted=False,
                angle_deg=canonical,
                attempt_number=None,
                new_unique_angle=False,
                warning=self.warning,
                finalization_only=self.finalization_only,
                reason="per-angle attempt budget exhausted",
            )
        new_unique = canonical not in self._unique_angle_set
        if (
            new_unique
            and assistant_turn == 1
            and self._first_turn_unique_angles >= MAX_FIRST_TURN_NEW_ANGLES
        ):
            return ScanReservation(
                accepted=False,
                angle_deg=canonical,
                attempt_number=None,
                new_unique_angle=False,
                warning=self.warning,
                finalization_only=self.finalization_only,
                reason="first assistant turn may add at most six unique angles",
            )
        if new_unique and len(self._unique_angles) >= self.limits.max_unique_scan_angles:
            return ScanReservation(
                accepted=False,
                angle_deg=canonical,
                attempt_number=None,
                new_unique_angle=False,
                warning=True,
                finalization_only=True,
                reason="unique scan-angle budget exhausted",
            )
        if new_unique:
            self._unique_angle_set.add(canonical)
            self._unique_angles.append(canonical)
            if assistant_turn == 1:
                self._first_turn_unique_angles += 1
        attempt_number = attempts + 1
        self._attempts[canonical] = attempt_number
        return ScanReservation(
            accepted=True,
            angle_deg=canonical,
            attempt_number=attempt_number,
            new_unique_angle=new_unique,
            warning=self.warning,
            finalization_only=self.finalization_only,
        )

    def record_scan_result(self, angle: Any, *, valid: bool) -> None:
        canonical = normalize_scan_angle(angle)
        if self._attempts.get(canonical, 0) < 1:
            raise BudgetError("cannot record a result for an unreserved scan")
        if not isinstance(valid, bool):
            raise BudgetError("scan validity must be boolean")
        if valid:
            self._valid_angles.add(canonical)

    def reserve_profile_fit(self) -> int:
        if self._profile_fits >= self.limits.max_profile_fits:
            raise BudgetExceededError("profile-fit budget exhausted")
        self._profile_fits += 1
        return self._profile_fits

    def record_profile_fit_result(self, fit_number: int, *, valid: bool) -> None:
        if (
            isinstance(fit_number, bool)
            or not isinstance(fit_number, int)
            or not 1 <= fit_number <= self._profile_fits
        ):
            raise BudgetError("profile-fit result has no reservation")
        if fit_number in self._profile_fit_results:
            raise BudgetError("profile-fit result is immutable")
        if not isinstance(valid, bool):
            raise BudgetError("profile-fit validity must be boolean")
        self._profile_fit_results[fit_number] = valid

    def snapshot(
        self,
        *,
        feedback_round_turns: Sequence[int] = (),
        violations: Sequence[str] = (),
    ) -> BudgetSnapshot:
        feedback = tuple(sorted(set(int(turn) for turn in feedback_round_turns)))
        cumulative = len(self._unique_angles)
        feedback_count = len(feedback)
        finalization = self.finalization_only
        current_phase = (
            "finalization_only"
            if finalization
            else ("post_feedback" if feedback_count >= 3 else "initial")
        )
        return BudgetSnapshot(
            tool_calls=self.tool_calls,
            agent_turns=self.agent_turns,
            unique_scan_angles=tuple(self._unique_angles),
            valid_unique_scan_angles=tuple(
                angle for angle in self._unique_angles if angle in self._valid_angles
            ),
            scan_attempts_by_angle=tuple(sorted(self._attempts.items())),
            profile_fits=self._profile_fits,
            valid_profile_fits=sum(self._profile_fit_results.values()),
            feedback_rounds=len(feedback),
            feedback_round_turns=feedback,
            first_turn_unique_scan_angles=self._first_turn_unique_angles,
            warning=self.warning,
            warning_message=self.warning_message,
            finalization_only=finalization,
            violations=tuple(violations),
            new_unique_angles_this_round=cumulative if feedback_count == 0 else 0,
            cumulative_unique_angles=cumulative,
            unique_angles_remaining=max(
                0, self.limits.max_unique_scan_angles - cumulative
            ),
            scan_feedback_rounds=feedback_count,
            profile_fit_rounds=sum(self._profile_fit_results.values()),
            current_sampling_phase=current_phase,
            phase_new_unique_angle_limit=(
                0 if finalization else MAX_FIRST_TURN_NEW_ANGLES
            ),
            phase_new_unique_angles_remaining=(
                0
                if finalization
                else max(0, MAX_FIRST_TURN_NEW_ANGLES - cumulative)
            ),
            initial_sampling_complete=feedback_count > 0,
            initial_sampling_new_unique_angles=self._first_turn_unique_angles,
            eligibility_gaps={
                "valid_unique_scan_angles_remaining": max(
                    0,
                    self.limits.min_unique_valid_scan_angles
                    - len(self._valid_angles),
                ),
                "scan_feedback_rounds_remaining": max(
                    0, self.limits.min_scan_feedback_rounds - feedback_count
                ),
                "profile_fit_rounds_remaining": max(
                    0,
                    self.limits.min_profile_fit_rounds
                    - sum(self._profile_fit_results.values()),
                ),
            },
        )


@dataclass
class _Operation:
    call_id: str
    name: str
    arguments: dict[str, Any]
    turn: int
    call_sequence: int
    turn_had_prior_feedback: bool


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, Mapping) else event


def _tool_name(payload: Mapping[str, Any]) -> str:
    for key in ("name", "tool", "tool_name"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    function = payload.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    return ""


def _arguments(payload: Mapping[str, Any]) -> dict[str, Any]:
    value: Any = payload.get("arguments", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _call_id(payload: Mapping[str, Any], sequence: int) -> str:
    for key in ("call_id", "tool_call_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return f"__unpaired_{sequence}"


def _result_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("result")
    return value if isinstance(value, Mapping) else payload


def _status(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    for source in (result, payload):
        value = source.get("status")
        if isinstance(value, str):
            return value.upper()
        error = source.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("status"), str):
            return str(error["status"]).upper()
    return ""


def _accepted(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    tool_name: str,
) -> bool:
    for source in (result, payload):
        value = source.get("accepted")
        if isinstance(value, bool):
            return value
    status = _status(payload, result)
    if tool_name == "run_scan_point" and (
        isinstance(result.get("scan"), Mapping)
        or result.get("native_executed") is True
        or status in {"SCAN_COMPLETE", "SCAN_FAILED"}
        or (
            isinstance(result.get("attempt_id"), str)
            and isinstance(result.get("unique_angle_reserved"), bool)
        )
    ):
        return True
    if status in {"REJECTED", "INVALID", "INVALID_ANGLE", "BUDGET_REJECTED"}:
        return False
    # An xTB failure/timeout occurs after reservation and therefore consumes the
    # new angle and attempt budgets.
    if status in {"FAILED", "FAILURE", "TIMEOUT", "TIMED_OUT"}:
        return True
    for source in (result, payload):
        ok = source.get("ok")
        if isinstance(ok, bool):
            return ok
    return bool(result)


def _valid_result(payload: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    scan = result.get("scan")
    if isinstance(scan, Mapping):
        eligible = scan.get("profile_eligible")
        if isinstance(eligible, bool):
            return eligible
    for key in ("profile_eligible", "valid", "eligible"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    status = _status(payload, result)
    if status in {"FAILED", "FAILURE", "TIMEOUT", "TIMED_OUT", "REJECTED", "INVALID"}:
        return False
    for source in (result, payload):
        ok = source.get("ok")
        if isinstance(ok, bool):
            return ok
    return True


def _operation_angle(operation: _Operation, result: Mapping[str, Any]) -> int | None:
    for source in (result, operation.arguments):
        for key in (
            "canonical_angle_deg",
            "angle_deg",
            "requested_angle_deg",
            "angle",
        ):
            if key in source:
                try:
                    return normalize_scan_angle(source[key])
                except BudgetError:
                    return None
    return None


_FEEDBACK_TOOLS = frozenset(
    {
        "run_scan_point",
        "inspect_scan_attempt",
        "list_scan_points",
        "fit_periodic_profile",
        "analyze_profile",
    }
)


def _profile_id_from_result(result: Mapping[str, Any]) -> str | None:
    profile = result.get("profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("profile_id"), str):
        return str(profile["profile_id"])
    value = result.get("profile_id")
    return str(value) if isinstance(value, str) and value else None


def _trajectory_sampling_metadata(
    completed: Sequence[tuple[_Operation, Mapping[str, Any], bool, int]],
    *,
    limits: BudgetLimits,
) -> tuple[dict[str, Any], list[str]]:
    """Independently derive the v1.1 sampling phase from paired events."""

    unique: list[int] = []
    unique_set: set[int] = set()
    valid: set[int] = set()
    attempts: dict[int, int] = {}
    first_sampling_turn: int | None = None
    first_sampling_turn_new = 0
    sampling_turns: list[int] = []
    initial_complete = False
    feedback_rounds = 0
    feedback_turns: list[int] = []
    fit_rounds = 0
    phase_opened_turn = 0
    phase_start_sequence = 0
    phase_new: list[dict[str, int]] = []
    phase_fit: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    violations: list[str] = []

    for operation, result, accepted, result_sequence in completed:
        if operation.name == "run_scan_point" and accepted:
            angle = _operation_angle(operation, result)
            if angle is None:
                violations.append(
                    f"tool_result:{operation.call_id}:accepted_invalid_angle"
                )
                continue
            attempts[angle] = attempts.get(angle, 0) + 1
            if attempts[angle] > limits.max_attempts_per_angle:
                violations.append(f"angle_{angle}:attempt_budget_exceeded")
            if angle not in unique_set:
                if len(unique) >= limits.max_unique_scan_angles:
                    violations.append(
                        f"tool_result:{operation.call_id}:scan_after_finalization_only"
                    )
                if phase_fit is not None and feedback_rounds < 3:
                    violations.append(
                        f"tool_result:{operation.call_id}:new_angle_after_phase_fit_before_analysis"
                    )
                phase_limit = 6 if not initial_complete else 3
                if feedback_rounds < 3 and len(phase_new) >= phase_limit:
                    violations.append(
                        f"tool_result:{operation.call_id}:sampling_phase_unique_angle_limit_exceeded"
                    )
                if (
                    initial_complete
                    and feedback_rounds < 3
                    and operation.turn <= phase_opened_turn
                ):
                    violations.append(
                        f"tool_result:{operation.call_id}:refinement_not_in_later_assistant_turn"
                    )
                unique_set.add(angle)
                unique.append(angle)
                phase_new.append(
                    {
                        "angle_deg": angle,
                        "turn": operation.turn,
                        "sequence": result_sequence,
                    }
                )
                if operation.turn not in sampling_turns:
                    sampling_turns.append(operation.turn)
                if first_sampling_turn is None:
                    first_sampling_turn = operation.turn
                if operation.turn == first_sampling_turn:
                    first_sampling_turn_new += 1
            if _valid_result({}, result):
                valid.add(angle)
            continue

        if (
            operation.name == "fit_periodic_profile"
            and accepted
            and _valid_result({}, result)
            and _profile_id_from_result(result) is not None
        ):
            fit_rounds += 1
            if fit_rounds > limits.max_profile_fits:
                violations.append("profile_fit_budget_exceeded")
            if feedback_rounds < 3:
                if phase_fit is not None:
                    violations.append(
                        f"tool_result:{operation.call_id}:multiple_fits_before_phase_analysis"
                    )
                elif initial_complete and not phase_new:
                    violations.append(
                        f"tool_result:{operation.call_id}:refinement_fit_without_new_angle"
                    )
                elif (
                    initial_complete
                    and any(
                        item["turn"] <= phase_opened_turn for item in phase_new
                    )
                ):
                    violations.append(
                        f"tool_result:{operation.call_id}:refinement_fit_without_later_turn_scan"
                    )
                else:
                    phase_fit = {
                        "profile_id": _profile_id_from_result(result),
                        "sequence": result_sequence,
                    }
            continue

        if (
            operation.name != "analyze_profile"
            or not accepted
            or not _valid_result({}, result)
            or phase_fit is None
            or _profile_id_from_result(result) != phase_fit["profile_id"]
        ):
            continue
        latest_new_sequence = max(
            (item["sequence"] for item in phase_new),
            default=phase_start_sequence,
        )
        if phase_fit["sequence"] <= latest_new_sequence:
            continue
        if not initial_complete:
            history.append(
                {
                    "phase": "initial",
                    "new_unique_angles": len(phase_new),
                    "angles": [item["angle_deg"] for item in phase_new],
                    "fit_profile_id": phase_fit["profile_id"],
                    "analysis_turn": operation.turn,
                    "feedback_round": None,
                }
            )
            initial_complete = True
        elif (
            phase_new
            and all(item["turn"] > phase_opened_turn for item in phase_new)
            and feedback_rounds < 3
        ):
            feedback_rounds += 1
            feedback_turn = min(item["turn"] for item in phase_new)
            feedback_turns.append(feedback_turn)
            history.append(
                {
                    "phase": f"refinement_{feedback_rounds}",
                    "new_unique_angles": len(phase_new),
                    "angles": [item["angle_deg"] for item in phase_new],
                    "fit_profile_id": phase_fit["profile_id"],
                    "analysis_turn": operation.turn,
                    "feedback_round": feedback_rounds,
                }
            )
        else:
            continue
        phase_opened_turn = operation.turn
        phase_start_sequence = result_sequence
        phase_new = []
        phase_fit = None

    finalization = len(unique) >= limits.max_unique_scan_angles
    if finalization:
        current_phase = "finalization_only"
        phase_limit = 0
        phase_remaining = 0
    elif not initial_complete:
        current_phase = "initial"
        phase_limit = 6
        phase_remaining = max(0, phase_limit - len(phase_new))
    elif feedback_rounds < 3:
        current_phase = f"refinement_{feedback_rounds + 1}"
        phase_limit = 3
        phase_remaining = max(0, phase_limit - len(phase_new))
    else:
        current_phase = "post_feedback"
        phase_limit = limits.max_unique_scan_angles
        phase_remaining = max(0, limits.max_unique_scan_angles - len(unique))

    feedback_remaining = max(0, limits.min_scan_feedback_rounds - feedback_rounds)
    unique_needed = max(
        0,
        feedback_remaining
        - (1 if initial_complete and bool(phase_new) else 0),
    )
    closures_needed = feedback_remaining + (0 if initial_complete else 1)
    third_reachable = (
        limits.max_unique_scan_angles - len(unique) >= unique_needed
        and limits.max_profile_fits - fit_rounds >= closures_needed
        and not violations
    )
    initial_new = next(
        (
            int(item["new_unique_angles"])
            for item in history
            if item["phase"] == "initial"
        ),
        len(phase_new) if not initial_complete else 0,
    )
    metadata = {
        "unique_angles": unique,
        "valid_angles": [angle for angle in unique if angle in valid],
        "attempts": attempts,
        "sampling_turns": sampling_turns,
        "feedback_round_turns": feedback_turns,
        "first_sampling_turn_new_angles": first_sampling_turn_new,
        "new_unique_angles_this_round": len(phase_new),
        "cumulative_unique_angles": len(unique),
        "unique_angles_remaining": max(
            0, limits.max_unique_scan_angles - len(unique)
        ),
        "scan_feedback_rounds": feedback_rounds,
        "profile_fit_rounds": fit_rounds,
        "current_sampling_phase": current_phase,
        "phase_new_unique_angle_limit": phase_limit,
        "phase_new_unique_angles_remaining": phase_remaining,
        "phase_fit_pending_analysis": phase_fit is not None,
        "phase_fit_profile_id": phase_fit["profile_id"] if phase_fit else None,
        "initial_sampling_complete": initial_complete,
        "initial_sampling_new_unique_angles": initial_new,
        "third_feedback_round_reachable": third_reachable,
        "eligibility_gaps": {
            "valid_unique_scan_angles_remaining": max(
                0, limits.min_unique_valid_scan_angles - len(valid)
            ),
            "scan_feedback_rounds_remaining": feedback_remaining,
            "profile_fit_rounds_remaining": max(
                0, limits.min_profile_fit_rounds - fit_rounds
            ),
        },
        "completed_sampling_rounds": history,
    }
    return metadata, violations


def recompute_budget(
    events: Sequence[Mapping[str, Any]],
    limits: BudgetLimits | Mapping[str, Any] | None = None,
) -> BudgetSnapshot:
    """Recompute accounting from paired events, ignoring claimed ``budget_state``."""

    if limits is None:
        parsed_limits = BudgetLimits()
    elif isinstance(limits, Mapping):
        parsed_limits = BudgetLimits.from_mapping(limits)
    elif isinstance(limits, BudgetLimits):
        parsed_limits = limits
    else:
        raise BudgetError("limits must be BudgetLimits or a mapping")

    assistant_turns = 0
    current_turn = 0
    current_turn_had_feedback = False
    last_feedback_sequence: int | None = None
    tool_calls = 0
    pending: dict[str, _Operation] = {}
    seen_call_ids: set[str] = set()
    unique_angles: list[int] = []
    unique_angle_set: set[int] = set()
    valid_angles: set[int] = set()
    attempts: dict[int, int] = {}
    profile_fits = 0
    valid_profile_fits = 0
    completed_operations: list[
        tuple[_Operation, Mapping[str, Any], bool, int]
    ] = []
    first_turn_unique_angles = 0
    first_sampling_turn: int | None = None
    violations: list[str] = []
    finalization_only = False

    for fallback_sequence, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            violations.append(f"event_{fallback_sequence}:not_an_object")
            continue
        event_name = raw_event.get("event", raw_event.get("type"))
        sequence_value = raw_event.get("sequence", fallback_sequence)
        sequence = sequence_value if isinstance(sequence_value, int) else fallback_sequence
        payload = _payload(raw_event)

        if event_name in {"assistant_message", "assistant_turn"}:
            assistant_turns += 1
            claimed_turn = payload.get("assistant_turn", payload.get("turn"))
            if claimed_turn is not None and claimed_turn != assistant_turns:
                violations.append(f"assistant_turn_{assistant_turns}:noncontiguous_claim")
            current_turn = assistant_turns
            current_turn_had_feedback = (
                last_feedback_sequence is not None and last_feedback_sequence < sequence
            )
            continue

        if event_name == "tool_call":
            tool_calls += 1
            call_id = _call_id(payload, sequence)
            if call_id in seen_call_ids:
                violations.append(f"tool_call:{call_id}:duplicate_id")
            seen_call_ids.add(call_id)
            pending[call_id] = _Operation(
                call_id=call_id,
                name=_tool_name(payload),
                arguments=_arguments(payload),
                turn=current_turn,
                call_sequence=sequence,
                turn_had_prior_feedback=current_turn_had_feedback,
            )
            continue

        if event_name != "tool_result":
            continue

        call_id = _call_id(payload, sequence)
        operation = pending.pop(call_id, None)
        if operation is None:
            violations.append(f"tool_result:{call_id}:unpaired")
            continue
        result = _result_mapping(payload)
        result_name = _tool_name(payload)
        if result_name and result_name != operation.name:
            violations.append(f"tool_result:{call_id}:name_mismatch")
        accepted = _accepted(payload, result, tool_name=operation.name)
        completed_operations.append((operation, result, accepted, sequence))

        if operation.name == "run_scan_point" and accepted:
            angle = _operation_angle(operation, result)
            if angle is None:
                violations.append(f"tool_result:{call_id}:accepted_invalid_angle")
            else:
                if finalization_only:
                    violations.append(f"tool_result:{call_id}:scan_after_finalization_only")
                new_unique = angle not in unique_angle_set
                if new_unique:
                    unique_angle_set.add(angle)
                    unique_angles.append(angle)
                    if first_sampling_turn is None:
                        first_sampling_turn = operation.turn
                    if operation.turn == first_sampling_turn:
                        first_turn_unique_angles += 1
                        if first_turn_unique_angles > MAX_FIRST_TURN_NEW_ANGLES:
                            violations.append("first_turn_unique_scan_angle_budget_exceeded")
                attempts[angle] = attempts.get(angle, 0) + 1
                if attempts[angle] > parsed_limits.max_attempts_per_angle:
                    violations.append(f"angle_{angle}:attempt_budget_exceeded")
                if len(unique_angles) > parsed_limits.max_unique_scan_angles:
                    violations.append("unique_scan_angle_budget_exceeded")
                if len(unique_angles) >= parsed_limits.max_unique_scan_angles:
                    finalization_only = True
                if _valid_result(payload, result):
                    valid_angles.add(angle)

        if operation.name == "fit_periodic_profile" and accepted:
            profile_fits += 1
            if profile_fits > parsed_limits.max_profile_fits:
                violations.append("profile_fit_budget_exceeded")
            if _valid_result(payload, result):
                valid_profile_fits += 1

        if operation.name in _FEEDBACK_TOOLS and accepted:
            last_feedback_sequence = sequence

    for call_id in pending:
        violations.append(f"tool_call:{call_id}:missing_result")
    if tool_calls > parsed_limits.max_tool_calls:
        violations.append("tool_call_budget_exceeded")
    if assistant_turns > parsed_limits.max_agent_turns:
        violations.append("agent_turn_budget_exceeded")

    phase, phase_violations = _trajectory_sampling_metadata(
        completed_operations,
        limits=parsed_limits,
    )
    violations.extend(phase_violations)

    warning = len(unique_angles) >= parsed_limits.unique_angle_warning_at
    warning_message = (
        f"unique scan-angle budget warning: {len(unique_angles)}/"
        f"{parsed_limits.max_unique_scan_angles} consumed"
        if warning
        else None
    )
    return BudgetSnapshot(
        tool_calls=tool_calls,
        agent_turns=assistant_turns,
        unique_scan_angles=tuple(unique_angles),
        valid_unique_scan_angles=tuple(
            angle for angle in unique_angles if angle in valid_angles
        ),
        scan_attempts_by_angle=tuple(sorted(attempts.items())),
        profile_fits=profile_fits,
        valid_profile_fits=valid_profile_fits,
        feedback_rounds=int(phase["scan_feedback_rounds"]),
        feedback_round_turns=tuple(phase["feedback_round_turns"]),
        first_turn_unique_scan_angles=int(
            phase["first_sampling_turn_new_angles"]
        ),
        warning=warning,
        warning_message=warning_message,
        finalization_only=finalization_only,
        violations=tuple(dict.fromkeys(violations)),
        new_unique_angles_this_round=int(
            phase["new_unique_angles_this_round"]
        ),
        cumulative_unique_angles=int(phase["cumulative_unique_angles"]),
        unique_angles_remaining=int(phase["unique_angles_remaining"]),
        scan_feedback_rounds=int(phase["scan_feedback_rounds"]),
        profile_fit_rounds=int(phase["profile_fit_rounds"]),
        current_sampling_phase=str(phase["current_sampling_phase"]),
        phase_new_unique_angle_limit=int(
            phase["phase_new_unique_angle_limit"]
        ),
        phase_new_unique_angles_remaining=int(
            phase["phase_new_unique_angles_remaining"]
        ),
        phase_fit_pending_analysis=bool(
            phase["phase_fit_pending_analysis"]
        ),
        phase_fit_profile_id=phase["phase_fit_profile_id"],
        initial_sampling_complete=bool(phase["initial_sampling_complete"]),
        initial_sampling_new_unique_angles=int(
            phase["initial_sampling_new_unique_angles"]
        ),
        third_feedback_round_reachable=bool(
            phase["third_feedback_round_reachable"]
        ),
        eligibility_gaps=dict(phase["eligibility_gaps"]),
        completed_sampling_rounds=tuple(
            dict(item) for item in phase["completed_sampling_rounds"]
        ),
        sampling_turns=tuple(int(item) for item in phase["sampling_turns"]),
    )


def reconstruct_feedback_round_turns(events: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return recompute_budget(events).feedback_round_turns


def reconstruct_feedback_rounds(events: Sequence[Mapping[str, Any]]) -> int:
    return recompute_budget(events).feedback_rounds


__all__ = [
    "ANGLE_GRID_DEG",
    "ANGLE_MAX_EXCLUSIVE_DEG",
    "ANGLE_MIN_DEG",
    "MAX_FIRST_TURN_NEW_ANGLES",
    "BudgetError",
    "BudgetExceededError",
    "BudgetLedger",
    "BudgetLimits",
    "BudgetSnapshot",
    "ScanReservation",
    "normalize_scan_angle",
    "recompute_budget",
    "reconstruct_feedback_round_turns",
    "reconstruct_feedback_rounds",
]
