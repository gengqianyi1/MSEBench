"""Route-neutral primitives for scoring observable scientific work.

The module deliberately ignores assistant prose and hidden reasoning.  It
operates only on Harness-recorded actions, trusted results, explicit parent or
retry links, and final evidence adoption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


def partial_result_fraction(completed: int, possible: int) -> float:
    """Return bounded evidence coverage without treating cutoff as failure."""
    if isinstance(completed, bool) or isinstance(possible, bool):
        raise TypeError("coverage counters must be integers")
    if possible <= 0:
        return 0.0
    return max(0.0, min(1.0, float(completed) / float(possible)))


def dependency_gate(
    earned_points: Mapping[str, float],
    dependent_item_ids: Sequence[str],
    *,
    gate_failed: bool,
) -> dict[str, float]:
    """Zero only claims that logically depend on a failed upstream check."""
    result = {str(key): float(value) for key, value in earned_points.items()}
    if gate_failed:
        for item_id in dependent_item_ids:
            if str(item_id) in result:
                result[str(item_id)] = 0.0
    return result


def equal_score_fields(raw_score: float) -> dict[str, float]:
    """Emit the four official, uncalibrated score fields."""
    numeric = float(raw_score)
    if not math.isfinite(numeric):
        numeric = 0.0
    value = round(max(0.0, min(100.0, numeric)), 2)
    return {
        "score": value,
        "task_local_score": value,
        "scientific_basis_score": value,
        "comparison_score": value,
    }


@dataclass(frozen=True)
class AllocationScore:
    fraction: float
    linked_actions: int
    trusted_results: int
    adaptation_observed: bool
    validation_observed: bool
    final_adoption_observed: bool
    redundant_actions: int


def _events(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(value) for value in values if isinstance(value, Mapping)]


def score_observable_allocation(
    values: Iterable[Mapping[str, Any]],
    *,
    useful_units: int,
    total_actions: int,
) -> AllocationScore:
    """Score evidence-to-action linkage without prescribing an operation route.

    A broad screen followed by focus and a targeted calculation followed by an
    independent validation are intentionally equivalent when their observable
    evidence quality is equivalent.
    """
    events = _events(values)
    actions = [event for event in events if event.get("event_type") in {"action", "revision"}]
    results = {
        str(event.get("job_id")): event
        for event in events
        if event.get("event_type") == "result" and event.get("job_id")
    }
    linked_actions = sum(str(action.get("job_id")) in results for action in actions)
    trusted_results = sum(bool(result.get("trusted_result")) for result in results.values())
    issue_ids = {
        str(result.get("job_id"))
        for result in results.values()
        if result.get("diagnostic_issue") is True
    }
    adaptation = any(
        event.get("event_type") == "revision"
        and str(event.get("parent_job_id") or "") in issue_ids
        and bool(results.get(str(event.get("job_id")), {}).get("trusted_result"))
        for event in events
    )
    validation = any(
        event.get("event_type") == "action"
        and bool(event.get("parent_job_id"))
        and bool(results.get(str(event.get("job_id")), {}).get("trusted_result"))
        for event in events
    )
    adopted = any(
        event.get("event_type") == "final_adoption"
        and str(event.get("job_id")) in results
        and bool(results[str(event.get("job_id"))].get("trusted_result"))
        for event in events
    )
    action_count = max(0, int(total_actions))
    useful_count = max(0, int(useful_units))
    useful_fraction = partial_result_fraction(useful_count, action_count)
    linkage_fraction = partial_result_fraction(linked_actions, len(actions))
    redundant = max(0, action_count - useful_count)
    fraction = (
        0.35 * useful_fraction
        + 0.30 * linkage_fraction
        + 0.20 * float(adaptation or validation)
        + 0.15 * float(adopted)
    )
    return AllocationScore(
        fraction=round(max(0.0, min(1.0, fraction)), 12),
        linked_actions=linked_actions,
        trusted_results=trusted_results,
        adaptation_observed=adaptation,
        validation_observed=validation,
        final_adoption_observed=adopted,
        redundant_actions=redundant,
    )


def job_events_from_evidence(
    jobs: Mapping[str, Any],
    *,
    adopted_job_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build conservative observable events from immutable job records.

    This fallback supports baselines and replay where no model conversation is
    present.  It never infers a decision from prose.
    """
    adopted = {str(value) for value in adopted_job_ids}
    events: list[dict[str, Any]] = []
    ordered = sorted(jobs.values(), key=lambda job: str(getattr(job, "job_id", "")))
    for job in ordered:
        job_id = str(getattr(job, "job_id", ""))
        record = getattr(job, "record", {}) or {}
        parsed = getattr(job, "parsed", {}) or {}
        parent = record.get("retry_of_job_id") or record.get("retry_of") or record.get("parent_job_id")
        status = parsed.get("status") if isinstance(parsed.get("status"), dict) else {}
        valid = bool(status.get("normal_termination")) and str(record.get("status")) == "completed"
        event_type = "revision" if parent and (record.get("retry_of_job_id") or record.get("retry_of")) else "action"
        events.append({"event_type": event_type, "job_id": job_id, "parent_job_id": parent})
        events.append({"event_type": "result", "job_id": job_id, "trusted_result": valid, "diagnostic_issue": not valid})
        if job_id in adopted:
            events.append({"event_type": "final_adoption", "job_id": job_id})
    return events
