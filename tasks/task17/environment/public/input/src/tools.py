"""The optional diagnostic tool layer the agent may call before committing a process.

WHY THIS EXISTS
---------------
External review found that the earlier version of this benchmark graded a bare chat model: the
harness called one evaluator automatically, so there was no tool selection, no argument filling, no
failure handling and no replanning, and the package could therefore not support any claim about agent
tool-interaction ability. This module is the fix. Each round the model may issue zero, one or several
tool calls, choose which tools, fill in the arguments itself, read structured results, recover from
real errors, and only then commit a process.

THE HARD CONSTRAINT ON WHAT A TOOL MAY RETURN
---------------------------------------------
No tool may turn the objective into a computable quantity. Concretely, and this is enforced by
`assert_no_objective_leak` plus a unit-style check in scripts/check_consistency.py:

  * NO tool returns the desirability D, an achievement fraction, a score, or a verdict.
  * NO tool returns any of the four simulated responses (d50, strength, porosity, yield_pct), nor a
    prediction, bound, ranking or sensitivity for any of them. Not one of the four, let alone all
    four -- returning all four would be handing over the answer, and returning one would be handing
    over a private handle the task is built on NOT having.
  * NO tool reads the surrogate, the population balance or the yield mass balance. `src/tools.py`
    does not import them.

What the tools DO return is diagnostics that are already derivable from the task description by
arithmetic: design-space membership, the hard-constraint checklist with its numeric margins, the
change-budget bookkeeping, the overlap with the registered process, and the declared level grid. The
point is not to give the model new information -- it is to make the *interaction* real, and to stop
rounds being lost to bookkeeping slips that have nothing to do with process judgement. Every number
below can be recomputed by hand from question.md; the shipped question text prints every formula and
every cost.

FAILURE IS PART OF THE CONTRACT
-------------------------------
Tools return real errors, not silent successes: unknown tool name, arguments that are not an object,
a missing required argument, an off-grid or malformed process, an unknown variable name, and a
per-round call quota that actually runs out. Each error carries `error`, `detail` and `recoverable`,
so a competent agent can repair the call and retry within the same round; the trajectory records
whether it did.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# tools.py imports ONLY the public half. It must not be able to reach the response mechanism, and
# `from design_grid import ...` is how that is enforced rather than promised.
from design_grid import (BP, BP_MARGIN_C, BUDGET, EDIT_COST, LEVELS, MAX_PROCESS_HOURS,
                         MAX_SOLVENT_L_PER_KG, SLOT_UNITS, SLOTS, START, TOTAL_EDIT_COST,
                         DesignError, change_overlap, changed_slots, constraints, edit_cost,
                         just_suspended_tip_speed, normalise_design, total_hours,
                         total_solvent_L_per_kg)

# The quantities a tool result must never mention. Checked, not trusted.
FORBIDDEN_RESULT_KEYS = {
    "d50", "strength", "porosity", "yield_pct", "desirability", "desirability_d", "d",
    "achievement", "achievement_fraction", "score", "points", "objective", "objective_value",
    "reference", "reference_objective", "best_objective_in_budget", "predicted_responses",
    "prediction", "sensitivity", "gradient",
}

MAX_TOOL_CALLS_PER_TURN = 4
MAX_TOOL_TURNS_PER_ROUND = 2


# =================================================================================================
# tool implementations
# =================================================================================================
def _require_process(arguments: Dict[str, Any], key: str = "process") -> Tuple[Any, Optional[Dict]]:
    if key not in arguments:
        return None, {
            "error": "missing_argument",
            "detail": f"this tool requires the argument {key!r}: an object with the twelve process "
                      f"variables.",
            "required_arguments": [key],
            "recoverable": True,
        }
    if not isinstance(arguments[key], dict):
        return None, {
            "error": "invalid_argument_type",
            "detail": f"{key!r} must be a JSON object keyed by variable name, got "
                      f"{type(arguments[key]).__name__}.",
            "recoverable": True,
        }
    return arguments[key], None


def tool_check_process_legality(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Is this process a member of the design space, and can it be run?

    Returns the per-variable membership result and the hard-constraint checklist with numeric
    margins. Returns nothing about quality: no response, no prediction, no ranking.
    """
    raw, err = _require_process(arguments)
    if err:
        return err

    # Membership, reported variable by variable rather than as one refusal, because "which key did I
    # get wrong" is exactly the thing a model wastes a round on.
    per_variable: Dict[str, Any] = {}
    missing = [s for s in SLOTS if s not in raw]
    unknown = [k for k in raw if k not in SLOTS]
    for s in SLOTS:
        if s not in raw:
            per_variable[s] = {"status": "missing", "allowed_values": LEVELS[s]}
            continue
        try:
            snapped = normalise_design({**{k: START[k] for k in SLOTS}, s: raw[s]})[s]
            per_variable[s] = {"status": "ok", "value": snapped}
        except DesignError as exc:
            per_variable[s] = {"status": "off_grid", "submitted": raw[s],
                               "allowed_values": LEVELS[s], "detail": str(exc)}

    on_grid = not missing and not unknown and all(
        v["status"] == "ok" for v in per_variable.values())
    result: Dict[str, Any] = {
        "tool": "check_process_legality",
        "design_space_membership": {
            "on_grid": on_grid,
            "missing_variables": missing,
            "unknown_variables": unknown,
            "per_variable": per_variable,
            "note": "C1. A process that is not on-grid is not measured at all in a graded round.",
        },
    }
    if not on_grid:
        result["hard_constraints"] = {
            "evaluated": False,
            "detail": "the process is not a design-space member, so C3-C7 were not evaluated.",
        }
        return result

    design = normalise_design(raw)
    viol = constraints(design)
    bp = BP[design["S1_bridging_liquid"]]
    result["hard_constraints"] = {
        "evaluated": True,
        "runnable": not viol,
        "violations": viol,
        "checklist": {
            "C3_boiling_point_margin": {
                "satisfied": not any(v.startswith("C3") for v in viol),
                "temperature_C": design["S6_temperature_C"],
                "max_allowed_temperature_C": round(bp - 10.0, 1),
                "bridging_liquid_bp_C": bp,
                "margin_C": round(bp - 10.0 - float(design["S6_temperature_C"]), 1),
            },
            "C4_just_suspended_speed": {
                "satisfied": not any(v.startswith("C4") for v in viol),
                "tip_speed_m_s": design["S5_tip_speed_m_s"],
                "minimum_required_m_s": round(just_suspended_tip_speed(design), 4),
                "margin_m_s": round(float(design["S5_tip_speed_m_s"])
                                    - just_suspended_tip_speed(design), 4),
            },
            "C5_total_solvent": {
                "satisfied": not any(v.startswith("C5") for v in viol),
                "total_solvent_L_per_kg": round(total_solvent_L_per_kg(design), 4),
                "limit_L_per_kg": 26.0,
                "margin_L_per_kg": round(26.0 - total_solvent_L_per_kg(design), 4),
            },
            "C6_total_process_time": {
                "satisfied": not any(v.startswith("C6") for v in viol),
                "total_hours": round(total_hours(design), 4),
                "limit_hours": 34.0,
                "margin_hours": round(34.0 - total_hours(design), 4),
            },
            "C7_scheme_self_consistency": {
                "satisfied": not any(v.startswith("C7") for v in viol),
                "classification": design["S10_classification"],
            },
        },
        "note": "C2, the controlled-change budget, is reported by change_budget_audit. "
                "No quality attribute is computed here.",
    }
    return result


def tool_change_budget_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """How many change points does this process spend against the registered process?

    Pure arithmetic on the cost table printed in the task description. Returned because a model that
    drifts over budget without noticing burns every remaining round on batches that cannot score --
    a bookkeeping failure, not a process-design failure.
    """
    raw, err = _require_process(arguments)
    if err:
        return err
    try:
        design = normalise_design(raw)
    except DesignError as exc:
        return {
            "error": "not_a_design_space_member",
            "detail": str(exc),
            "hint": "call check_process_legality to see which variable is wrong.",
            "recoverable": True,
        }

    moved = changed_slots(design)
    spent = edit_cost(design)
    return {
        "tool": "change_budget_audit",
        "change_points_spent": spent,
        "change_points_allowed": BUDGET,
        "change_points_remaining": BUDGET - spent,
        "within_budget": spent <= BUDGET,
        "cost_if_all_twelve_changed": TOTAL_EDIT_COST,
        "surviving_overlap_with_registered_process": round(change_overlap(design), 6),
        "minimum_required_overlap": round(1.0 - BUDGET / float(TOTAL_EDIT_COST), 6),
        "variables_changed": moved,
        "n_variables_changed": len(moved),
        "per_variable": [
            {"variable": s, "registered": START[s], "proposed": design[s],
             "changed": design[s] != START[s],
             "change_cost": EDIT_COST[s] if design[s] != START[s] else 0}
            for s in SLOTS
        ],
        "note": "C2. Over-budget batches are still measured and reported back, but they score zero.",
    }


def tool_compare_processes(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Which variables differ between two processes, and what would moving between them cost?

    Useful for "how far is my round-3 idea from my round-1 batch" without re-deriving the whole cost
    table by hand. Reports overlap and cost only -- never a quality comparison.
    """
    a, err = _require_process(arguments, "process_a")
    if err:
        return err
    b, err = _require_process(arguments, "process_b")
    if err:
        return err
    try:
        da, db = normalise_design(a), normalise_design(b)
    except DesignError as exc:
        return {"error": "not_a_design_space_member", "detail": str(exc),
                "hint": "both process_a and process_b must be on-grid; "
                        "call check_process_legality on each.",
                "recoverable": True}

    diff = [s for s in SLOTS if da[s] != db[s]]
    return {
        "tool": "compare_processes",
        "variables_differing": diff,
        "n_variables_differing": len(diff),
        "differences": [{"variable": s, "process_a": da[s], "process_b": db[s],
                         "change_cost": EDIT_COST[s]} for s in diff],
        "weighted_cost_of_moving_a_to_b": sum(EDIT_COST[s] for s in diff),
        "parameter_overlap_fraction_unweighted": round(1.0 - len(diff) / float(len(SLOTS)), 6),
        "parameter_overlap_fraction_cost_weighted":
            round(1.0 - sum(EDIT_COST[s] for s in diff) / float(TOTAL_EDIT_COST), 6),
        "change_points_of_a_vs_registered": edit_cost(da),
        "change_points_of_b_vs_registered": edit_cost(db),
        "note": "No quality attribute of either process is computed here.",
    }


def tool_variable_levels(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """The declared levels, unit and change cost of one process variable.

    Everything here is printed in the task description; the tool exists so a lookup slip is a
    recoverable tool error instead of an off-grid submission that wastes a graded round.
    """
    if "variable" not in arguments:
        return {"error": "missing_argument",
                "detail": "this tool requires the argument 'variable' (a variable name such as "
                          "'S5_tip_speed_m_s').",
                "required_arguments": ["variable"],
                "valid_variables": list(SLOTS),
                "recoverable": True}
    name = arguments["variable"]
    if not isinstance(name, str) or name not in LEVELS:
        return {"error": "unknown_variable",
                "detail": f"{name!r} is not one of the twelve declared process variables.",
                "valid_variables": list(SLOTS),
                "recoverable": True}
    return {
        "tool": "variable_levels",
        "variable": name,
        "unit": SLOT_UNITS[name],
        "allowed_values": LEVELS[name],
        "change_cost": EDIT_COST[name],
        "registered_value": START[name],
        "note": "Values not on this list are not members of the design space.",
    }


TOOL_IMPLEMENTATIONS = {
    "check_process_legality": tool_check_process_legality,
    "change_budget_audit": tool_change_budget_audit,
    "compare_processes": tool_compare_processes,
    "variable_levels": tool_variable_levels,
}

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "check_process_legality",
        "purpose": "Check whether a candidate process is a member of the design space, and whether "
                   "it can be run at all. Returns per-variable membership plus the C3-C7 "
                   "hard-constraint checklist with numeric margins.",
        "arguments": {"process": "object with the twelve process variables"},
        "returns": "design_space_membership (per variable), hard_constraints.checklist "
                   "(C3 boiling-point margin, C4 just-suspended speed, C5 total solvent, "
                   "C6 total time, C7 scheme consistency), each with its margin",
        "does_not_return": "any quality attribute, any prediction, the desirability D, or a score",
    },
    {
        "name": "change_budget_audit",
        "purpose": "Bookkeeping of the controlled-change budget for a candidate process against the "
                   "registered process.",
        "arguments": {"process": "object with the twelve process variables"},
        "returns": "change points spent / allowed / remaining, whether it is within budget, the "
                   "surviving overlap with the registered process, which variables moved, and the "
                   "per-variable cost breakdown",
        "does_not_return": "any quality attribute, any prediction, the desirability D, or a score",
    },
    {
        "name": "compare_processes",
        "purpose": "Compare two candidate processes: which variables differ, the weighted cost of "
                   "moving from one to the other, and their parameter overlap.",
        "arguments": {"process_a": "object with the twelve process variables",
                      "process_b": "object with the twelve process variables"},
        "returns": "variables_differing, per-difference change costs, "
                   "weighted_cost_of_moving_a_to_b, parameter overlap fractions",
        "does_not_return": "any quality attribute of either process",
    },
    {
        "name": "variable_levels",
        "purpose": "Look up the declared levels, unit and change cost of one process variable.",
        "arguments": {"variable": "one of the twelve variable names"},
        "returns": "allowed_values, unit, change_cost, registered_value",
        "does_not_return": "any quality attribute or any hint about which level is better",
    },
]


# =================================================================================================
# dispatch, quota and the leak guard
# =================================================================================================
def assert_no_objective_leak(result: Dict[str, Any]) -> None:
    """Refuse to hand back a tool result that mentions a graded quantity.

    A guard, not documentation. If a tool is ever extended carelessly this raises during the run and
    during scripts/check_consistency.py, instead of quietly making the benchmark trivial.
    """
    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in FORBIDDEN_RESULT_KEYS:
                    raise RuntimeError(
                        f"tool result leaks a graded quantity at {path}.{k}: tools must never return "
                        f"a response, a prediction, the desirability or a score")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(result, "result")


def tool_catalogue_text() -> str:
    """The catalogue as it is printed into the prompt. Generated from TOOL_SPECS, never hand-written,
    so the prompt cannot advertise a tool that does not exist or hide one that does."""
    lines = []
    for spec in TOOL_SPECS:
        args = ", ".join(f'"{k}": <{v}>' for k, v in spec["arguments"].items())
        lines.append(f'  - {spec["name"]}')
        lines.append(f'      purpose : {spec["purpose"]}')
        lines.append(f'      call    : {{"tool": "{spec["name"]}", "arguments": {{{args}}}}}')
        lines.append(f'      returns : {spec["returns"]}')
        lines.append(f'      never   : {spec["does_not_return"]}')
    return "\n".join(lines)


def run_tool_calls(calls: Any, calls_already_used_this_round: int = 0,
                   quota_per_turn: int = MAX_TOOL_CALLS_PER_TURN) -> Dict[str, Any]:
    """Execute one turn's worth of tool calls.

    Returns {"results": [...], "n_ok": int, "n_error": int, "n_executed": int}. Malformed input is a
    tool error, never an exception: the model is expected to read the error and fix the call.
    """
    if not isinstance(calls, list):
        return {"results": [{"error": "invalid_tool_calls",
                             "detail": "'tool_calls' must be a JSON array of "
                                       "{\"tool\": ..., \"arguments\": {...}} objects.",
                             "recoverable": True}],
                "n_ok": 0, "n_error": 1, "n_executed": 0}
    if not calls:
        return {"results": [{"error": "empty_tool_calls",
                             "detail": "'tool_calls' was an empty array. Either request at least "
                                       "one tool call or submit a candidate process.",
                             "recoverable": True}],
                "n_ok": 0, "n_error": 1, "n_executed": 0}

    results: List[Dict[str, Any]] = []
    n_ok = n_err = n_exec = 0
    used = calls_already_used_this_round
    for i, call in enumerate(calls, start=1):
        if used >= quota_per_turn:
            results.append({
                "call_index": i,
                "error": "quota_exceeded",
                "detail": f"at most {quota_per_turn} tool calls may be executed in one turn; this "
                          f"call and any after it were not run.",
                "calls_executed_this_turn": used,
                "recoverable": True,
            })
            n_err += 1
            continue
        if not isinstance(call, dict):
            results.append({"call_index": i, "error": "invalid_tool_call",
                            "detail": f"each entry of 'tool_calls' must be an object, got "
                                      f"{type(call).__name__}.",
                            "recoverable": True})
            n_err += 1
            used += 1
            continue
        name = call.get("tool") or call.get("name")
        args = call.get("arguments", call.get("args", {}))
        if name not in TOOL_IMPLEMENTATIONS:
            results.append({"call_index": i, "error": "unknown_tool",
                            "detail": f"{name!r} is not an available tool.",
                            "available_tools": sorted(TOOL_IMPLEMENTATIONS),
                            "recoverable": True})
            n_err += 1
            used += 1
            continue
        if not isinstance(args, dict):
            results.append({"call_index": i, "tool": name, "error": "invalid_arguments",
                            "detail": "'arguments' must be a JSON object.",
                            "recoverable": True})
            n_err += 1
            used += 1
            continue
        try:
            out = TOOL_IMPLEMENTATIONS[name](args)
        except Exception as exc:                                  # never kill the run on a tool bug
            out = {"error": "tool_internal_error", "detail": f"{type(exc).__name__}: {exc}",
                   "recoverable": False}
        assert_no_objective_leak(out)
        out.setdefault("tool", name)          # so an error result is still attributable to a tool
        out = {"call_index": i, **out}
        results.append(out)
        n_exec += 1
        used += 1
        if "error" in out:
            n_err += 1
        else:
            n_ok += 1
    return {"results": results, "n_ok": n_ok, "n_error": n_err, "n_executed": n_exec}
