"""Neutral, deterministic, segment-aware parsing of xTB text logs.

The parser intentionally reports observations rather than declaring that a
scientific task succeeded.  It supports terse synthetic records and the
sectioned output emitted by current xTB releases.  Every inferred value has a
line-bound observation in ``facts`` and ambiguous observations are surfaced in
``conflicts`` instead of being silently resolved.
"""

from __future__ import annotations

import re
from typing import Any

from .file_tools import ToolInputError, WorkspaceSandbox, sha256_bytes, sha256_file


UNKNOWN = "UNKNOWN"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"

_COMMAND_SEGMENT_START = re.compile(
    r"(?ix)^\s*(?:"
    r"(?:\$|>)\s*xtb(?:\.exe)?(?:\s|$)|"
    r"xtb(?:\.exe)?\s+\S+\.(?:xyz|coord)(?:\s|$)"
    r")"
)

_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "OPTIMIZATION_CONVERGED",
        re.compile(r"(?i)geometry\s+optimization\s+converged|optimization\s+converged"),
    ),
    (
        "OPTIMIZATION_MAX_CYCLES",
        re.compile(
            r"(?i)(?:maximum|max\.?)[ -]*(?:optimization[ -]*)?(?:cycles|iterations)"
            r".*(?:reach|exceed)|optimization.*failed\s+to\s+converge|"
            r"failed\s+to\s+converge\s+geometry\s+optimization|"
            r"did\s+not\s+converge.*(?:cycle|iteration)"
        ),
    ),
    (
        "SCC_NON_CONVERGENCE",
        re.compile(
            r"(?i)(?:scc|scf).*(?:not|failed\s+to|did\s+not)\s+converg|"
            r"electronic.*non[ -]?convergen|"
            r"convergence\s+criteria\s+cannot\s+be\s+satisfied\s+within\s+\d+\s+iterations"
        ),
    ),
    (
        "SCC_CONVERGED",
        re.compile(r"(?i)\*{3}\s*convergence\s+criteria\s+satisfied\s+after\s+\d+\s+iterations"),
    ),
    (
        "HESSIAN_STARTED",
        re.compile(
            r"(?i)\b(?:hessian|vibrational\s+analysis|frequency\s+calculation)\b"
            r".*(?:start|calculation|evaluation)|--hess\b"
        ),
    ),
    (
        "HESSIAN_COMPLETED",
        re.compile(
            r"(?i)(?:hessian|vibrational\s+analysis|frequency\s+calculation)"
            r".*(?:complete|completed|finished|done)"
        ),
    ),
    (
        "HESSIAN_OUTPUT_OBSERVED",
        re.compile(r"(?i)\bvibrational\s+frequencies\b|\bfrequency\s+printout\b|\bnormal\s+modes\b"),
    ),
    (
        "NORMAL_TERMINATION",
        re.compile(
            r"(?i)(?<!ab)normal\s+termination\s+of\s+xtb|"
            r"xtb.*finished\s+successfully|^\s*\*\s*finished\s+run\s+on\b"
        ),
    ),
    (
        "TIMEOUT",
        re.compile(r"(?i)time(?:d)?\s*out|walltime.*(?:reached|exceeded)|time\s+limit"),
    ),
    (
        "OUT_OF_MEMORY",
        re.compile(r"(?i)out\s+of\s+memory|cannot\s+allocate\s+memory|memory\s+allocation.*fail"),
    ),
    ("SEGMENTATION_FAULT", re.compile(r"(?i)segmentation\s+(?:fault|violation)|sigsegv")),
    (
        "UNSUPPORTED_OPTION",
        re.compile(r"(?i)(?:unknown|unsupported|invalid|unrecognized)\s+(?:option|argument|flag)"),
    ),
    (
        "MISSING_PARAMETER",
        re.compile(
            r"(?i)(?:missing|required|expected).*(?:parameter|argument|value)|"
            r"(?:parameter|argument).*(?:missing|required)"
        ),
    ),
    (
        "CHARGE_SPIN_INCONSISTENCY",
        re.compile(
            r"(?i)(?:charge|electron).*(?:multiplicity|spin|uhf).*(?:inconsistent|invalid|impossible)|"
            r"(?:multiplicity|spin|uhf).*(?:charge|electron).*(?:inconsistent|invalid|impossible)"
        ),
    ),
    (
        "COORDINATE_FORMAT_ERROR",
        re.compile(
            r"(?i)(?:coordinate|geometry|xyz).*(?:format|parse|read).*(?:error|invalid|fail)|"
            r"(?:invalid|malformed).*(?:coordinate|geometry|xyz)|"
            r"could\s+not\s+parse\s+coordinates?\s+from\s+(?:an?\s+)?xyz"
        ),
    ),
    ("ABNORMAL_TERMINATION", re.compile(r"(?i)abnormal\s+termination|fatal\s+error|terminated\s+abnormally")),
)

_WARNING = re.compile(r"(?i)(?:^|\s)(?:warning|warn)\s*[:!]|\bWARNING\b")
_ERROR = re.compile(r"(?i)(?:^|\s)(?:fatal\s+)?error\s*[:!]|\bERROR\b|failed\s+to")
_OPT_PHASE = re.compile(r"(?i)--opt\b|geometry\s+optimization|optimization\s+cycle|\bANCOPT\b")
_HESS_PHASE = re.compile(
    r"(?i)--hess\b|vibrational\s+frequencies|frequency\s+printout|normal\s+modes|"
    r"(?:hessian|vibrational\s+analysis|frequency\s+calculation).*(?:start|calculation|evaluation)"
)
_VERSION = re.compile(r"(?i)\bxtb\s+version\s+([A-Za-z0-9][A-Za-z0-9._+\-]*)")
_PROGRAM_CALL = re.compile(r"(?i)(?:program\s+call\s*[:=]|^\s*(?:\$|>)\s*)\s*(xtb(?:\.exe)?\b.*)$")
_METHOD_SETUP = re.compile(r"(?i)\bHamiltonian\s+(GFN\s*[-]?\s*[012]\s*[-]?\s*xTB|GFN[- ]?FF)\b")
_METHOD_ARG = re.compile(r"(?i)(?:^|\s)--gfn\s+([012])(?:\s|$)")
_CHARGE_SETUP = re.compile(r"(?i)\bnet\s+charge\s*[:=]?\s*(-?\d+)\b")
_CHARGE_ARG = re.compile(r"(?i)(?:^|\s)--chrg\s+(-?\d+)(?:\s|$)")
_UHF_SETUP = re.compile(r"(?i)\bunpaired\s+electrons\s*[:=]?\s*(\d+)\b")
_UHF_ARG = re.compile(r"(?i)(?:^|\s)--uhf\s+(\d+)(?:\s|$)")
_SOLVATION_ARG = re.compile(r"(?i)(?:^|\s)--(alpb|gbsa|cpcm|cosmo)\s+([^\s]+)")
_SOLVATION_FALSE = re.compile(r"(?i)\b(?:GBSA|ALPB|implicit)\s+solvation\s*[:=]?\s*(?:false|none|off|no)\b")
_SOLVATION_SETUP = re.compile(r"(?i)\b(GBSA|ALPB|CPCM|COSMO)\s+solvation\s*[:=]?\s*([^\s:|]+)")
_OPT_CYCLE = re.compile(r"(?i)(?:\bCYCLE\s+|optimization\s+cycle\s+)(\d+)\b")
_OPT_OUTCOME_CYCLE = re.compile(
    r"(?i)(?:optimization\s+converged\s+after|failed\s+to\s+converge\s+geometry\s+optimization\s+in)\s+(\d+)\s+iterations"
)
_SCC_ITERATION = re.compile(r"(?i)convergence\s+criteria\s+satisfied\s+after\s+(\d+)\s+iterations")
_TOTAL_ENERGY = re.compile(rf"(?i)\btotal\s+energy\s*(?::)?\s*({_NUMBER})\s*(Eh|eV)?")
_GRADIENT_NORM = re.compile(rf"(?i)\bgradient\s+norm\s*(?::)?\s*({_NUMBER})\s*([^\s|:]*)")


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _fact(code: str, line_number: int, text: str, **values: Any) -> dict[str, Any]:
    stripped = text.rstrip("\r\n")
    return {
        "code": code,
        "line_start": line_number,
        "line_end": line_number,
        "text": stripped[:500],
        "content_sha256": sha256_bytes((stripped + "\n").encode("utf-8")),
        **values,
    }


def _segment_ranges(lines: list[str]) -> list[tuple[int, int]]:
    if not lines:
        return []
    starts = [1]
    prior_terminated = False
    for number, line in enumerate(lines[1:], 2):
        starts_new_run = bool(
            _COMMAND_SEGMENT_START.search(line)
            or (prior_terminated and (_VERSION.search(line) or _PROGRAM_CALL.search(line)))
        )
        if starts_new_run and number - starts[-1] > 1:
            starts.append(number)
            prior_terminated = False
        if re.search(
            r"(?i)normal\s+termination\s+of\s+xtb|abnormal\s+termination|"
            r"^\s*\*\s*finished\s+run\s+on\b|segmentation\s+fault|"
            r"walltime.*(?:reached|exceeded)|time(?:d)?\s*out",
            line,
        ):
            prior_terminated = True
    return [
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else len(lines))
        for index, start in enumerate(starts)
    ]


def _normalized_method(raw: str) -> str:
    compact = re.sub(r"\s+", "", raw).upper().replace("_", "-")
    if compact in {"GFN0-XTB", "GFN1-XTB", "GFN2-XTB"}:
        return compact.replace("XTB", "xTB")
    if compact in {"GFNFF", "GFN-FF"}:
        return "GFN-FF"
    return raw.strip() or UNKNOWN


def _observation_conflict(
    code: str,
    observations: list[dict[str, Any]],
    *,
    value_key: str = "value",
) -> dict[str, Any] | None:
    distinct = []
    for observation in observations:
        value = observation.get(value_key)
        if value not in distinct:
            distinct.append(value)
    if len(distinct) <= 1:
        return None
    return {
        "code": code,
        "values": distinct,
        "line_ranges": [
            {"line_start": item["line_start"], "line_end": item["line_end"]}
            for item in observations
        ],
    }


def _last_value(observations: list[dict[str, Any]], default: Any = UNKNOWN) -> Any:
    return observations[-1]["value"] if observations else default


def _parse_segment(lines: list[str], start: int, end: int, index: int) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    phase_markers: set[str] = set()
    observations: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "version",
            "method",
            "charge",
            "uhf",
            "solvation",
            "optimization_cycle",
            "scc_iteration",
            "final_energy",
            "gradient_norm",
        )
    }
    state_history: list[dict[str, Any]] = []

    def observe(key: str, code: str, number: int, line: str, value: Any, **extra: Any) -> None:
        entry = _fact(code, number, line, value=value, **extra)
        observations[key].append(entry)
        facts.append(entry)

    def transition(state: str, number: int, code: str) -> None:
        if not state_history or state_history[-1]["state"] != state:
            state_history.append({"state": state, "line": number, "trigger": code})

    for number in range(start, end + 1):
        line = lines[number - 1]
        if _OPT_PHASE.search(line):
            phase_markers.add("geometry_optimization")
            transition("OPTIMIZATION_ACTIVE", number, "OPTIMIZATION_PHASE")
        if _HESS_PHASE.search(line):
            phase_markers.add("hessian")
            transition("HESSIAN_ACTIVE", number, "HESSIAN_PHASE")

        matched_codes: set[str] = set()
        for code, pattern in _MARKERS:
            if pattern.search(line):
                facts.append(_fact(code, number, line))
                matched_codes.add(code)
                transition(code, number, code)

        version = _VERSION.search(line)
        if version:
            observe("version", "XTB_VERSION", number, line, version.group(1))

        call = _PROGRAM_CALL.search(line)
        call_text = call.group(1) if call else None
        method_setup = _METHOD_SETUP.search(line)
        method_arg = _METHOD_ARG.search(call_text or "")
        if method_setup:
            observe("method", "METHOD", number, line, _normalized_method(method_setup.group(1)))
        elif method_arg:
            observe("method", "METHOD", number, line, f"GFN{method_arg.group(1)}-xTB")

        charge_setup = _CHARGE_SETUP.search(line)
        charge_arg = _CHARGE_ARG.search(call_text or "")
        if charge_setup:
            observe("charge", "CHARGE", number, line, int(charge_setup.group(1)))
        elif charge_arg:
            observe("charge", "CHARGE", number, line, int(charge_arg.group(1)))

        uhf_setup = _UHF_SETUP.search(line)
        uhf_arg = _UHF_ARG.search(call_text or "")
        if uhf_setup:
            observe("uhf", "UHF", number, line, int(uhf_setup.group(1)))
        elif uhf_arg:
            observe("uhf", "UHF", number, line, int(uhf_arg.group(1)))

        solvation_arg = _SOLVATION_ARG.search(call_text or "")
        solvation_setup = _SOLVATION_SETUP.search(line)
        if solvation_arg:
            observe(
                "solvation",
                "SOLVATION",
                number,
                line,
                f"{solvation_arg.group(1).upper()}:{solvation_arg.group(2)}",
            )
        elif _SOLVATION_FALSE.search(line):
            observe("solvation", "SOLVATION", number, line, "NONE")
        elif solvation_setup and solvation_setup.group(2).casefold() not in {"false", "none", "off", "no"}:
            observe(
                "solvation",
                "SOLVATION",
                number,
                line,
                f"{solvation_setup.group(1).upper()}:{solvation_setup.group(2)}",
            )

        cycle = _OPT_CYCLE.search(line) or _OPT_OUTCOME_CYCLE.search(line)
        if cycle:
            observe("optimization_cycle", "OPTIMIZATION_CYCLE", number, line, int(cycle.group(1)))
        scc_iteration = _SCC_ITERATION.search(line)
        if scc_iteration:
            observe("scc_iteration", "SCC_ITERATIONS", number, line, int(scc_iteration.group(1)))

        energy = _TOTAL_ENERGY.search(line)
        if energy:
            unit = energy.group(2) or UNKNOWN
            observe("final_energy", "TOTAL_ENERGY", number, line, _as_float(energy.group(1)), unit=unit)
        gradient = _GRADIENT_NORM.search(line)
        if gradient:
            unit = gradient.group(2).rstrip(":") or UNKNOWN
            observe("gradient_norm", "GRADIENT_NORM", number, line, _as_float(gradient.group(1)), unit=unit)

        if _WARNING.search(line):
            warnings.append(_fact("WARNING", number, line))
        if _ERROR.search(line) and not {"OPTIMIZATION_CONVERGED", "NORMAL_TERMINATION"}.intersection(matched_codes):
            errors.append(_fact("ERROR_TEXT", number, line))

    codes = {entry["code"] for entry in facts}
    phase = (
        next(iter(phase_markers))
        if len(phase_markers) == 1
        else "mixed_or_unknown" if phase_markers else "unknown"
    )
    optimization_converged = "OPTIMIZATION_CONVERGED" in codes
    optimization_failed = bool({"OPTIMIZATION_MAX_CYCLES", "SCC_NON_CONVERGENCE"}.intersection(codes))
    scc_converged = "SCC_CONVERGED" in codes
    scc_failed = "SCC_NON_CONVERGENCE" in codes
    normal_termination = "NORMAL_TERMINATION" in codes
    abnormal_termination = bool(
        {"ABNORMAL_TERMINATION", "TIMEOUT", "OUT_OF_MEMORY", "SEGMENTATION_FAULT"}.intersection(codes)
    )

    conflicts = [
        item
        for item in (
            _observation_conflict("VERSION_CONFLICT", observations["version"]),
            _observation_conflict("METHOD_CONFLICT", observations["method"]),
            _observation_conflict("CHARGE_CONFLICT", observations["charge"]),
            _observation_conflict("UHF_CONFLICT", observations["uhf"]),
            _observation_conflict("SOLVATION_CONFLICT", observations["solvation"]),
        )
        if item is not None
    ]
    if optimization_converged and optimization_failed:
        conflicts.append({"code": "OPTIMIZATION_OUTCOME_CONFLICT"})
    if scc_converged and scc_failed:
        conflicts.append({"code": "SCC_OUTCOME_CONFLICT"})
    if normal_termination and abnormal_termination:
        conflicts.append({"code": "TERMINATION_CONFLICT"})

    nonempty = [line for line in lines[start - 1 : end] if line.strip()]
    incomplete = bool(nonempty and not normal_termination and not abnormal_termination)
    if not nonempty:
        parser_state = "UNKNOWN"
    elif conflicts:
        parser_state = "CONFLICTING_EVIDENCE"
    elif abnormal_termination:
        parser_state = "ABNORMAL_TERMINATION"
    elif normal_termination:
        parser_state = "NORMAL_TERMINATION"
    elif incomplete:
        parser_state = "INCOMPLETE_OR_TRUNCATED"
    else:
        parser_state = "UNKNOWN"

    confidence = 0.30 if nonempty else 0.0
    confidence += 0.10 if observations["version"] else 0.0
    confidence += 0.10 if observations["method"] else 0.0
    confidence += 0.10 if phase_markers else 0.0
    confidence += 0.20 if normal_termination or abnormal_termination else 0.0
    confidence += 0.10 if optimization_converged or optimization_failed or scc_converged or scc_failed else 0.0
    confidence -= 0.20 if incomplete else 0.0
    confidence -= min(0.30, 0.10 * len(conflicts))
    confidence = round(max(0.0, min(1.0, confidence)), 2)
    confidence_label = "HIGH" if confidence >= 0.80 else "MEDIUM" if confidence >= 0.55 else "LOW"

    energy_observation = observations["final_energy"][-1] if observations["final_energy"] else None
    gradient_observation = observations["gradient_norm"][-1] if observations["gradient_norm"] else None
    final_energy = (
        {
            "value": energy_observation["value"],
            "unit": energy_observation["unit"],
            "line_start": energy_observation["line_start"],
            "line_end": energy_observation["line_end"],
        }
        if energy_observation
        else {"value": None, "unit": UNKNOWN, "line_start": None, "line_end": None}
    )
    gradient = (
        {
            "value": gradient_observation["value"],
            "unit": gradient_observation["unit"],
            "line_start": gradient_observation["line_start"],
            "line_end": gradient_observation["line_end"],
        }
        if gradient_observation
        else {"value": None, "unit": UNKNOWN, "line_start": None, "line_end": None}
    )

    return {
        "segment_index": index,
        "line_start": start,
        "line_end": end,
        "phase": phase,
        "phase_markers": sorted(phase_markers),
        "state": parser_state,
        "state_history": state_history,
        "facts": facts,
        "warnings": warnings,
        "errors": errors,
        "version": _last_value(observations["version"]),
        "xtb_version": _last_value(observations["version"]),
        "method": _last_value(observations["method"]),
        "charge": _last_value(observations["charge"]),
        "uhf": _last_value(observations["uhf"]),
        "solvation": _last_value(observations["solvation"]),
        "optimization_cycles": [item["value"] for item in observations["optimization_cycle"]],
        "optimization_cycle_count": _last_value(observations["optimization_cycle"]),
        "optimization_convergence": (
            "CONVERGED" if optimization_converged and not optimization_failed
            else "NOT_CONVERGED" if optimization_failed and not optimization_converged
            else "CONFLICT" if optimization_converged and optimization_failed
            else UNKNOWN
        ),
        "scc_convergence": (
            "CONVERGED" if scc_converged and not scc_failed
            else "NOT_CONVERGED" if scc_failed and not scc_converged
            else "CONFLICT" if scc_converged and scc_failed
            else UNKNOWN
        ),
        "scc_iterations": [item["value"] for item in observations["scc_iteration"]],
        "final_energy": final_energy,
        "final_energy_eh": (
            final_energy["value"] if str(final_energy["unit"]).casefold() == "eh" else None
        ),
        "gradient": gradient,
        "gradient_norm": gradient,
        "gradient_norm_eh_per_bohr": (
            gradient["value"]
            if str(gradient["unit"]).casefold() in {
                "eh/a0", "eh/bohr", "eh/a.u.", "eh/au", "eh/α", "eh/alpha"
            }
            else None
        ),
        "incomplete_or_truncated_sections": incomplete,
        "truncated": incomplete,
        "parser_confidence": confidence,
        "parser_confidence_label": confidence_label,
        "conflicts": conflicts,
        # Legacy booleans retained for runner/evaluator compatibility.
        "optimization_converged_observed": optimization_converged,
        "optimization_failure_observed": optimization_failed,
        "hessian_started_observed": "HESSIAN_STARTED" in codes,
        "hessian_completed_observed": "HESSIAN_COMPLETED" in codes,
        "hessian_output_observed": "HESSIAN_OUTPUT_OBSERVED" in codes,
        "normal_termination_observed": normal_termination,
        "abnormal_termination_observed": abnormal_termination,
    }


def parse_xtb_log_text(text: str, *, path: str = "<memory>") -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    lines = text.splitlines()
    ranges = _segment_ranges(lines)
    segments = [_parse_segment(lines, start, end, index) for index, (start, end) in enumerate(ranges)]
    latest = segments[-1] if segments else None
    return {
        "path": path,
        "line_count": len(lines),
        "segment_count": len(segments),
        "segments": segments,
        "latest_segment_index": latest["segment_index"] if latest else None,
        "latest_segment": latest,
        "version": latest["version"] if latest else UNKNOWN,
        "method": latest["method"] if latest else UNKNOWN,
        "charge": latest["charge"] if latest else UNKNOWN,
        "uhf": latest["uhf"] if latest else UNKNOWN,
        "solvation": latest["solvation"] if latest else UNKNOWN,
        "parser_confidence": latest["parser_confidence"] if latest else 0.0,
        "conflicts": latest["conflicts"] if latest else [],
        "neutrality_note": (
            "markers are reported per segment; scheduler exit and filenames are not "
            "treated as scientific completion"
        ),
    }


def parse_xtb_log(sandbox: WorkspaceSandbox, path: str) -> dict[str, Any]:
    resolved = sandbox.resolve(path)
    size = sandbox._assert_readable_text(resolved)
    if size > sandbox.max_file_bytes:
        raise ToolInputError("FILE_TOO_LARGE", "xTB log exceeds the parser size limit")
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolInputError("INVALID_UTF8", "xTB log is not valid UTF-8") from exc
    result = parse_xtb_log_text(text, path=sandbox.relative(resolved))
    result["file_size_bytes"] = size
    result["file_sha256"] = sha256_file(resolved)
    return result


__all__ = ["UNKNOWN", "parse_xtb_log", "parse_xtb_log_text"]
