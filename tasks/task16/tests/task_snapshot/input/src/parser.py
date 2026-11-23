from __future__ import annotations

import json
import re
from typing import Any, Dict


class ParseError(Exception):
    pass


def extract_json(text: str, allow_markdown_fence: bool = True) -> Dict[str, Any]:
    """Extract and parse the JSON object from an LLM response."""
    if not isinstance(text, str):
        raise ParseError("LLM response is not a string.")

    cleaned = text.strip()

    if allow_markdown_fence:
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ParseError(f"Failed to parse JSON object: {exc}") from exc
        raise ParseError("No JSON object found in LLM response.")


def parse_turn(response_text: str, allow_markdown_fence: bool = True) -> Dict[str, Any]:
    """One assistant turn is EITHER a tool request or a submission.

        {"tool_calls": [{"tool": "...", "arguments": {...}}, ...]}      -> {"kind": "tool_calls"}
        {"candidate":  {"name": ..., "rationale": ..., "formulation": {...}}}
                                                                       -> {"kind": "candidate"}

    Which one it is is the model's decision, and the harness does not nudge it (audit finding #5).
    A turn carrying both is treated as a submission, because the submission is the committing act.
    A turn carrying neither is a ParseError, exactly as before.
    """
    data = extract_json(response_text, allow_markdown_fence=allow_markdown_fence)
    if "candidate" in data:
        return {"kind": "candidate",
                "candidate": _parse_candidate_object(data),
                "also_requested_tools": "tool_calls" in data}
    if "tool_calls" in data:
        calls = data["tool_calls"]
        if not isinstance(calls, list):
            raise ParseError("tool_calls must be a JSON array of {tool, arguments} objects.")
        return {"kind": "tool_calls", "tool_calls": calls}
    raise ParseError("Response must contain either `candidate` or `tool_calls`.")


def parse_candidate(response_text: str, allow_markdown_fence: bool = True) -> Dict[str, Any]:
    """Expected shape:

        {"candidate": {"name": str, "rationale": str,
                       "formulation": {"T1_mcc_grade": ..., ..., "T12_man_pct": ...}}}

    Slot values are NOT validated here -- src/formulation.normalise_design does that, so that an
    off-grid value is recorded as an evaluated-and-rejected design rather than as a parse failure.
    """
    return _parse_candidate_object(extract_json(response_text,
                                                allow_markdown_fence=allow_markdown_fence))


def _parse_candidate_object(data: Dict[str, Any]) -> Dict[str, Any]:
    if "candidate" not in data or not isinstance(data["candidate"], dict):
        raise ParseError("Missing top-level object: candidate.")

    candidate = data["candidate"]

    for key in ("name", "rationale"):
        if key not in candidate:
            raise ParseError(f"Missing candidate field: {key}.")
        if not isinstance(candidate[key], str):
            raise ParseError(f"Candidate field must be a string: {key}.")

    if "formulation" not in candidate or not isinstance(candidate["formulation"], dict):
        raise ParseError("Missing candidate field: formulation (must be a JSON object of slots).")

    return {
        "name": candidate["name"].strip(),
        "rationale": candidate["rationale"].strip(),
        "formulation": candidate["formulation"],
    }
