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
    """Classify one model turn: a tool request, or a committed candidate.

    The model chooses which of the two it emits, so the harness must not assume. Returns
    {"kind": "tool_calls", "tool_calls": [...]} or {"kind": "candidate", "candidate": {...}}.

    A turn carrying BOTH is treated as a candidate: committing wins over asking, so a model cannot
    quietly get a free extra measurement out of a tool turn.
    """
    data = extract_json(response_text, allow_markdown_fence=allow_markdown_fence)
    if isinstance(data, dict) and isinstance(data.get("candidate"), dict):
        return {"kind": "candidate", "candidate": parse_candidate_object(data["candidate"])}
    if isinstance(data, dict) and "tool_calls" in data:
        return {"kind": "tool_calls", "tool_calls": data["tool_calls"]}
    raise ParseError("The turn contained neither a 'candidate' object nor a 'tool_calls' array.")


def parse_candidate_object(candidate: Any) -> Dict[str, Any]:
    """Validate the candidate object itself (shared by parse_candidate and parse_turn)."""
    if not isinstance(candidate, dict):
        raise ParseError("Missing top-level object: candidate.")

    for key in ("name", "rationale"):
        if key not in candidate:
            raise ParseError(f"Missing candidate field: {key}.")
        if not isinstance(candidate[key], str):
            raise ParseError(f"Candidate field must be a string: {key}.")

    if "process" not in candidate or not isinstance(candidate["process"], dict):
        raise ParseError("Missing candidate field: process (must be a JSON object of the twelve "
                         "process variables).")

    return {
        "name": candidate["name"].strip(),
        "rationale": candidate["rationale"].strip(),
        "process": candidate["process"],
    }


def parse_candidate(response_text: str, allow_markdown_fence: bool = True) -> Dict[str, Any]:
    """Expected shape:

        {"candidate": {"name": str, "rationale": str,
                       "process": {"S1_bridging_liquid": ..., ..., "S12_HPMC_ppm": ...}}}

    Variable values are NOT validated here -- src/agglomeration.normalise_design does that, so that
    an off-grid value is recorded as an evaluated-and-rejected design rather than as a parse failure.
    """
    data = extract_json(response_text, allow_markdown_fence=allow_markdown_fence)

    if "candidate" not in data or not isinstance(data["candidate"], dict):
        raise ParseError("Missing top-level object: candidate.")

    return parse_candidate_object(data["candidate"])
