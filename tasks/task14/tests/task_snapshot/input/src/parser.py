"""Parse the single JSON object a round's response must contain.

Every failure mode raises `ParseError` with a human-readable reason. `main.py` turns that into a
lost round -- never into a crash and never into an early exit; `tests/test_harness_robustness.py`
pins that behaviour.
"""
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
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort extraction of outermost JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ParseError(f"Failed to parse JSON object: {exc}") from exc
        raise ParseError("No JSON object found in LLM response.")


def parse_candidate(response_text: str, allow_markdown_fence: bool = True) -> Dict[str, str]:
    data = extract_json(response_text, allow_markdown_fence=allow_markdown_fence)

    if "candidate" not in data or not isinstance(data["candidate"], dict):
        raise ParseError("Missing top-level object: candidate.")

    candidate = data["candidate"]

    for key in ["name", "smiles", "rationale"]:
        if key not in candidate:
            raise ParseError(f"Missing candidate field: {key}.")
        if not isinstance(candidate[key], str):
            raise ParseError(f"Candidate field must be a string: {key}.")

    return {
        "name": candidate["name"].strip(),
        "smiles": candidate["smiles"].strip(),
        "rationale": candidate["rationale"].strip(),
    }
