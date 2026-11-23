from __future__ import annotations

import json
import re
from typing import Any, Dict


class ParseError(Exception):
    pass


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ParseError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ParseError("JSON object appears incomplete.")


def parse_candidate(raw_response: str, allow_markdown_fence: bool = True) -> Dict[str, Any]:
    if not raw_response or not raw_response.strip():
        raise ParseError("Empty LLM response.")

    text = raw_response.strip()
    if allow_markdown_fence:
        text = _strip_markdown_fence(text)

    json_text = _extract_first_json_object(text)

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(obj, dict) or "candidate" not in obj:
        raise ParseError("Top-level JSON must contain key `candidate`.")

    cand = obj["candidate"]
    if not isinstance(cand, dict):
        raise ParseError("`candidate` must be an object.")

    required = [
        "candidate_id",
        "parent_candidate_id",
        "proposal_mode",
        "smiles",
        "structural_modification",
        "modification_reasoning",
    ]
    missing = [k for k in required if k not in cand]
    if missing:
        raise ParseError(f"Missing candidate fields: {missing}")

    if cand["proposal_mode"] not in {"direct_smiles", "rdkit_edit_plan"}:
        raise ParseError("candidate.proposal_mode must be `direct_smiles` or `rdkit_edit_plan`.")

    if not isinstance(cand["smiles"], str) or not cand["smiles"].strip():
        raise ParseError("candidate.smiles must be a non-empty string.")

    return cand
