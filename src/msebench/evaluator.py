"""Lightweight evaluation helpers (TextQA matching + result persistence)."""
from __future__ import annotations

import json
import re
from pathlib import Path

_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize(text) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower().translate(_SUBSCRIPT).translate(_SUPERSCRIPT)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.%+-]", "", text)
    return text.rstrip(".")


def _numword_to_digit(text: str) -> str:
    return " ".join(_NUMWORDS.get(tok, tok) for tok in text.split())


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for key in ("answer", "final_answer", "result"):
        m = re.search(r'["\']?\b%s\b["\']?\s*[:：]\s*["\']?(.+?)["\']?$' % key, text, re.I | re.S)
        if m:
            return m.group(1).strip().strip('"\'')
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if m:
        return m.group(1).strip()
    return text


def textqa_match(prediction: str, gold: str) -> bool:
    pred = normalize(extract_final_answer(prediction))
    gold = normalize(gold)
    if not pred or not gold:
        return False
    if pred == gold or pred in gold or gold in pred:
        return True
    pred_n = _numword_to_digit(pred)
    gold_n = _numword_to_digit(gold)
    return pred_n == gold_n or pred_n in gold_n or gold_n in pred_n


def append_result(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
