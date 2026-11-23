from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from scoring import score_rules
from utils import write_json


CRITERIA = {
    "continuum_solvation_scope": 2,
    "orbital_gap_scope": 2,
    "conformational_sampling_scope": 2,
    "geometry_specific_sapt_scope": 2,
    "quantitative_next_calculation": 2,
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_llm_eval_prompt(workspace: Path, result: Dict[str, Any], rule_eval: Dict[str, Any]) -> str:
    verified = {
        "formal_recompute": rule_eval.get("details", {}).get("formal_recompute"),
        "endpoint_geometry": rule_eval.get("details", {}).get("endpoint_geometry"),
        "sapt_recompute": rule_eval.get("details", {}).get("sapt_recompute"),
        "anomaly_audit": rule_eval.get("details", {}).get("anomaly_audit"),
        "fabrication_signals": rule_eval.get("details", {}).get("fabrication_signals"),
    }
    stage_data = {
        "stage_A_audit": _load_json(workspace / "stage_A_audit.json", None),
        "stage_B_bulk": _load_json(workspace / "stage_B_bulk.json", None),
        "stage_C_local": _load_json(workspace / "stage_C_local.json", None),
    }
    return (
        "You are the independent scientific judge for an evidence-audited computational chemistry benchmark.\n"
        "Use only the verified evidence bundle, stage summaries, and final result below. Do not infer correctness from confident prose.\n"
        "Return JSON only, without Markdown.\n\n"
        "Score exactly 10 points using this schema:\n"
        "{\n"
        '  "score": 0,\n'
        '  "max_score": 10,\n'
        '  "criteria": {\n'
        '    "continuum_solvation_scope": {"score": 0, "max_score": 2, "evidence": [], "comment": ""},\n'
        '    "orbital_gap_scope": {"score": 0, "max_score": 2, "evidence": [], "comment": ""},\n'
        '    "conformational_sampling_scope": {"score": 0, "max_score": 2, "evidence": [], "comment": ""},\n'
        '    "geometry_specific_sapt_scope": {"score": 0, "max_score": 2, "evidence": [], "comment": ""},\n'
        '    "quantitative_next_calculation": {"score": 0, "max_score": 2, "evidence": [], "comment": ""}\n'
        "  },\n"
        '  "hard_failure_suspected": false,\n'
        '  "hard_failure_reason": null,\n'
        '  "overall_comment": ""\n'
        "}\n\n"
        "Judge whether the answer correctly limits fixed-geometry continuum solvation, explains the orbital-energy difference and the need for an excitation calculation, "
        "distinguishes selected local minima from populations, treats SAPT as geometry-specific rather than a solution-level motif, and proposes a specific next calculation with a quantitative decision metric and threshold. "
        "Unsupported generalizations receive zero for the affected criterion.\n\n"
        f"Verified evidence bundle:\n{json.dumps(verified, ensure_ascii=False, indent=2)[:30000]}\n\n"
        f"Final result.json:\n{json.dumps(result, ensure_ascii=False, indent=2)[:30000]}\n\n"
        f"Stage summaries:\n{json.dumps(stage_data, ensure_ascii=False, indent=2)[:30000]}\n"
    )


def _validate(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(data.get("score"), (int, float)):
        errors.append("missing numeric score")
    criteria = data.get("criteria")
    if not isinstance(criteria, dict):
        return errors + ["missing criteria"]
    total = 0.0
    for key, maximum in CRITERIA.items():
        item = criteria.get(key)
        if not isinstance(item, dict):
            errors.append(f"missing criterion {key}")
            continue
        score = item.get("score")
        if not isinstance(score, (int, float)) or score < 0 or score > maximum:
            errors.append(f"invalid score for {key}")
        else:
            total += float(score)
        if not isinstance(item.get("evidence"), list):
            errors.append(f"missing evidence list for {key}")
    if isinstance(data.get("score"), (int, float)) and abs(float(data["score"]) - total) > 1e-6:
        errors.append("top-level score does not equal criterion sum")
    return errors


def call_independent_evaluator(prompt: str, api_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"), timeout=api_cfg.get("timeout", 300))
    max_retries = int(api_cfg.get("max_retries", 2))
    delay = float(api_cfg.get("retry_delay_seconds", 2))
    last_error: str | None = None
    raw: Dict[str, Any] = {}
    for attempt in range(1, max_retries + 2):
        try:
            request: Dict[str, Any] = {
                "model": api_cfg["model"],
                "messages": [
                    {"role": "system", "content": "You are a strict independent scientific benchmark judge. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": api_cfg.get("temperature", 0),
                "max_tokens": api_cfg.get("max_tokens", 6000),
            }
            for optional_key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
                if optional_key in api_cfg:
                    request[optional_key] = api_cfg[optional_key]
            response = client.chat.completions.create(
                **request,
            )
            raw = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else response
            content = response.choices[0].message.content or ""
            parsed = _extract_json(content)
            errors = _validate(parsed)
            if errors:
                raise ValueError("; ".join(errors))
            parsed["attempts"] = attempt
            return {"ok": True, "parsed": parsed, "raw": raw, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt <= max_retries:
                time.sleep(delay)
    return {
        "ok": False,
        "parsed": {"score": 0, "max_score": 10, "criteria": {}, "overall_comment": "Independent evaluation failed."},
        "raw": raw,
        "error": last_error,
    }


def combine_scores(rule_eval: Dict[str, Any], llm_eval: Dict[str, Any]) -> Dict[str, Any]:
    objective = float(rule_eval.get("score", 0))
    scientific = float((llm_eval.get("parsed") or {}).get("score", 0))
    raw_total = objective + scientific
    caps = [x.get("cap") for x in rule_eval.get("hard_caps", []) if isinstance(x.get("cap"), (int, float))]
    final_cap = min(caps) if caps else 100
    return {
        "score": min(raw_total, final_cap),
        "max_score": 100,
        "raw_total_before_caps": raw_total,
        "final_cap": final_cap,
        "deterministic_score": objective,
        "independent_scientific_score": scientific,
    }


def evaluate_run(
    run_dir: Path,
    workspace: Path,
    result: Dict[str, Any],
    *,
    evaluator_api_cfg: Dict[str, Any] | None = None,
    candidate_model: str = "",
    instance_truth: Dict[str, Any] | None = None,
    task_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rule_eval = score_rules(run_dir, workspace, result, instance_truth=instance_truth, task_cfg=task_cfg)
    write_json(run_dir / "rule_evaluation.json", rule_eval)

    evaluator_model = "deterministic-hard-v2"
    independent = False
    llm_eval = {"ok":True,"parsed":{"score":0,"max_score":0,"criteria":{},"overall_comment":"LLM judge removed; all 100 points are deterministic."},"raw":{},"error":None}
    write_json(run_dir / "llm_evaluation.json", llm_eval["parsed"])
    combined = {"score":float(rule_eval["score"]),"max_score":100,"raw_total_before_caps":float(rule_eval["raw_score_before_tool_call_penalty"]),"final_cap":100,"deterministic_score":float(rule_eval["score"]),"independent_scientific_score":0.0}
    return {
        "passed": combined["score"] >= 60,
        "combined": combined,
        "rule_evaluation": rule_eval,
        "independent_evaluator": {
            "candidate_model": candidate_model,
            "evaluator_model": evaluator_model or None,
            "independence_satisfied": independent,
            "ok": llm_eval.get("ok", False),
            "error": llm_eval.get("error"),
            "evaluation": llm_eval.get("parsed"),
        },
    }


def evaluate_result(result: Dict[str, Any], workspace: Path | None = None) -> Dict[str, Any]:
    workspace = workspace or Path.cwd()
    run_dir = workspace.parent if workspace.name == "workspace" else workspace
    return evaluate_run(run_dir, workspace, result)
