from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evaluator import evaluate_candidate, CandidateEvaluation
from feedback import build_feedback
from llm_client import call_llm
from markdown_report import initialize_markdown_report, append_round_to_markdown_report, append_final_report_to_markdown
from parser import parse_candidate, ParseError
from scoring import final_score
from utils import append_jsonl, copy_file, load_json, load_yaml, make_run_dir, write_json, write_yaml, redact_api_config


def build_initial_messages(question_text: str, system_msg: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question_text + "\n\nNow propose the molecule for round 1."},
    ]


def compact_evaluation_for_model(evaluation: CandidateEvaluation) -> Dict[str, Any]:
    """Remove bulky trial-level internals from model-visible feedback only.

    The complete evaluator record remains unchanged in the archived run evidence.
    """
    ev = evaluation.to_dict()
    details = dict(ev.get("details") or {})
    dipro = dict(details.get("dipro") or {})
    packing = dict(dipro.get("packing_search") or {})
    packing.pop("trials", None)
    dipro["packing_search"] = packing
    details["dipro"] = dipro
    ev["details"] = details
    return ev


def build_next_user_message(
    *,
    round_idx: int,
    candidate: Optional[Dict[str, Any]],
    evaluation: CandidateEvaluation,
    feedback: Dict[str, Any],
    best_so_far: Optional[CandidateEvaluation],
) -> str:
    submitted = "Candidate could not be parsed." if candidate is None else json.dumps(candidate, ensure_ascii=False, indent=2)
    ev = compact_evaluation_for_model(evaluation)

    best_text = "None"
    if best_so_far is not None:
        best_text = json.dumps(compact_evaluation_for_model(best_so_far), ensure_ascii=False, indent=2)

    return f"""Round {round_idx} evaluation result:

Submitted candidate:
{submitted}

Evaluator result:
{json.dumps(ev, ensure_ascii=False, indent=2)}

Feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Best candidate so far:
{best_text}

Now propose the molecule for round {round_idx + 1}.
Return exactly one JSON object with the required format and no extra text.
Use the computed feedback to guide the next structural modification.
Focus on a neutral single-component five-ring fused conjugated semiconductor backbone.
"""


def update_best(best: Optional[CandidateEvaluation], ev: CandidateEvaluation) -> Optional[CandidateEvaluation]:
    if ev.marcus_hole_rate_s_inv is None:
        return best
    if best is None or best.marcus_hole_rate_s_inv is None:
        return ev
    if ev.marcus_hole_rate_s_inv > best.marcus_hole_rate_s_inv:
        return ev
    return best


def request_final_recommendation(evaluations, messages, llm_cfg, llm_debug_path):
    summary = [{
        "candidate_id": ev.candidate_id,
        "smiles": ev.canonical_smiles or ev.smiles,
        "hard_constraints_passed": bool(ev.constraint_flags and all(ev.constraint_flags.values())),
        "constraint_flags": ev.constraint_flags,
        "lambda_h_eV": ev.lambda_h_eV,
        "Vh_DIPRO_eV": ev.Vh_DIPRO_eV,
        "marcus_hole_rate_s_inv": ev.marcus_hole_rate_s_inv,
        "status": ev.status,
        "failure_reason": ev.error,
    } for ev in evaluations]
    prompt = (
        "Molecular design is now closed. Review the R1-R5 evaluator summary below and select exactly one existing candidate. "
        "Do not propose or modify a molecule. Return only JSON: {\"recommended_candidate_id\":\"R1\"}.\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )
    result = call_llm(messages + [{"role": "user", "content": prompt}], llm_cfg)
    append_jsonl(llm_debug_path, {"phase": "final_recommendation", "llm_call": result.to_dict()})
    if not result.ok:
        return None, result.content, result.error
    try:
        data = json.loads(result.content.strip().removeprefix("```json").removesuffix("```").strip())
        candidate_id = data.get("recommended_candidate_id")
    except Exception as exc:
        return None, result.content, str(exc)
    allowed = {f"R{i}" for i in range(1, 6)}
    return (candidate_id if candidate_id in allowed else None), result.content, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rdkit-only", action="store_true", help="Run only RDKit constraints, skip xTB/DIPRO.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_dir = Path(cfg["task_dir"])
    task_cfg_path = task_dir / "task_config.yaml"
    question_path = task_dir / "question.md"

    task_cfg = load_yaml(task_cfg_path)
    question_text = question_path.read_text(encoding="utf-8")
    llm_cfg = load_json(cfg["llm"]["api_config_file"])

    run_dir = make_run_dir(cfg["run"]["output_root"], cfg["run"]["run_prefix"])
    trajectory_path = run_dir / cfg["logging"]["trajectory_filename"]
    llm_debug_path = run_dir / cfg["logging"]["llm_debug_filename"]
    conversation_md_path = run_dir / cfg["logging"]["conversation_markdown_filename"]
    final_report_path = run_dir / cfg["logging"]["final_report_filename"]

    copy_file(question_path, run_dir / "question.md")
    copy_file(task_cfg_path, run_dir / "task_config.yaml")
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_json(run_dir / "api_config_used.json", redact_api_config(llm_cfg))

    initialize_markdown_report(
        conversation_md_path,
        title="Five-Ring xTB-DIPRO Marcus Benchmark Conversation",
        task_cfg=task_cfg,
        api_cfg=llm_cfg,
    )

    system_msg = (
        "You are a chemistry-focused molecular design agent. "
        "You must propose exactly one neutral small-molecule organic semiconductor candidate per round. "
        "The molecule must have a five-ring fused conjugated backbone. "
        "Return the final JSON object immediately and keep the rationale concise. "
        "Do not include markdown, comments, or extra text outside the JSON object."
    )

    messages = build_initial_messages(question_text, system_msg)
    max_rounds = int(cfg["run"]["max_rounds"])
    evaluations: List[CandidateEvaluation] = []
    raw_records: List[Dict[str, Any]] = []
    best_so_far: Optional[CandidateEvaluation] = None
    seen_canonical: Set[str] = set()
    parse_failures = 0

    for round_idx in range(1, max_rounds + 1):
        prompt_messages = list(messages)
        current_user_messages = [msg for msg in prompt_messages if msg.get("role") == "user"][-1:]
        llm_result = call_llm(prompt_messages, llm_cfg)
        raw_response = llm_result.content
        assistant_message = {
            "role": "assistant",
            "content": raw_response
            if raw_response
            else '{"candidate":{"candidate_id":"empty","parent_candidate_id":null,"proposal_mode":"direct_smiles","smiles":"","structural_modification":"empty output","modification_reasoning":"empty output"}}',
        }

        candidate = None
        parse_error = None

        if not llm_result.ok:
            parse_failures += 1
            parse_error = llm_result.error or "LLM call failed."
            evaluation = CandidateEvaluation(
                candidate_id=f"R{round_idx}",
                smiles="",
                valid=False,
                status="llm_call_failed",
                error=parse_error,
            )
        else:
            try:
                candidate = parse_candidate(
                    raw_response,
                    allow_markdown_fence=cfg["parsing"].get("allow_markdown_fence", True),
                )
                # Enforce expected candidate_id if model used something else.
                candidate["candidate_id"] = f"R{round_idx}"
                round_workdir = run_dir / f"round_{round_idx}_{candidate.get('candidate_id', f'R{round_idx}')}"
                evaluation = evaluate_candidate(
                    candidate,
                    task_cfg,
                    round_workdir,
                    seen_canonical,
                    run_xtb=not args.rdkit_only,
                )
            except ParseError as exc:
                parse_failures += 1
                parse_error = str(exc)
                evaluation = CandidateEvaluation(
                    candidate_id=f"R{round_idx}",
                    smiles="",
                    valid=False,
                    status="parse_failed",
                    error=parse_error,
                )
            except Exception as exc:
                evaluation = CandidateEvaluation(
                    candidate_id=f"R{round_idx}",
                    smiles=candidate.get("smiles", "") if candidate else "",
                    valid=False,
                    status="evaluator_exception",
                    error=str(exc),
                )

        if evaluation.canonical_smiles:
            seen_canonical.add(evaluation.canonical_smiles)

        best_so_far = update_best(best_so_far, evaluation)
        feedback = build_feedback(evaluation, task_cfg, best_so_far)

        record = {
            "round": round_idx,
            "messages_input_to_llm": current_user_messages,
        }

        append_jsonl(trajectory_path, record)
        append_jsonl(llm_debug_path, {"round": round_idx, "llm_call": llm_result.to_dict()})
        append_round_to_markdown_report(
            conversation_md_path,
            round_idx=round_idx,
            messages_input_to_llm=prompt_messages + [assistant_message],
            llm_result=llm_result,
            raw_response=raw_response,
            parsed_candidate=candidate,
            parse_error=parse_error,
            evaluation=evaluation,
            feedback=feedback,
            best_so_far=best_so_far,
        )

        raw_records.append(record)
        evaluations.append(evaluation)

        if round_idx < max_rounds:
            messages.append(assistant_message)
            messages.append({
                "role": "user",
                "content": build_next_user_message(
                    round_idx=round_idx,
                    candidate=candidate,
                    evaluation=evaluation,
                    feedback=feedback,
                    best_so_far=best_so_far,
                ),
            })

    recommended_candidate_id, recommendation_raw, recommendation_error = request_final_recommendation(
        evaluations, messages, llm_cfg, llm_debug_path
    )
    score = final_score(evaluations, task_cfg, recommended_candidate_id)

    report = {
        "success": True,
        "task_name": task_cfg["task_name"],
        "fixed_iteration_budget": max_rounds,
        "score": score,
        "final_recommendation": {
            "recommended_candidate_id": recommended_candidate_id,
            "raw_response": recommendation_raw,
            "error": recommendation_error,
        },
        "target": {
            "rate_threshold_s_inv": task_cfg["marcus"]["rate_threshold_s_inv"],
            "temperature_K": task_cfg["marcus"]["temperature_K"],
            "packing_model": task_cfg["packing"]["model"],
            "packing_dx_A": task_cfg["packing"]["dx_A"],
            "packing_dy_A": task_cfg["packing"]["dy_A"],
            "packing_dz_A": task_cfg["packing"]["dz_A"],
            "packing_search": task_cfg["packing"].get("search"),
        },
        "checks": {
            "n_iterations_completed": len(evaluations),
            "n_unique_smiles": len(seen_canonical),
            "n_valid": sum(1 for ev in evaluations if ev.valid),
            "n_constraint_valid": sum(1 for ev in evaluations if ev.constraint_flags and all(ev.constraint_flags.values())),
            "n_xtb_success": sum(1 for ev in evaluations if ev.xtb_success),
            "n_dipro_success": sum(1 for ev in evaluations if ev.dipro_success),
            "n_target_hits": sum(1 for ev in evaluations if ev.target_hit),
            "n_parse_failures": parse_failures,
        },
        "files": {
            "trajectory_file": str(trajectory_path),
            "llm_debug_file": str(llm_debug_path),
            "conversation_markdown_file": str(conversation_md_path),
            "run_dir": str(run_dir),
        },
        "candidates": [ev.to_dict() for ev in evaluations],
    }

    write_json(final_report_path, report)
    append_final_report_to_markdown(conversation_md_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
