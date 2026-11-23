"""Run harness for the public (model-visible) side of the benchmark.

THIS HARNESS DOES NOT SCORE (second-round review: 姚涵予 #2, 田双燕 §2.5 -- evaluation isolation).

Moving `answers/` out of the public tree would not have been enough on its own: the old
`final_report.json` printed the reference QED, the achievement ratio and the band table into every
run directory, so a model with filesystem access could read the answer out of a previous run. So the
split is functional, not cosmetic. This file runs the rounds, computes the candidate's properties,
returns reduced-bandwidth feedback and writes the trajectory; it produces `run_summary.json`, which
carries no score. Grading happens afterwards and elsewhere:

    python ../grader/scripts/score_run.py ../runs/<run_dir>

Nothing on this side imports the scorer, opens the grader's reference file, or knows the score bands.

Run:
    python src/main.py --config configs/default_deepseek.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import RDLogger

# RDKit writes parser and kekulization warnings to stderr for every malformed candidate a model
# submits. Those are expected events here -- the candidate is rejected and the reason is recorded in
# the trajectory -- so the raw warnings only make a run log look like it crashed.
RDLogger.DisableLog("rdApp.*")

from evaluator import MoleculeEvaluation, check_constraints, constraint_margins, evaluate_smiles  # noqa: E402
from feedback import generate_feedback  # noqa: E402
from llm_client import call_llm  # noqa: E402
from markdown_report import (  # noqa: E402
    initialize_markdown_report,
    append_round_to_markdown_report,
    append_final_report_to_markdown,
)
from objective import select_best_by_objective  # noqa: E402
from parser import ParseError, parse_candidate  # noqa: E402
from utils import append_jsonl, copy_file, load_json, load_yaml, make_run_dir, read_text, set_seed, write_json  # noqa: E402


def _public_eval_view(ev: Optional[MoleculeEvaluation]) -> Optional[Dict[str, Any]]:
    """The evaluation fields the model may see. Nothing here is reference-derived."""
    if ev is None:
        return None
    d = ev.to_dict()
    if not d.get("rejection_reasons"):
        d.pop("rejection_reasons", None)
    return d


def build_initial_messages(question_md: str, task_cfg: Dict[str, Any], start_eval: MoleculeEvaluation) -> List[Dict[str, str]]:
    start_smiles = task_cfg["start_smiles"]

    system_msg = (
        "You are a chemistry-focused molecular design agent. "
        "You must optimize the provided starting molecule through local structural modifications, "
        "balancing several competing objectives at once (drug-likeness, synthetic accessibility, "
        "structural-alert avoidance, and structural similarity). "
        "You must obey the required JSON output format exactly. "
        "Return the final JSON object immediately and keep the rationale concise. "
        "Do not include markdown, comments, or extra text outside the JSON object."
    )

    user_msg = f"""
{question_md}

## Starting Molecule

Starting SMILES:
{start_smiles}

Starting molecule properties computed by RDKit:
{json.dumps(_public_eval_view(start_eval), ensure_ascii=False, indent=2)}

The candidate must remain structurally related to the starting molecule.
The evaluator will compute Morgan fingerprint Tanimoto similarity to the starting molecule,
the SA synthetic-accessibility score, and PAINS/Brenk/NIH structural alerts.

Now propose the molecule for round 1.
Remember: output exactly one JSON object with one candidate.
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_next_user_message(
    round_idx: int,
    candidate: Optional[Dict[str, str]],
    evaluation: MoleculeEvaluation,
    feedback: Dict[str, Any],
    start_eval: MoleculeEvaluation,
    best_so_far: Optional[MoleculeEvaluation],
) -> str:
    return f"""
Round {round_idx} evaluation result:

Submitted candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2) if candidate else "Candidate could not be parsed."}

RDKit evaluation:
{json.dumps(_public_eval_view(evaluation), ensure_ascii=False, indent=2)}

Feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Starting molecule QED:
{start_eval.qed}

Best molecule by QED so far:
{json.dumps(_public_eval_view(best_so_far), ensure_ascii=False, indent=2) if best_so_far else "None"}

Now propose the molecule for round {round_idx + 1}.
Return exactly one JSON object with the required format and no extra text.
Preserve the starting molecule's main scaffold and use local structural modifications.
The evaluator reports computed property values only; you must check the constraint windows
stated in the task description yourself and judge your own progress.
""".strip()


def run_benchmark(config_path: str) -> Path:
    cfg = load_yaml(config_path)

    api_config_file = cfg["llm"]["api_config_file"]
    llm_cfg = load_json(api_config_file)

    set_seed(cfg["runtime"].get("random_seed", 42))

    task_dir = Path(cfg["task"]["task_dir"])
    question_path = task_dir / cfg["task"]["question_file"]
    task_config_path = task_dir / cfg["task"]["task_config_file"]

    question_md = read_text(question_path)
    task_cfg = load_yaml(task_config_path)
    start_smiles = task_cfg["start_smiles"]
    similarity_cfg = task_cfg["constraints"].get("similarity")

    run_dir = make_run_dir(cfg["runtime"]["run_dir"], task_cfg["task_id"],
                           label=llm_cfg.get("model"))
    trajectory_path = run_dir / cfg["logging"]["trajectory_filename"]
    llm_debug_path = run_dir / cfg["logging"].get("llm_debug_filename", "llm_api_debug.jsonl")
    conversation_markdown_path = run_dir / cfg["logging"].get("conversation_markdown_filename", "conversation.md")
    summary_path = run_dir / cfg["logging"].get("run_summary_filename", "run_summary.json")

    copy_file(config_path, run_dir / "config_used.yaml")
    copy_file(question_path, run_dir / "question.md")
    copy_file(task_config_path, run_dir / "task_config.yaml")
    _write_redacted_api_config(llm_cfg, run_dir / "api_config_used.json")

    start_eval = evaluate_smiles(
        start_smiles,
        start_smiles=start_smiles,
        similarity_cfg=similarity_cfg,
    )
    if not start_eval.valid:
        raise RuntimeError(f"Starting molecule is invalid: {start_eval.error}")

    initialize_markdown_report(
        conversation_markdown_path,
        title="Multi-Objective Molecule Design Benchmark Conversation",
        task_cfg=task_cfg,
        api_cfg=llm_cfg,
    )

    messages = build_initial_messages(question_md, task_cfg, start_eval)

    all_evaluations: List[MoleculeEvaluation] = []
    best_so_far: Optional[MoleculeEvaluation] = None
    parse_failures = 0
    llm_call_failures = 0
    empty_llm_responses = 0
    truncated_llm_responses = 0

    max_rounds = int(task_cfg["iteration"]["max_rounds"])

    for round_idx in range(1, max_rounds + 1):
        prompt_messages = list(messages)
        current_user_messages = [msg for msg in prompt_messages if msg.get("role") == "user"][-1:]
        llm_result = call_llm(prompt_messages, llm_cfg)
        raw_response = llm_result.content
        assistant_message = {
            "role": "assistant",
            "content": raw_response
            if raw_response
            else '{"candidate": {"name": "llm_empty_response", "smiles": "", "rationale": "LLM returned empty content."}}',
        }

        if not llm_result.ok:
            llm_call_failures += 1
        if llm_result.empty_content:
            empty_llm_responses += 1
        if getattr(llm_result, "truncated", False):
            truncated_llm_responses += 1

        candidate = None
        parse_error = None

        if not llm_result.ok:
            parse_failures += 1
            parse_error = llm_result.error or "LLM call failed or returned empty content."
            evaluation = MoleculeEvaluation(
                smiles="",
                valid=False,
                error=f"LLM failure before parsing: {parse_error}",
            )
        else:
            try:
                candidate = parse_candidate(
                    raw_response,
                    allow_markdown_fence=cfg["parsing"].get("allow_markdown_fence", True),
                )
                evaluation = evaluate_smiles(
                    candidate["smiles"],
                    start_smiles=start_smiles,
                    similarity_cfg=similarity_cfg,
                )
            except ParseError as exc:
                parse_failures += 1
                parse_error = str(exc)
                evaluation = MoleculeEvaluation(
                    smiles="",
                    valid=False,
                    error=f"Parse failure: {parse_error}",
                )

        all_evaluations.append(evaluation)
        if evaluation.valid:
            best_so_far = select_best_by_objective(all_evaluations, task_cfg["optimization_target"])

        feedback = generate_feedback(
            evaluation=evaluation,
            start_eval=start_eval,
            constraints=task_cfg["constraints"],
            best_so_far=best_so_far,
        )

        # The trajectory is the grader's only input, so the submitted candidate and its evaluation
        # are recorded here rather than only in the prompt log. A round with no submission still
        # gets a record, so a lost round stays visible instead of silently disappearing.
        record = {
            "round": round_idx,
            "messages_input_to_llm": current_user_messages,
            "raw_response": raw_response,
            "parse_error": parse_error,
            "submitted_candidate": candidate,
            "evaluation": evaluation.to_dict(),
            # AUDIT ONLY -- never sent to the model. Records which constraints the submission
            # missed and by how much, so a reviewer can see "LogP 3.0477 against a 3.0 ceiling,
            # missed by 0.0477" without re-deriving it. `evaluation.rejection_reasons` only ever
            # covers INADMISSIBLE input (salts, ions, bad valences) and is empty for the common
            # case of an admissible molecule that simply fell outside a window.
            "constraint_check": constraint_margins(evaluation, task_cfg["constraints"]),
            "feedback_returned": feedback,
        }
        append_jsonl(trajectory_path, record)
        append_jsonl(llm_debug_path, {
            "round": round_idx,
            "llm_call": llm_result.to_dict(),
        })
        append_round_to_markdown_report(
            conversation_markdown_path,
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

        if round_idx < max_rounds:
            messages.append(assistant_message)
            messages.append({
                "role": "user",
                "content": build_next_user_message(
                    round_idx=round_idx,
                    candidate=candidate,
                    evaluation=evaluation,
                    feedback=feedback,
                    start_eval=start_eval,
                    best_so_far=best_so_far,
                )
            })

    # No score is computed here. Everything below is a factual record of what happened, which the
    # grader turns into a score without calling the API again.
    # Prompt identity, recomputed from the file that was actually shown to the model. A trajectory
    # carrying a different digest than the current question.md is not evidence for the current task.
    question_sha = hashlib.sha256(question_path.read_bytes()).hexdigest()

    summary = {
        "task_name": task_cfg["task_name"],
        "task_id": task_cfg["task_id"],
        "prompt_version": task_cfg.get("prompt_version"),
        "question_sha256": question_sha,
        "positioning": task_cfg.get("positioning"),
        "fixed_iteration_budget": max_rounds,
        "scored_by": "grader/scripts/score_run.py -- this harness deliberately does not score",
        "target": {
            "optimization_target": task_cfg.get("optimization_target"),
            "start_smiles": start_smiles,
            "constraints": task_cfg["constraints"],
        },
        "checks": {
            "n_iterations_completed": len(all_evaluations),
            "n_valid": sum(1 for ev in all_evaluations if ev.valid),
            "n_constraint_valid": sum(
                1
                for ev in all_evaluations
                if ev.valid and all(check_constraints(ev, task_cfg["constraints"]).values())
            ),
            "n_parse_failures": parse_failures,
            "n_llm_call_failures": llm_call_failures,
            "n_empty_llm_responses": empty_llm_responses,
            # Rounds lost to a response that was still truncated after the retry schedule was
            # exhausted. Should be 0; a non-zero value means the token budget needs raising.
            "n_truncated_llm_responses": truncated_llm_responses,
            "n_rejected_by_input_validation": sum(
                1 for ev in all_evaluations if ev.rejection_reasons
            ),
            "rejection_reason_codes": sorted({
                r for ev in all_evaluations for r in ev.rejection_reasons
            }),
        },
        "files": {
            "trajectory_file": str(trajectory_path),
            "llm_debug_file": str(llm_debug_path),
            "conversation_markdown_file": str(conversation_markdown_path),
            "run_dir": str(run_dir),
        },
        "start_molecule": start_eval.to_dict(),
        "candidates": [ev.to_dict() for ev in all_evaluations],
    }

    write_json(summary_path, summary)
    append_final_report_to_markdown(conversation_markdown_path, summary)
    print(f"Run complete. Summary: {summary_path}")
    print("This harness does not score. To grade the run:")
    print(f"    python ../grader/scripts/score_run.py {run_dir}")
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=2))
    return run_dir


def _write_redacted_api_config(llm_cfg: Dict[str, Any], dst: Path) -> None:
    """Persist the API config used for this run WITHOUT leaking the API key."""
    redacted = dict(llm_cfg)
    if redacted.get("api_key"):
        redacted["api_key"] = "<redacted>"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(redacted, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to benchmark config YAML.")
    args = parser.parse_args()

    run_benchmark(args.config)


if __name__ == "__main__":
    main()
