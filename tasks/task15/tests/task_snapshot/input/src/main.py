"""Run harness for the public (model-visible) side of the benchmark.

Two review findings shaped this file.

1. Evaluation isolation (issue 1). This harness DOES NOT SCORE. It runs the rounds, computes the
   candidate's properties, returns reduced-bandwidth feedback and writes the trajectory. It has no
   access to the reference target, the achievement ratio or the score bands -- those live in
   `grader/`, and `grader/scripts/score_run.py` turns a finished run into `final_report.json`. So an
   agent that can read everything on this side still cannot read the answer.

2. Tool interaction (issue 4). A round is no longer one prompt and one answer. The model may spend a
   quota of diagnostic tool calls first, across as many tool messages as the per-round limit allows,
   and it decides which tool to use on which molecule. Tool calls, tool responses, tool errors and
   the submission that followed are all recorded per round, so the trajectory shows the decisions and
   not just the outcome.

Run:
    python src/main.py --config configs/default_deepseek.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import RDLogger

# RDKit writes parser and kekulization warnings to stderr for every malformed candidate a model
# submits. Those are expected events here -- the candidate is rejected and the reason is recorded in
# the trajectory -- so the raw warnings only make a run log look like it crashed. Silence them; the
# evaluator's own rejection reason codes are the record.
RDLogger.DisableLog("rdApp.*")

from evaluator import MoleculeEvaluation, check_constraints, evaluate_smiles  # noqa: E402
from feedback import generate_feedback  # noqa: E402
from llm_client import call_llm
from markdown_report import (
    initialize_markdown_report,
    append_round_to_markdown_report,
    append_final_report_to_markdown,
)
from objective import select_best_by_objective
from parser import ParseError, parse_agent_message
from tools import TOOL_NAMES, ToolSession, tool_catalogue_markdown
from utils import append_jsonl, copy_file, load_json, load_yaml, make_run_dir, read_text, set_seed, write_json


def _environment_record() -> Dict[str, Any]:
    """The toolchain that produced this run.

    QED, SA_Score, the PAINS/Brenk/NIH catalogs and the Morgan fingerprint all come from RDKit, and
    good designs have to operate close to the similarity floor, so the RDKit version is part of the
    measurement rather than an environment detail. requirements.txt pins it; this records what
    actually ran, so a re-score can tell the two apart.
    """
    import platform
    import rdkit
    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "platform": platform.platform(),
        "pinned_in": "requirements.txt",
    }


def _public_eval_view(ev: Optional[MoleculeEvaluation]) -> Optional[Dict[str, Any]]:
    """The evaluation fields the model may see. Nothing here is reference-derived."""
    if ev is None:
        return None
    d = ev.to_dict()
    if not d.get("rejection_reasons"):
        d.pop("rejection_reasons", None)
    return d


def build_initial_messages(
    question_md: str,
    task_cfg: Dict[str, Any],
    start_eval: MoleculeEvaluation,
    tool_cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    start_smiles = task_cfg["start_smiles"]

    system_msg = (
        "You are a chemistry-focused molecular design agent working on a surrogate-property "
        "optimization task. You modify a given starting molecule under hard property constraints, "
        "balancing a computed drug-likeness surrogate against synthetic accessibility, structural "
        "alerts and a fingerprint-similarity budget. "
        "You have a small set of diagnostic tools and a per-round call quota; you decide whether to "
        "use them, which ones, and on which molecules. "
        "Every message you send must be exactly one JSON object and nothing else: either "
        '{"tool_calls": [...]} or {"candidate": {...}}. '
        "No markdown, no commentary outside the JSON object."
    )

    user_msg = f"""
{question_md}

## Starting Molecule

Starting SMILES:
{start_smiles}

Starting molecule properties computed by RDKit:
{json.dumps(_public_eval_view(start_eval), ensure_ascii=False, indent=2)}

## Tools available to you

{tool_catalogue_markdown()}

Tool budget: {tool_cfg['max_calls_per_round']} tool calls per round, spread over at most
{tool_cfg['max_tool_messages_per_round']} tool messages. A call that fails because of a bad tool name
or bad arguments still costs you one call. Once the budget is gone further calls are refused, and you
must submit.

Send either a tool message:
{{"tool_calls": [{{"tool": "constraint_checklist", "arguments": {{"smiles": "..."}}}}]}}
or your submission for this round:
{{"candidate": {{"name": "candidate_round_1", "smiles": "...", "rationale": "..."}}}}

Now begin round 1.
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_tool_result_message(
    round_idx: int,
    responses: List[Dict[str, Any]],
    tool_messages_used: int,
    tool_messages_allowed: int,
    calls_remaining: int,
) -> str:
    if tool_messages_used >= tool_messages_allowed or calls_remaining <= 0:
        closing = (
            'You have no tool budget left this round. Your next message MUST be your '
            '{"candidate": ...} submission; a tool message now forfeits the round.'
        )
    else:
        closing = (
            f"You may send at most {tool_messages_allowed - tool_messages_used} more tool "
            f"message(s) this round ({calls_remaining} call(s) left), or submit your candidate now."
        )

    return f"""
Tool results for round {round_idx}:

{json.dumps(responses, ensure_ascii=False, indent=2)}

{closing}
Reply with exactly one JSON object and no other text.
""".strip()


def build_next_user_message(
    round_idx: int,
    candidate: Optional[Dict[str, str]],
    evaluation: MoleculeEvaluation,
    feedback: Dict[str, Any],
    start_eval: MoleculeEvaluation,
    best_so_far: Optional[MoleculeEvaluation],
    tool_cfg: Dict[str, Any],
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

Reminder: only a design that satisfies every constraint AND has QED strictly above the starting
molecule's {start_eval.qed} can score at all. A feasible design that does not beat that value is worth
nothing, and resubmitting the starting molecule is worth nothing.

Round {round_idx + 1} begins now. Your tool budget resets to {tool_cfg['max_calls_per_round']} calls
over at most {tool_cfg['max_tool_messages_per_round']} tool messages. Send either a tool message or
your candidate for round {round_idx + 1}.
The evaluator reports computed property values only; you must check the constraint windows stated in
the task description yourself and judge your own progress.
""".strip()


def _tool_cfg(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(task_cfg.get("tools") or {})
    cfg.setdefault("max_calls_per_round", 6)
    cfg.setdefault("max_tool_messages_per_round", 2)
    return cfg


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
    constraints = task_cfg["constraints"]
    similarity_cfg = constraints.get("similarity")
    tool_cfg = _tool_cfg(task_cfg)

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
        title="Surrogate-Property Molecule Design Benchmark Conversation",
        task_cfg=task_cfg,
        api_cfg=llm_cfg,
    )

    messages = build_initial_messages(question_md, task_cfg, start_eval, tool_cfg)

    all_evaluations: List[MoleculeEvaluation] = []
    best_so_far: Optional[MoleculeEvaluation] = None
    parse_failures = 0
    llm_call_failures = 0
    empty_llm_responses = 0
    truncated_llm_responses = 0
    tool_stats: Dict[str, Any] = {
        "total_tool_calls": 0,
        "successful_tool_calls": 0,
        "failed_tool_calls": 0,
        "error_codes": {},
        "calls_by_tool": {name: 0 for name in TOOL_NAMES},
        "rounds_using_tools": 0,
        "rounds_forfeited_without_submission": 0,
    }

    max_rounds = int(task_cfg["iteration"]["max_rounds"])

    for round_idx in range(1, max_rounds + 1):
        session = ToolSession(
            start_smiles=start_smiles,
            constraints=constraints,
            similarity_cfg=similarity_cfg,
            max_calls=int(tool_cfg["max_calls_per_round"]),
        )
        candidate, parse_error, evaluation, turns = _run_one_round(
            round_idx=round_idx,
            messages=messages,
            session=session,
            tool_cfg=tool_cfg,
            start_smiles=start_smiles,
            similarity_cfg=similarity_cfg,
            llm_cfg=llm_cfg,
            cfg=cfg,
            llm_debug_path=llm_debug_path,
        )

        for turn in turns:
            if turn["llm_call"]["ok"] is False:
                llm_call_failures += 1
            if turn["llm_call"].get("empty_content"):
                empty_llm_responses += 1
            if turn["llm_call"].get("truncated"):
                truncated_llm_responses += 1
        if parse_error:
            parse_failures += 1
        if candidate is None:
            tool_stats["rounds_forfeited_without_submission"] += 1

        # Tool accounting. Every response envelope is counted, including the ones the dispatcher
        # rejected before a tool ran (unknown name, malformed call, exhausted quota), because those
        # are exactly the failure branches the review asked to see exercised.
        round_calls = 0
        for turn in turns:
            for resp in turn.get("tool_responses") or []:
                round_calls += 1
                tool_stats["total_tool_calls"] += 1
                name = resp.get("tool")
                if name in tool_stats["calls_by_tool"]:
                    tool_stats["calls_by_tool"][name] += 1
                if resp.get("ok"):
                    tool_stats["successful_tool_calls"] += 1
                else:
                    tool_stats["failed_tool_calls"] += 1
                    code = (resp.get("error") or {}).get("code", "unknown")
                    tool_stats["error_codes"][code] = tool_stats["error_codes"].get(code, 0) + 1
        if round_calls:
            tool_stats["rounds_using_tools"] += 1

        all_evaluations.append(evaluation)
        if evaluation.valid:
            best_so_far = select_best_by_objective(all_evaluations, task_cfg["optimization_target"])

        feedback = generate_feedback(
            evaluation=evaluation,
            start_eval=start_eval,
            constraints=constraints,
            best_so_far=best_so_far,
        )

        record = {
            "round": round_idx,
            "turns": turns,
            "tool_calls_this_round": session.log,
            "tool_calls_used": session.calls_used,
            "tool_calls_allowed": session.max_calls,
            "parse_error": parse_error,
            "submitted_candidate": candidate,
            "evaluation": evaluation.to_dict(),
            "feedback_returned": feedback,
        }
        append_jsonl(trajectory_path, record)
        append_round_to_markdown_report(
            conversation_markdown_path,
            round_idx=round_idx,
            messages_input_to_llm=messages,
            llm_result=None,
            raw_response=turns[-1]["raw_response"] if turns else "",
            parsed_candidate=candidate,
            parse_error=parse_error,
            evaluation=evaluation,
            feedback=feedback,
            best_so_far=best_so_far,
        )

        if round_idx < max_rounds:
            messages.append({
                "role": "user",
                "content": build_next_user_message(
                    round_idx=round_idx,
                    candidate=candidate,
                    evaluation=evaluation,
                    feedback=feedback,
                    start_eval=start_eval,
                    best_so_far=best_so_far,
                    tool_cfg=tool_cfg,
                )
            })

    summary = {
        "task_name": task_cfg["task_name"],
        "task_id": task_cfg["task_id"],
        "positioning": task_cfg.get("positioning"),
        # Version stamps travel with the run. prompt_version decides whether this TRAJECTORY is still
        # evidence about the task the package poses; ruler_version decides whether a stored SCORE is
        # comparable. Recording both here means a stale run is detectable without reading its text.
        "prompt_version": int(task_cfg.get("prompt_version", 1)),
        "ruler_version": str(task_cfg.get("ruler_version", "unversioned")),
        "environment": _environment_record(),
        "fixed_iteration_budget": max_rounds,
        "scored_by": "grader/scripts/score_run.py -- this harness deliberately does not score",
        "target": {
            "optimization_target": task_cfg.get("optimization_target"),
            "start_smiles": start_smiles,
            "constraints": constraints,
        },
        "checks": {
            "n_iterations_completed": len(all_evaluations),
            "n_valid": sum(1 for ev in all_evaluations if ev.valid),
            "n_constraint_valid": sum(
                1
                for ev in all_evaluations
                if ev.valid and all(check_constraints(ev, constraints).values())
            ),
            "n_parse_failures": parse_failures,
            "n_llm_call_failures": llm_call_failures,
            "n_empty_llm_responses": empty_llm_responses,
            "n_truncated_llm_responses": truncated_llm_responses,
            "n_rejected_by_input_validation": sum(
                1 for ev in all_evaluations if ev.rejection_reasons
            ),
            "rejection_reason_codes": sorted({
                r for ev in all_evaluations for r in ev.rejection_reasons
            }),
        },
        "tool_usage": tool_stats,
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
    print(json.dumps(summary["tool_usage"], ensure_ascii=False, indent=2))
    return run_dir


def _run_one_round(
    round_idx: int,
    messages: List[Dict[str, str]],
    session: ToolSession,
    tool_cfg: Dict[str, Any],
    start_smiles: str,
    similarity_cfg: Optional[Dict[str, Any]],
    llm_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
    llm_debug_path: Path,
) -> Tuple[Optional[Dict[str, str]], Optional[str], MoleculeEvaluation, List[Dict[str, Any]]]:
    """Drive one round: zero or more tool messages, then a submission.

    Returns (candidate, parse_error, evaluation, turns). `turns` records every LLM call in the round
    together with the tool responses it produced, which is what makes the tool decisions auditable.
    """
    max_tool_messages = int(tool_cfg["max_tool_messages_per_round"])
    turns: List[Dict[str, Any]] = []
    tool_messages_used = 0

    for turn_idx in range(1, max_tool_messages + 2):
        llm_result = call_llm(list(messages), llm_cfg)
        raw_response = llm_result.content
        assistant_message = {
            "role": "assistant",
            "content": raw_response if raw_response else
            '{"candidate": {"name": "llm_empty_response", "smiles": "", "rationale": "LLM returned empty content."}}',
        }
        messages.append(assistant_message)

        append_jsonl(llm_debug_path, {
            "round": round_idx,
            "turn": turn_idx,
            "llm_call": llm_result.to_dict(),
        })

        turn: Dict[str, Any] = {
            "turn": turn_idx,
            "kind": None,
            "raw_response": raw_response,
            "llm_call": {
                "ok": llm_result.ok,
                "model": llm_result.model,
                "attempts": llm_result.attempts,
                "finish_reason": llm_result.finish_reason,
                "empty_content": llm_result.empty_content,
                "truncated": getattr(llm_result, "truncated", False),
                "error": llm_result.error,
                "max_tokens_used": llm_result.max_tokens_used,
            },
            "tool_responses": None,
            "parse_error": None,
        }

        if not llm_result.ok:
            turn["kind"] = "llm_failure"
            turn["parse_error"] = llm_result.error or "LLM call failed or returned empty content."
            turns.append(turn)
            evaluation = MoleculeEvaluation(
                smiles="", valid=False,
                error=f"LLM failure before parsing: {turn['parse_error']}",
            )
            return None, turn["parse_error"], evaluation, turns

        try:
            kind, payload = parse_agent_message(
                raw_response,
                allow_markdown_fence=cfg["parsing"].get("allow_markdown_fence", True),
            )
        except ParseError as exc:
            turn["kind"] = "parse_failure"
            turn["parse_error"] = str(exc)
            turns.append(turn)
            evaluation = MoleculeEvaluation(
                smiles="", valid=False, error=f"Parse failure: {exc}",
            )
            return None, str(exc), evaluation, turns

        if kind == "candidate":
            turn["kind"] = "submission"
            turns.append(turn)
            evaluation = evaluate_smiles(
                payload["smiles"],
                start_smiles=start_smiles,
                similarity_cfg=similarity_cfg,
            )
            return payload, None, evaluation, turns

        # kind == "tool_calls"
        if tool_messages_used >= max_tool_messages:
            # The model was told this message had to be a submission and sent tool calls anyway. The
            # round is forfeited -- that is the cost of mismanaging the budget, and it has to be
            # visible in the trajectory rather than silently repaired.
            turn["kind"] = "tool_call_after_budget"
            turn["parse_error"] = (
                f"Round {round_idx} used all {max_tool_messages} tool messages and then sent "
                f"another tool message instead of a candidate; the round is forfeited."
            )
            turns.append(turn)
            evaluation = MoleculeEvaluation(smiles="", valid=False, error=turn["parse_error"])
            return None, turn["parse_error"], evaluation, turns

        responses = session.call_batch(payload)
        tool_messages_used += 1
        turn["kind"] = "tool_calls"
        turn["tool_responses"] = responses
        turns.append(turn)

        messages.append({
            "role": "user",
            "content": build_tool_result_message(
                round_idx=round_idx,
                responses=responses,
                tool_messages_used=tool_messages_used,
                tool_messages_allowed=max_tool_messages,
                calls_remaining=max(0, session.max_calls - session.calls_used),
            ),
        })

    evaluation = MoleculeEvaluation(smiles="", valid=False, error="Round ended without a submission.")
    return None, "Round ended without a submission.", evaluation, turns


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
