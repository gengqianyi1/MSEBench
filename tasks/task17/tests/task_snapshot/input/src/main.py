"""Benchmark harness: five graded rounds, each with an optional agent-driven diagnostic tool phase.

Per round the model may, before committing:
  * issue tool calls of its own choosing (`{"tool_calls": [...]}`), read the structured results,
    recover from real tool errors, and issue more -- up to `tools.MAX_TOOL_TURNS_PER_ROUND` turns of
    at most `tools.MAX_TOOL_CALLS_PER_TURN` calls each;
  * then commit exactly one process (`{"candidate": {...}}`), which is measured and graded.
Tool calls are free of charge in graded terms: only the committed process is measured. A model may
also skip the tool phase entirely and commit immediately.

Fixed budget, no early stop: every model is graded on the same five measurements, and the score is
the best usable batch anywhere in the trajectory (src/scoring.py), so exploring in a later round is
never punished.

Everything a reviewer needs to replay a run is written into the run directory: the question and task
config actually used, the harness config, a redacted copy of the API config, the full trajectory with
raw responses, parse errors and every tool call with its result, the raw API debug log, and the final
report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from design_grid import RESPONSE_UNITS, START, TOTAL_EDIT_COST
from evaluator import (BatchEvaluation, check_constraints, evaluate_design,
                       evaluate_reference_start, slot_table_text)
from feedback import generate_feedback
from llm_client import call_llm
from markdown_report import (append_final_report_to_markdown, append_round_to_markdown_report,
                            initialize_markdown_report)
from parser import ParseError, parse_turn
from scoring import final_score, select_best_by_objective
from tools import (MAX_TOOL_CALLS_PER_TURN, MAX_TOOL_TURNS_PER_ROUND, run_tool_calls,
                   tool_catalogue_text)
from utils import (append_jsonl, copy_file, load_json, load_yaml, make_run_dir, read_text, set_seed,
                   write_json)


def build_initial_messages(question_md: str, task_cfg: Dict[str, Any],
                           start_eval: BatchEvaluation) -> List[Dict[str, str]]:
    system_msg = (
        "You are a crystallization and particle-engineering process design agent working on a "
        "SIMULATED quasi-emulsion spherical agglomeration process for a needle-habit drug substance. "
        "You must redesign the baseline process under a limited controlled-change budget so that all "
        "four simulated quality attributes sit inside their specification windows at once. "
        "In each round you may first call diagnostic tools, and you must then commit exactly one "
        "process. "
        "You must obey the required JSON output formats exactly: emit either a tool-call object or a "
        "candidate object, and nothing else. "
        "Do not include markdown, comments, or extra text outside the JSON object."
    )

    start_summary = {
        "baseline_process": START,
        "simulated_responses_of_the_baseline_process": start_eval.responses,
        "response_units": RESPONSE_UNITS,
        "change_points_spent": start_eval.edit_cost,
        "change_points_if_all_twelve_were_changed": TOTAL_EDIT_COST,
    }

    user_msg = f"""
{question_md}

## Baseline Process (your starting point)

{json.dumps(start_summary, ensure_ascii=False, indent=2)}

## The 12 variables you may set, their allowed values, and what changing each one costs

{slot_table_text()}

Every submission must give all twelve variables. Values must be taken from the lists above; no new
material, no new unit operation and no value outside a list is admissible.

## Diagnostic tools you may call (optional)

You decide whether to call a tool, which tool, how many times, and with what arguments. Tool calls
are not measurements and cost you no round. In each round you may use at most
{MAX_TOOL_TURNS_PER_ROUND} tool turns of at most {MAX_TOOL_CALLS_PER_TURN} calls each; after that you
must commit a process.

{tool_catalogue_text()}

None of these tools returns any quality attribute, any prediction of one, the desirability D, an
achievement fraction or a score. They tell you whether a process is legal, runnable and affordable --
not whether it is good. Tool calls can fail (unknown tool, bad arguments, off-grid process, exhausted
quota); a failure returns an `error` field with a `detail`, and you may fix the call and retry within
the same round.

To call tools, return exactly:

```json
{{"tool_calls": [{{"tool": "check_process_legality", "arguments": {{"process": {{...}}}}}}]}}
```

To commit the process for this round, return exactly the candidate object described under
*Required Output Format*.

Now begin round 1. You may call tools first, or commit a process immediately.
Return exactly one JSON object: either a tool-call object or one candidate.
""".strip()

    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_next_user_message(round_idx: int, candidate: Optional[Dict[str, Any]],
                            evaluation: BatchEvaluation, feedback: Dict[str, Any],
                            start_eval: BatchEvaluation,
                            best_so_far: Optional[BatchEvaluation]) -> str:
    return f"""
Round {round_idx} evaluation result:

Submitted process:
{json.dumps(candidate, ensure_ascii=False, indent=2) if candidate else "Candidate could not be parsed."}

Evaluator feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Simulated responses of the baseline process, for reference:
{json.dumps(start_eval.responses, ensure_ascii=False)}

Best desirability D reached so far in this run:
{best_so_far.desirability if best_so_far is not None and best_so_far.valid else "None"}

Now work on round {round_idx + 1}. You may call diagnostic tools first, or commit a process
immediately. Return exactly one JSON object: either a tool-call object or one candidate.
The evaluator reports simulated values only. It never tells you the reference optimum, your
achievement or your score, and it never suggests which variable to move next.
""".strip()


def build_tool_result_message(round_idx: int, tool_turn: int, tool_result: Dict[str, Any],
                              turns_left: int) -> str:
    """What the model sees after a tool turn. No measurement, no quality information."""
    if turns_left > 0:
        tail = (f"You may call tools once more ({turns_left} tool turn(s) left in round "
                f"{round_idx}), or commit the process for round {round_idx} now.")
    else:
        tail = (f"The tool phase for round {round_idx} is over. You must now commit the process for "
                f"round {round_idx}.")
    return f"""
Tool results for round {round_idx}, tool turn {tool_turn}
(these are diagnostics only: no quality attribute, no desirability, no score):

{json.dumps(tool_result["results"], ensure_ascii=False, indent=2)}

Calls executed: {tool_result["n_executed"]}   errors: {tool_result["n_error"]}

{tail}
Return exactly one JSON object: either a tool-call object or one candidate.
""".strip()


def history_entry(round_idx: int, evaluation: BatchEvaluation) -> Dict[str, Any]:
    """One compact row of the history table the model is shown. Pure recap of what it already saw."""
    return {
        "round": round_idx,
        "variables_changed": list(evaluation.changed_slots or []),
        "change_points_spent": evaluation.edit_cost,
        "status": ("measured" if evaluation.valid
                   else "not_a_design_space_member" if evaluation.design is None
                   else "batch_cannot_be_run"),
        "simulated_responses": evaluation.responses,
        "desirability_D": evaluation.desirability,
    }


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
    objective_cfg = task_cfg["objective"]
    constraints_cfg = task_cfg["constraints"]

    run_dir = make_run_dir(cfg["runtime"]["run_dir"], task_cfg["task_id"],
                           label=llm_cfg.get("model"))
    trajectory_path = run_dir / cfg["logging"]["trajectory_filename"]
    llm_debug_path = run_dir / cfg["logging"].get("llm_debug_filename", "llm_api_debug.jsonl")
    conversation_markdown_path = run_dir / cfg["logging"].get("conversation_markdown_filename",
                                                             "conversation.md")
    final_report_path = run_dir / cfg["logging"]["final_report_filename"]

    copy_file(config_path, run_dir / "config_used.yaml")
    copy_file(question_path, run_dir / "question.md")
    copy_file(task_config_path, run_dir / "task_config.yaml")
    _write_redacted_api_config(llm_cfg, run_dir / "api_config_used.json")

    start_eval = evaluate_reference_start(objective_cfg)
    if not start_eval.valid:
        raise RuntimeError(f"Registered process is not evaluable: {start_eval.error}")

    initialize_markdown_report(conversation_markdown_path,
                               title="Spherical Agglomeration Four-Window Co-Design Benchmark Conversation",
                               task_cfg=task_cfg, api_cfg=llm_cfg)

    messages = build_initial_messages(question_md, task_cfg, start_eval)

    all_evaluations: List[BatchEvaluation] = []
    best_so_far: Optional[BatchEvaluation] = None
    history: List[Dict[str, Any]] = []
    parse_failures = 0
    llm_call_failures = 0
    empty_llm_responses = 0
    truncated_llm_responses = 0
    tool_turns_total = 0
    tool_calls_total = 0
    tool_errors_total = 0
    tool_usage: Dict[str, int] = {}
    rounds_using_tools = 0
    tool_recoveries = 0

    max_rounds = int(task_cfg["iteration"]["max_rounds"])

    for round_idx in range(1, max_rounds + 1):
        # ------------------------------------------------------------------ tool phase (optional)
        # The model decides. Each turn it may request tool calls or commit a process; after
        # MAX_TOOL_TURNS_PER_ROUND tool turns it must commit. Nothing here is measured or graded.
        tool_phase: List[Dict[str, Any]] = []
        raw_response = ""
        llm_result = None
        turn_kind = None
        parse_error = None
        candidate = None

        for tool_turn in range(MAX_TOOL_TURNS_PER_ROUND + 1):
            prompt_messages = list(messages)
            current_user_messages = [m for m in prompt_messages if m.get("role") == "user"][-1:]
            llm_result = call_llm(prompt_messages, llm_cfg)
            raw_response = llm_result.content
            assistant_message = {
                "role": "assistant",
                "content": raw_response if raw_response else json.dumps(
                    {"candidate": {"name": "llm_empty_response", "rationale":
                                   "LLM returned empty content.", "process": {}}}),
            }
            append_jsonl(llm_debug_path, {"round": round_idx, "tool_turn": tool_turn,
                                          "llm_call": llm_result.to_dict()})

            if not llm_result.ok:
                llm_call_failures += 1
            if llm_result.empty_content:
                empty_llm_responses += 1
            if getattr(llm_result, "truncated", False):
                truncated_llm_responses += 1

            if not llm_result.ok:
                turn_kind = "llm_failure"
                parse_error = llm_result.error or "LLM call failed or returned empty content."
                break

            try:
                turn = parse_turn(
                    raw_response,
                    allow_markdown_fence=cfg["parsing"].get("allow_markdown_fence", True))
            except ParseError as exc:
                turn_kind = "parse_failure"
                parse_error = str(exc)
                break

            if turn["kind"] == "candidate":
                turn_kind = "candidate"
                candidate = turn["candidate"]
                break

            # a tool turn
            if tool_turn >= MAX_TOOL_TURNS_PER_ROUND:
                # The quota is spent and the model asked for tools anyway. That is a real failure the
                # model has to handle; it is told so and given one last chance to commit.
                # `closed: True` marks this as a refusal, not a tool turn: nothing ran. Both the
                # per-round counters below and the run-level tool_interaction block exclude it, so the
                # two never disagree about how many calls were made.
                tool_phase.append({"tool_turn": tool_turn + 1, "requested": turn["tool_calls"],
                                   "results": [{"error": "tool_phase_closed",
                                                "detail": f"the tool phase of round {round_idx} is "
                                                          f"over; commit a process now.",
                                                "recoverable": True}],
                                   "closed": True,
                                   "n_executed": 0, "n_error": 1})
                tool_errors_total += 1
                messages.append(assistant_message)
                messages.append({"role": "user", "content":
                                 f"The tool phase for round {round_idx} is closed. Return the "
                                 f"candidate object for round {round_idx} now, and nothing else."})
                continue

            result = run_tool_calls(turn["tool_calls"])
            tool_turns_total += 1
            tool_calls_total += result["n_executed"] + result["n_error"]
            tool_errors_total += result["n_error"]
            for r in result["results"]:
                if r.get("tool"):
                    tool_usage[r["tool"]] = tool_usage.get(r["tool"], 0) + 1
            if tool_phase and tool_phase[-1]["n_error"] > 0 and result["n_error"] == 0:
                tool_recoveries += 1          # a failed turn followed by a clean one
            tool_phase.append({"tool_turn": tool_turn + 1, "requested": turn["tool_calls"],
                               **result})

            messages.append(assistant_message)
            messages.append({"role": "user", "content": build_tool_result_message(
                round_idx=round_idx, tool_turn=tool_turn + 1, tool_result=result,
                turns_left=MAX_TOOL_TURNS_PER_ROUND - tool_turn - 1)})

        if tool_phase:
            rounds_using_tools += 1

        # ------------------------------------------------------------------ graded measurement
        if turn_kind == "candidate" and candidate is not None:
            evaluation = evaluate_design(candidate["process"], objective_cfg)
        elif turn_kind == "llm_failure":
            parse_failures += 1
            evaluation = BatchEvaluation(error=f"LLM failure before parsing: {parse_error}")
        elif turn_kind == "parse_failure":
            parse_failures += 1
            evaluation = BatchEvaluation(error=f"Parse failure: {parse_error}")
        else:
            parse_failures += 1
            parse_error = (parse_error or "the model never committed a process in this round "
                                          "(it spent the round on tool calls).")
            evaluation = BatchEvaluation(error=f"No candidate committed: {parse_error}")

        all_evaluations.append(evaluation)
        if evaluation.valid:
            best_so_far = select_best_by_objective(all_evaluations)

        feedback = generate_feedback(evaluation=evaluation, constraints_cfg=constraints_cfg,
                                    best_so_far=best_so_far, round_idx=round_idx,
                                    max_rounds=max_rounds, history=list(history))
        history.append(history_entry(round_idx, evaluation))

        # The trajectory file is a deliverable: it records what was submitted and what the evaluator
        # measured, plus the raw response, any parse error and every tool call with its result, so a
        # reviewer can replay a round without the LLM debug log.
        record = {
            "round": round_idx,
            "messages_input_to_llm": current_user_messages,
            "tool_phase": tool_phase,
            # Counted over REAL tool turns only (a refused turn ran nothing). Same definition as the
            # run-level tool_interaction block.
            "n_tool_turns": len([t for t in tool_phase if not t.get("closed")]),
            "n_tool_calls": sum(t.get("n_executed", 0) + t.get("n_error", 0)
                                for t in tool_phase if not t.get("closed")),
            "n_tool_errors": sum(t.get("n_error", 0) for t in tool_phase),
            "n_tool_phase_closed": sum(1 for t in tool_phase if t.get("closed")),
            "raw_llm_response": raw_response,
            "turn_kind": turn_kind,
            "parse_error": parse_error,
            "submitted_candidate": candidate,
            "evaluation": evaluation.to_dict(),
            "feedback_returned": feedback,
        }
        append_jsonl(trajectory_path, record)
        append_round_to_markdown_report(
            conversation_markdown_path, round_idx=round_idx,
            messages_input_to_llm=prompt_messages + [assistant_message],
            llm_result=llm_result, raw_response=raw_response, parsed_candidate=candidate,
            parse_error=parse_error, evaluation=evaluation, feedback=feedback,
            best_so_far=best_so_far)

        if round_idx < max_rounds:
            messages.append(assistant_message)
            messages.append({"role": "user", "content": build_next_user_message(
                round_idx=round_idx, candidate=candidate, evaluation=evaluation, feedback=feedback,
                start_eval=start_eval, best_so_far=best_so_far)})

    # SCORING IS A GRADER-SIDE STEP.
    # The achievement denominator is the reference optimum, which lives only in the grader-side
    # task_config.yaml. Running against the PUBLIC config (task_config.public.yaml, which carries no
    # `reference:` block by construction) therefore produces a trajectory and no score, and that is
    # the intended behaviour rather than a degraded one: the run directory can be handed to the
    # grading side, which scores it with scripts/rescore_run.py against the full config. It also means
    # a participant-side run never writes the reference value into its own output.
    reference_cfg = task_cfg.get("reference")
    if reference_cfg is None:
        score = {
            "total_score": None,
            "scored": False,
            "note": ("Not scored here. This run used a task config with no `reference:` block (the "
                     "public build), so the achievement denominator is unavailable. Hand the run "
                     "directory to the grading side and score it with "
                     "scripts/rescore_run.py <run_dir>."),
        }
    else:
        score = final_score(start_eval=start_eval, round_evaluations=all_evaluations,
                            constraints_cfg=constraints_cfg, scoring_cfg=task_cfg["scoring"],
                            reference_cfg=reference_cfg)
        score["scored"] = True

    constraint_details = score.get("constraint_diagnostics_for_best_by_objective") or {}
    constraint_flags = constraint_details.get("constraint_flags") or {}

    report = {
        "success": bool(score.get("success", False)),
        "task_name": task_cfg["task_name"],
        "fixed_iteration_budget": max_rounds,
        "score": score,
        "data_nature": (
            "SYNTHETIC. Every response in this report is the output of a synthetic process "
            "simulator (mechanistic population balance + solubility mass balance + a surrogate "
            "network fitted to the same mechanism with synthetic noise, seed 20260726). There is no "
            "registered product, no filed process and no experimental measurement anywhere in this "
            "benchmark. Scores measure coordinated multi-variable decision-making under a change "
            "budget with narrow-band numerical feedback, and must not be read as evidence about "
            "real spherical agglomeration performance. See data/MODEL_CARD.md."
        ),
        "target": {
            "objective": {k: v for k, v in objective_cfg.items() if k != "windows"},
            "specification_windows": objective_cfg["windows"],
            "baseline_process": START,
            "constraints": constraints_cfg,
        },
        "checks": {
            "n_iterations_completed": len(all_evaluations),
            "n_design_space_members": sum(1 for ev in all_evaluations if ev.design is not None),
            "n_runnable_batches": sum(1 for ev in all_evaluations if ev.valid),
            "n_constraint_valid": sum(1 for ev in all_evaluations
                                      if ev.valid and all(check_constraints(ev, constraints_cfg).values())),
            "n_successful_designs": score.get("num_successful_designs", 0),
            "best_by_objective_satisfies_all_constraints": bool(constraint_flags) and all(constraint_flags.values()),
            "n_parse_failures": parse_failures,
            "n_llm_call_failures": llm_call_failures,
            "n_empty_llm_responses": empty_llm_responses,
            # Rounds lost to a response still truncated after the retry schedule was exhausted.
            # Should be 0; non-zero means the token budget needs raising.
            "n_truncated_llm_responses": truncated_llm_responses,
        },
        # Agent tool interaction, so a reviewer can see the tool layer was actually exercised rather
        # than merely offered.
        "tool_interaction": {
            "tools_offered": sorted(tool_usage) or "see src/tools.TOOL_SPECS",
            "max_tool_turns_per_round": MAX_TOOL_TURNS_PER_ROUND,
            "max_tool_calls_per_turn": MAX_TOOL_CALLS_PER_TURN,
            "n_rounds_using_tools": rounds_using_tools,
            "n_tool_turns": tool_turns_total,
            "n_tool_calls": tool_calls_total,
            "n_tool_errors": tool_errors_total,
            "n_error_then_clean_turns": tool_recoveries,
            "calls_per_tool": tool_usage,
            "note": "No tool returns a quality attribute, a prediction of one, the desirability D, "
                    "an achievement fraction or a score; src/tools.assert_no_objective_leak "
                    "enforces that on every result.",
        },
        "files": {
            "trajectory_file": str(trajectory_path),
            "llm_debug_file": str(llm_debug_path),
            "conversation_markdown_file": str(conversation_markdown_path),
            "run_dir": str(run_dir),
        },
        "baseline_process_evaluation": start_eval.to_dict(),
        "candidates": [ev.to_dict() for ev in all_evaluations],
    }

    write_json(final_report_path, report)
    append_final_report_to_markdown(conversation_markdown_path, report)
    print(f"Run complete. Final report: {final_report_path}")
    print(json.dumps(report["score"], ensure_ascii=False, indent=2))
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
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to benchmark config YAML.")
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
