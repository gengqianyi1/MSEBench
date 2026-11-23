from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator import (FormulationEvaluation, check_constraints, evaluate_design,
                       evaluate_reference_start)
from feedback import generate_feedback
from formulation import LEVELS, SLOTS, SLOT_UNITS, START
from llm_client import call_llm
from markdown_report import (append_final_report_to_markdown, append_round_to_markdown_report,
                             initialize_markdown_report)
from parser import ParseError, parse_turn
from scoring import final_score, select_best_by_objective
from tools import (DEFAULT_MAX_TOOL_TURNS_PER_ROUND, DEFAULT_TOOL_BUDGET_PER_ROUND, TOOL_NAMES,
                   run_tool_calls, tool_catalogue)
from utils import (append_jsonl, copy_file, load_json, load_yaml, make_run_dir, read_text, set_seed,
                   write_json)


def _slot_table() -> str:
    lines = []
    for s in SLOTS:
        levels = ", ".join(str(v) for v in LEVELS[s])
        lines.append(f"  {s} ({SLOT_UNITS[s]}): allowed values = [{levels}]")
    return "\n".join(lines)


def _tool_block(task_cfg: Dict[str, Any]) -> str:
    """The tool contract, rendered from src/tools.tool_catalogue() so the prompt cannot drift from
    the implementation."""
    tcfg = task_cfg.get("tools") or {}
    budget = int(tcfg.get("call_budget_per_round", DEFAULT_TOOL_BUDGET_PER_ROUND))
    turns = int(tcfg.get("max_tool_turns_per_round", DEFAULT_MAX_TOOL_TURNS_PER_ROUND))
    lines = [json.dumps(tool_catalogue(), ensure_ascii=False, indent=2)]
    return f"""
## Optional diagnostic tools

Before you commit a formulation you MAY call diagnostic tools. Calling them is entirely your
decision -- a round with no tool call is perfectly valid, and so is spending the whole budget.

To call tools, reply with this and nothing else:

```json
{{"tool_calls": [{{"tool": "validate_formulation", "arguments": {{"formulation": {{"T1_mcc_grade": "PH102", "...": "..."}}}}}}]}}
```

To submit the round's formulation, reply with the `candidate` object described below instead.

Budget per round: at most {budget} tool calls, spread over at most {turns} tool-request replies.
After that you must submit. Tool errors come back as
`{{"ok": false, "error": ..., "detail": ..., "recoverable": ...}}`; fix the call and retry, or stop
calling and submit.

**No tool measures a formulation.** None of them returns a CQA value, a desirability, a score, or a
verdict on the specification windows. They only re-state legality and composition arithmetic you
could do yourself, plus generic excipient cautions. Submitting a formulation remains the only way to
learn its five CQAs.

{lines[0]}
""".strip()


def build_initial_messages(question_md: str, task_cfg: Dict[str, Any],
                           start_eval: FormulationEvaluation) -> List[Dict[str, str]]:
    system_msg = (
        "You are a pharmaceutical formulation and process design agent working on an "
        "immediate-release direct-compression tablet. You must redesign the starting formulation "
        "under a limited-change budget so that five critical quality attributes all sit comfortably "
        "inside their specification windows at once. A formulation with ANY critical quality "
        "attribute outside its window is not releasable and earns nothing, whatever its aggregate "
        "desirability. "
        "Each reply is exactly one JSON object: either a `tool_calls` request or a `candidate` "
        "submission. "
        "Keep the rationale concise. "
        "Do not include markdown, comments, or extra text outside the JSON object."
    )

    start_summary = {
        "baseline_formulation": START,
        "composition_mg_per_200mg_core": start_eval.composition_mg,
        "qs_filler_pct_of_core": start_eval.qs_filler_pct,
        "measured_cqa": start_eval.cqa,
    }

    user_msg = f"""
{question_md}

## Baseline Formulation (your starting point)

{json.dumps(start_summary, ensure_ascii=False, indent=2)}

## The 12 variables you may set, and their allowed values

{_slot_table()}

Every submission must give all twelve variables. Values must be taken from the lists above; no new
excipient and no value outside a list is admissible.

{_tool_block(task_cfg)}

Now work on round 1: either call diagnostic tools, or submit the round-1 formulation.
Remember: output exactly one JSON object.
""".strip()

    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_next_user_message(round_idx: int, candidate: Optional[Dict[str, Any]],
                            evaluation: FormulationEvaluation, feedback: Dict[str, Any],
                            start_eval: FormulationEvaluation,
                            best_so_far: Optional[FormulationEvaluation]) -> str:
    return f"""
Round {round_idx} evaluation result:

Submitted formulation:
{json.dumps(candidate, ensure_ascii=False, indent=2) if candidate else "Candidate could not be parsed."}

Evaluator feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Baseline formulation CQAs, for reference:
{json.dumps(start_eval.cqa, ensure_ascii=False)}

Best desirability D reached so far in this run:
{best_so_far.desirability if best_so_far is not None and best_so_far.valid else "None"}

Now work on round {round_idx + 1}: either call diagnostic tools again (fresh budget), or submit the
round-{round_idx + 1} formulation. Return exactly one JSON object and no extra text.
The evaluator reports measured CQA values only; you must check the specification windows and your
own change budget against the task description yourself and judge your own progress. Remember that a
formulation with any CQA outside its window is not releasable and scores nothing.
""".strip()


def build_tool_result_message(round_idx: int, batch: Dict[str, Any], turns_left: int) -> str:
    must_submit = ("You have no tool-request replies left for this round -- your next reply MUST be "
                   "a `candidate` submission." if turns_left <= 0 else
                   f"You may make {turns_left} more tool-request reply(ies) this round, with "
                   f"{batch['budget_left']} tool call(s) of budget left, or submit now.")
    return f"""
Round {round_idx} tool results:

{json.dumps(batch["results"], ensure_ascii=False, indent=2)}

Tool calls executed this reply: {batch["n_executed"]}. Remaining tool-call budget this round: {batch["budget_left"]}.
{must_submit}
Return exactly one JSON object and no extra text.
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
    objective_cfg = task_cfg["objective"]
    constraints = task_cfg["constraints"]

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
        raise RuntimeError(f"Baseline formulation is not evaluable: {start_eval.error}")

    initialize_markdown_report(conversation_markdown_path,
                               title="Tablet Microenvironment CQA Co-Design Benchmark Conversation",
                               task_cfg=task_cfg, api_cfg=llm_cfg)

    messages = build_initial_messages(question_md, task_cfg, start_eval)

    all_evaluations: List[FormulationEvaluation] = []
    best_so_far: Optional[FormulationEvaluation] = None
    parse_failures = 0
    llm_call_failures = 0
    empty_llm_responses = 0
    truncated_llm_responses = 0

    max_rounds = int(task_cfg["iteration"]["max_rounds"])
    tools_cfg = task_cfg.get("tools") or {}
    tools_enabled = bool(tools_cfg.get("enabled", True))
    tool_budget_per_round = int(tools_cfg.get("call_budget_per_round",
                                             DEFAULT_TOOL_BUDGET_PER_ROUND))
    max_tool_turns = int(tools_cfg.get("max_tool_turns_per_round",
                                       DEFAULT_MAX_TOOL_TURNS_PER_ROUND)) if tools_enabled else 0
    tool_stats: Dict[str, Any] = {"n_tool_calls": 0, "n_tool_errors": 0, "by_tool": {},
                                  "by_error": {}, "n_rounds_with_tool_use": 0,
                                  "n_rounds_recovered_after_tool_error": 0,
                                  "n_tool_requests_after_budget_exhausted": 0}

    for round_idx in range(1, max_rounds + 1):
        # ---------------------------------------------------------------------------------------
        # One round = an inner agent loop. The model decides each turn whether to call tools or to
        # commit a formulation; the harness never calls a tool on its own behalf (audit finding #5).
        # It gets at most `max_tool_turns` tool-request turns and `tool_budget_per_round` calls, so a
        # round costs at most 1 + max_tool_turns chat-completions calls.
        # ---------------------------------------------------------------------------------------
        budget_left = tool_budget_per_round if tools_enabled else 0
        turns_left = max_tool_turns
        tool_interactions: List[Dict[str, Any]] = []
        candidate = None
        parse_error = None
        evaluation: Optional[FormulationEvaluation] = None
        llm_result = None
        step = 0
        round_prompt_snapshot = [m for m in messages if m.get("role") == "user"][-1:]
        had_tool_error = False
        recovered = False

        while True:
            step += 1
            prompt_messages = list(messages)
            llm_result = call_llm(prompt_messages, llm_cfg)
            raw_response = llm_result.content
            assistant_message = {
                "role": "assistant",
                "content": raw_response if raw_response else json.dumps(
                    {"candidate": {"name": "llm_empty_response", "rationale":
                                   "LLM returned empty content.", "formulation": {}}}),
            }
            append_jsonl(llm_debug_path, {"round": round_idx, "step": step,
                                          "llm_call": llm_result.to_dict()})
            if not llm_result.ok:
                llm_call_failures += 1
            if llm_result.empty_content:
                empty_llm_responses += 1
            if getattr(llm_result, "truncated", False):
                truncated_llm_responses += 1

            if not llm_result.ok:
                parse_failures += 1
                parse_error = llm_result.error or "LLM call failed or returned empty content."
                evaluation = FormulationEvaluation(
                    error=f"LLM failure before parsing: {parse_error}")
                break

            try:
                turn = parse_turn(
                    raw_response,
                    allow_markdown_fence=cfg["parsing"].get("allow_markdown_fence", True))
            except ParseError as exc:
                parse_failures += 1
                parse_error = str(exc)
                evaluation = FormulationEvaluation(error=f"Parse failure: {parse_error}")
                break

            if turn["kind"] == "candidate":
                candidate = turn["candidate"]
                evaluation = evaluate_design(candidate["formulation"], objective_cfg)
                break

            # --- the model asked for tools -----------------------------------------------------
            if turns_left <= 0 or not tools_enabled:
                # It was told to submit and asked for tools anyway. That costs it the round: no
                # formulation is submitted, exactly as an unparseable reply would.
                tool_stats["n_tool_requests_after_budget_exhausted"] += 1
                parse_failures += 1
                parse_error = ("model requested tools after its tool budget for this round was "
                               "exhausted and it was told to submit; no formulation submitted")
                evaluation = FormulationEvaluation(error=f"Protocol failure: {parse_error}")
                messages.append(assistant_message)
                break

            batch = run_tool_calls(turn["tool_calls"], constraints, budget_left)
            budget_left = batch["budget_left"]
            turns_left -= 1
            for res in batch["results"]:
                tool_stats["n_tool_calls"] += 1
                name = res.get("tool") or "(unnamed)"
                tool_stats["by_tool"][name] = tool_stats["by_tool"].get(name, 0) + 1
                if not res.get("ok", False):
                    tool_stats["n_tool_errors"] += 1
                    kind = res.get("error", "unknown")
                    tool_stats["by_error"][kind] = tool_stats["by_error"].get(kind, 0) + 1
                    had_tool_error = True
                elif had_tool_error:
                    recovered = True
            tool_interactions.append({
                "step": step, "requested": turn["tool_calls"], "results": batch["results"],
                "n_executed": batch["n_executed"], "budget_left": budget_left,
                "tool_turns_left": turns_left,
            })
            messages.append(assistant_message)
            messages.append({"role": "user",
                             "content": build_tool_result_message(round_idx, batch, turns_left)})

        if tool_interactions:
            tool_stats["n_rounds_with_tool_use"] += 1
        if recovered:
            tool_stats["n_rounds_recovered_after_tool_error"] += 1

        assert evaluation is not None
        all_evaluations.append(evaluation)
        if evaluation.valid:
            best_so_far = select_best_by_objective(all_evaluations)

        feedback = generate_feedback(evaluation=evaluation, constraints=constraints,
                                    best_so_far=best_so_far)

        # The trajectory file is a deliverable: it records what was submitted and what the evaluator
        # measured, plus the raw response, every tool call with its return, and any parse error, so a
        # reviewer can replay a round -- including the tool decisions -- without the LLM debug log.
        record = {
            "round": round_idx,
            "messages_input_to_llm": round_prompt_snapshot,
            "n_llm_calls_this_round": step,
            "tool_interactions": tool_interactions,
            "raw_llm_response": llm_result.content if llm_result else None,
            "parse_error": parse_error,
            "submitted_candidate": candidate,
            "evaluation": evaluation.to_dict(),
            "feedback_returned": feedback,
        }
        append_jsonl(trajectory_path, record)
        append_round_to_markdown_report(
            conversation_markdown_path, round_idx=round_idx,
            messages_input_to_llm=list(messages),
            llm_result=llm_result, raw_response=llm_result.content if llm_result else None,
            parsed_candidate=candidate,
            parse_error=parse_error, evaluation=evaluation, feedback=feedback,
            best_so_far=best_so_far, tool_interactions=tool_interactions)

        if round_idx < max_rounds:
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant",
                                 "content": llm_result.content if (llm_result and llm_result.content)
                                 else "(no content)"})
            messages.append({"role": "user", "content": build_next_user_message(
                round_idx=round_idx, candidate=candidate, evaluation=evaluation, feedback=feedback,
                start_eval=start_eval, best_so_far=best_so_far)})

    # ------------------------------------------------------------------------------------------
    # Scoring. The PUBLIC package carries no reference optimum (common review point A: the reference
    # design and the band anchors must not sit on the model-visible side), so a public run records
    # every submission and every evaluation and leaves the arithmetic to the grader:
    #     $PY scripts/grade_run.py <run_dir>          # run inside the grader package
    # The grader package carries the reference block, so a run there is scored in place, and both
    # paths produce the same number from the same stored submissions -- the score is a pure function
    # of what was submitted.
    # ------------------------------------------------------------------------------------------
    reference_cfg = task_cfg.get("reference") or {}
    if reference_cfg.get("best_objective_in_budget") is None:
        score = {
            "ruler_version": str(task_cfg.get("ruler_version", "unversioned")),
        # What the model was SHOWN, as distinct from how it was scored. A trajectory recorded under
        # an older prompt_version is evidence about a task the package no longer poses.
        "prompt_version": int(task_cfg.get("prompt_version", 1)),
            "scored_by": "grader",
            "total_score": None,
            "note": ("This package is the PUBLIC side and deliberately carries no reference optimum, "
                     "so it cannot compute a score. Every submission and every evaluation is "
                     "recorded above and in trajectory.jsonl. Score this run inside the grader "
                     "package with: python scripts/grade_run.py <this run directory>"),
            "reference_block_present": False,
        }
        print("Run complete (unscored: public package). Grade it with scripts/grade_run.py "
              "inside the grader package.")
    else:
        score = final_score(start_eval=start_eval, round_evaluations=all_evaluations,
                            constraints=constraints, scoring_cfg=task_cfg["scoring"],
                            reference_cfg=reference_cfg,
                            release_gate_cfg=task_cfg.get("release_gate"))

    constraint_details = score.get("constraint_diagnostics_for_best_by_objective") or {}
    constraint_flags = constraint_details.get("constraint_flags") or {}

    report = {
        "success": bool(score.get("success", False)),
        "ruler_version": str(task_cfg.get("ruler_version", "unversioned")),
        # What the model was SHOWN, as distinct from how it was scored. A trajectory recorded under
        # an older prompt_version is evidence about a task the package no longer poses.
        "prompt_version": int(task_cfg.get("prompt_version", 1)),
        "data_nature": "synthetic / simulated -- see data/DATA_CARD.md",
        "task_name": task_cfg["task_name"],
        "fixed_iteration_budget": max_rounds,
        "score": score,
        "target": {
            "objective": {k: v for k, v in objective_cfg.items() if k != "windows"},
            "specification_windows": objective_cfg["windows"],
            "baseline_formulation": START,
            "constraints": constraints,
        },
        "checks": {
            "n_iterations_completed": len(all_evaluations),
            "n_design_space_members": sum(1 for ev in all_evaluations if ev.design is not None),
            "n_manufacturable": sum(1 for ev in all_evaluations if ev.valid),
            "n_constraint_valid": sum(1 for ev in all_evaluations
                                      if ev.valid and all(check_constraints(ev, constraints).values())),
            "n_successful_designs": score.get("num_successful_designs", 0),
            "best_by_objective_satisfies_all_constraints": bool(constraint_flags) and all(constraint_flags.values()),
            "n_parse_failures": parse_failures,
            "n_llm_call_failures": llm_call_failures,
            "n_empty_llm_responses": empty_llm_responses,
            # Rounds lost to a response still truncated after the retry schedule was exhausted.
            # Should be 0; non-zero means the token budget needs raising.
            "n_truncated_llm_responses": truncated_llm_responses,
            # Tool-layer coverage (audit finding #5). n_tool_calls counts every call the MODEL chose
            # to make; n_tool_errors counts the structured failures it got back;
            # n_rounds_recovered_after_tool_error counts rounds where a failed call was followed by a
            # successful one in the same round, i.e. the model actually recovered rather than giving
            # up. n_tool_requests_after_budget_exhausted counts rounds it lost to ignoring the quota.
            "tool_layer": tool_stats,
        },
        "files": {
            "trajectory_file": str(trajectory_path),
            "llm_debug_file": str(llm_debug_path),
            "conversation_markdown_file": str(conversation_markdown_path),
            "run_dir": str(run_dir),
        },
        "baseline_formulation_evaluation": start_eval.to_dict(),
        "candidates": [ev.to_dict() for ev in all_evaluations],
    }

    write_json(final_report_path, report)
    append_final_report_to_markdown(conversation_markdown_path, report)
    print(f"Run complete. Final report: {final_report_path}")
    print(json.dumps({k: v for k, v in report["score"].items()
                      if k not in ("continuous_metrics", "release_gate",
                                   "best_by_objective_design", "best_successful_design",
                                   "constraint_diagnostics_for_best_by_objective")},
                     ensure_ascii=False, indent=2))
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
