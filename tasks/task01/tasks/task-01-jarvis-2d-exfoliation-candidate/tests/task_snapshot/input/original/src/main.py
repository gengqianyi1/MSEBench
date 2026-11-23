from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluator import evaluate_result
from llm_client import call_llm
from markdown_report import write_conversation_markdown
from tool_executor import run_python_tool
from chgnet_stability import run_selected_relaxation
from utils import append_jsonl, copy_file, load_json, load_yaml, make_run_dir, redact_api_config, write_json, write_yaml


def build_system_message() -> str:
    return (
        "You are an autonomous computational materials science and machine-learning agent. "
        "You must solve the benchmark by planning your own workflow and calling the provided tools. "
        "Do not emit standalone Python code in Markdown. Use tool calls. "
        "Write result.json and all required prediction and candidate-generator artifacts in the workspace when the study is complete. "
        "Inspect the public dataset and installed packages before launching model training. "
        "No benchmark helper modules are provided; write any needed workflow code yourself using the available Python packages and command-line tools. "
        "If a required calculation is unavailable or too slow, clearly record the limitation and use only a defensible self-implemented fallback. "
        "Persist independently scoreable artifacts incrementally and before exact-count assertions, so a late failure does not erase completed work. "
        "Be honest about failed calculations and do not fabricate successful external-tool results."
    )


def build_initial_user_message(question_text: str, task_cfg: Dict[str, Any]) -> str:
    return (
        question_text
        + "\n\nTask configuration:\n"
        + json.dumps(task_cfg, ensure_ascii=False, indent=2)
        + "\n\nBegin by inspecting the available Python packages and command-line tools, then plan and execute the study using run_python. Reserve the single CHGNet selection tool for the final five generated candidates. Keep the workflow bounded enough to finish in one benchmark run."
    )


def normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    normalized = []
    for tc in tool_calls or []:
        if hasattr(tc, "model_dump"):
            normalized.append(tc.model_dump(exclude_none=True))
        else:
            normalized.append(tc)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_yaml(args.config)
    task_dir = project_root / cfg["task_dir"]
    question_path = task_dir / "question.md"
    task_cfg_path = task_dir / "task_config.yaml"
    question_text = question_path.read_text(encoding="utf-8")
    task_cfg = load_yaml(task_cfg_path)
    api_cfg = load_json(cfg["llm"]["api_config_file"])

    run_dir = make_run_dir(cfg["run"]["output_root"], cfg["run"]["run_prefix"])
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy only public task data into the agent workspace. Private labels must be
    # supplied later to an isolated scorer and must never be present here.
    public_data_dir = project_root / "data"
    workspace_data_dir = workspace / "data"
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.json.gz", "validation.json.gz", "test_public.json.gz", "dataset_metadata.json", "candidate_generation_constraints.json"):
        source = public_data_dir / name
        if source.exists():
            copy_file(source, workspace_data_dir / name)

    workspace_schema_dir = workspace / "schemas"
    workspace_schema_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "result_schema.json",
        "candidate_record_schema.json",
        "candidate_summary_schema.json",
        "generation_manifest_schema.json",
    ):
        source = task_dir / name
        if source.exists():
            copy_file(source, workspace_schema_dir / name)

    candidate_spec = project_root / "CANDIDATE_SCORING_SPEC.md"
    if candidate_spec.exists():
        copy_file(candidate_spec, workspace / "CANDIDATE_SCORING_SPEC.md")

    trajectory_path = run_dir / cfg["logging"]["trajectory_filename"]
    tool_calls_path = run_dir / cfg["logging"]["tool_calls_filename"]
    llm_debug_path = run_dir / cfg["logging"]["llm_debug_filename"]
    conversation_path = run_dir / cfg["logging"]["conversation_markdown_filename"]
    final_answer_path = run_dir / cfg["logging"]["final_answer_filename"]
    final_report_path = run_dir / cfg["logging"]["final_report_filename"]

    copy_file(question_path, run_dir / "question.md")
    copy_file(task_cfg_path, run_dir / "task_config.yaml")
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_json(run_dir / "api_config_used.json", redact_api_config(api_cfg))

    user_message = build_initial_user_message(question_text, task_cfg)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": user_message},
    ]
    append_jsonl(trajectory_path, {"turn": 1, "messages_input_to_llm": [{"role": "user", "content": user_message}]})
    write_conversation_markdown(conversation_path, messages)

    configured_turns = cfg["run"].get("max_llm_turns")
    configured_calls = cfg["run"].get("max_tool_calls")
    max_turns = int(configured_turns) if configured_turns is not None else 2**31 - 1
    max_tool_calls = int(configured_calls) if configured_calls is not None else 2**31 - 1
    repair_attempts = int(cfg["run"].get("public_validation_repair_attempts", 0))
    reserved_repair_tool_calls = int(cfg["run"].get("reserved_repair_tool_calls", 0))
    total_tool_call_budget = max_tool_calls + reserved_repair_tool_calls
    total_turn_budget = max_turns + repair_attempts
    repair_attempts_used = 0
    repair_mode = False
    tool_index = 0
    chgnet_tool_attempts = 0
    final_result: Dict[str, Any] | None = None
    last_public_evaluation: Dict[str, Any] | None = None

    for turn in range(1, total_turn_budget + 1):
        llm_result = call_llm(messages, api_cfg)
        append_jsonl(llm_debug_path, {"turn": turn, "llm_call": llm_result.raw_response, "ok": llm_result.ok, "error": llm_result.error})
        if not llm_result.ok:
            messages.append({"role": "assistant", "content": f"LLM call failed: {llm_result.error}"})
            break

        assistant_msg = llm_result.message
        messages.append(assistant_msg)
        write_conversation_markdown(conversation_path, messages)

        tool_calls = normalize_tool_calls(assistant_msg.get("tool_calls"))
        if not tool_calls:
            if (workspace / "result.json").exists():
                break
            messages.append({
                "role": "user",
                "content": "No tool call was returned and result.json is not present. Continue by calling run_python.",
            })
            append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": messages[-1]["content"]}]})
            continue

        for tool_call in tool_calls:
            active_tool_budget = total_tool_call_budget if repair_mode else max_tool_calls
            if tool_index >= active_tool_budget:
                messages.append({"role": "user", "content": "The tool-call budget is exhausted. Finalize using existing artifacts; no further tool calls are permitted."})
                break
            tool_index += 1
            function = tool_call.get("function") or {}
            name = function.get("name")
            raw_args = function.get("arguments") or "{}"
            try:
                args_obj = json.loads(raw_args)
            except Exception as exc:
                result = {"error": f"Could not parse tool arguments: {exc}", "raw_arguments": raw_args}
            else:
                if name == "relax_selected_candidates_chgnet":
                    chgnet_tool_attempts += 1
                    if chgnet_tool_attempts > 1:
                        result = {
                            "error": "Only one CHGNet relaxation attempt is allowed; this attempt was not executed.",
                            "returncode": -1,
                            "chgnet_tool_attempts": chgnet_tool_attempts,
                        }
                    else:
                        try:
                            result = run_selected_relaxation(
                                candidate_ids=args_obj.get("candidate_ids") or [],
                                candidate_path=workspace / "candidate_library.json.gz",
                                constraints_path=workspace / "data" / "candidate_generation_constraints.json",
                                output_dir=workspace / "selected_chgnet",
                            )
                            result["returncode"] = 0
                            result["chgnet_tool_attempts"] = chgnet_tool_attempts
                        except Exception as exc:
                            result = {
                                "error": str(exc),
                                "returncode": -1,
                                "chgnet_tool_attempts": chgnet_tool_attempts,
                            }
                elif name != "run_python":
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = run_python_tool(
                            script=args_obj.get("script", ""),
                            timeout_seconds=int(args_obj.get("timeout_seconds") or cfg["execution"]["default_script_runtime_seconds"]),
                            workspace=workspace,
                            project_root=project_root,
                            python_executable=cfg["execution"].get("python_executable", "python"),
                            max_script_runtime_seconds=int(cfg["execution"]["max_script_runtime_seconds"]),
                            max_script_size_bytes=int(cfg["execution"]["max_script_size_bytes"]),
                            stdout_stderr_tail_chars=int(cfg["execution"]["stdout_stderr_tail_chars"]),
                            tool_index=tool_index,
                        )
                    except Exception as exc:
                        result = {"error": str(exc), "returncode": -1, "result_json_exists": (workspace / "result.json").exists()}

            append_jsonl(tool_calls_path, {"turn": turn, "tool_call": tool_call, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": json.dumps(result, ensure_ascii=False),
            })
            write_conversation_markdown(conversation_path, messages)

        if cfg["run"].get("stop_when_result_json_exists", True) and (workspace / "result.json").exists():
            try:
                candidate_result = load_json(workspace / "result.json")
                last_public_evaluation = evaluate_result(candidate_result, workspace=workspace)
            except Exception as exc:
                last_public_evaluation = {
                    "passed": False,
                    "hard_gate_passed": False,
                    "errors": [f"could not evaluate result.json: {exc}"],
                    "score": 0,
                    "max_score": 0,
                }
            if last_public_evaluation.get("passed"):
                break
            if repair_attempts_used < repair_attempts and tool_index < total_tool_call_budget:
                repair_attempts_used += 1
                repair_mode = True
                feedback = {
                    "public_validator_passed": False,
                    "errors": last_public_evaluation.get("errors", []),
                    "warnings": last_public_evaluation.get("warnings", []),
                    "evidence": last_public_evaluation.get("evidence", {}),
                    "repair_attempts_remaining": repair_attempts - repair_attempts_used,
                    "reserved_repair_tool_calls_remaining": total_tool_call_budget - tool_index,
                }
                messages.append({
                    "role": "user",
                    "content": (
                        "Public validator failed on the current result.json. "
                        "Fix only public/schema/artifact issues; no private labels are available. "
                        + json.dumps(feedback, ensure_ascii=False, indent=2)
                    ),
                })
                append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": messages[-1]["content"]}]})
                write_conversation_markdown(conversation_path, messages)
                continue
            break
        if tool_index >= max_tool_calls and not (workspace / "result.json").exists():
            if repair_attempts_used < repair_attempts and tool_index < total_tool_call_budget:
                repair_attempts_used += 1
                repair_mode = True
                messages.append({
                    "role": "user",
                    "content": (
                        "The primary tool-call budget is exhausted without result.json. "
                        "Use the reserved repair call only to preserve existing predictions, candidate records, "
                        "generator files, failure metadata, and a schema-valid result.json. Do not start a new search."
                    ),
                })
                append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": messages[-1]["content"]}]})
                write_conversation_markdown(conversation_path, messages)
                continue
            break
        if tool_index >= max_tool_calls - 2 and not (workspace / "result.json").exists():
            messages.append({
                "role": "user",
                "content": (
                    f"Only {max_tool_calls - tool_index} run_python calls remain and result.json is still missing. "
                    "Stop exploratory work unless it is essential; use the remaining calls to create predictions.csv, "
                    "required model and candidate-generator artifacts, candidate_library.json.gz, and a schema-valid result.json."
                ),
            })
            append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": messages[-1]["content"]}]})
            write_conversation_markdown(conversation_path, messages)
        if tool_index >= total_tool_call_budget:
            break

    result_path = workspace / "result.json"
    if result_path.exists():
        try:
            final_result = load_json(result_path)
        except Exception as exc:
            evaluation = {
                "passed": False,
                "hard_gate_passed": False,
                "errors": [f"result.json is not parseable JSON: {exc}"],
                "score": 0,
                "max_score": 0,
            }
            final_report = {
                "success": False,
                "run_status": "completed",
                "submission_status": "invalid_result_json",
                "score_eligible": False,
                "run_dir": str(run_dir),
                "workspace": str(workspace),
                "tool_calls_used": tool_index,
                "chgnet_tool_attempts": chgnet_tool_attempts,
                "primary_tool_call_budget": max_tool_calls,
                "reserved_repair_tool_call_budget": reserved_repair_tool_calls,
                "tool_call_budget": total_tool_call_budget,
                "public_validation_repair_attempts_used": repair_attempts_used,
                "evaluation": evaluation,
            }
        else:
            write_json(final_answer_path, final_result)
            evaluation = last_public_evaluation or evaluate_result(final_result, workspace=workspace)
            final_report = {
                "success": bool(evaluation.get("passed")),
                "run_status": "completed",
                "submission_status": "valid" if evaluation.get("passed") else "valid_with_public_errors",
                "score_eligible": bool(evaluation.get("score_eligible")),
                "run_dir": str(run_dir),
                "workspace": str(workspace),
                "final_answer_file": str(final_answer_path),
                "tool_calls_used": tool_index,
                "chgnet_tool_attempts": chgnet_tool_attempts,
                "primary_tool_call_budget": max_tool_calls,
                "reserved_repair_tool_call_budget": reserved_repair_tool_calls,
                "tool_call_budget": total_tool_call_budget,
                "public_validation_repair_attempts_used": repair_attempts_used,
                "evaluation": evaluation,
            }
    else:
        final_report = {
            "success": False,
            "run_status": "run_failed",
            "submission_status": "missing_result_json",
            "score_eligible": False,
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "error": "workspace/result.json was not produced",
            "tool_calls_used": tool_index,
            "chgnet_tool_attempts": chgnet_tool_attempts,
            "primary_tool_call_budget": max_tool_calls,
            "reserved_repair_tool_call_budget": reserved_repair_tool_calls,
            "tool_call_budget": total_tool_call_budget,
        }

    write_json(final_report_path, final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
