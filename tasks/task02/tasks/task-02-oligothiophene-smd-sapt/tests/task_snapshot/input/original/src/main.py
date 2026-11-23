from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from evaluator import evaluate_run
from instance_generator import prepare_runtime_instance
from llm_client import call_llm
from markdown_report import write_conversation_markdown
from tool_executor import run_python_tool, timeout_floor_for_script
from stage_validation import validate_stage
from utils import (
    append_jsonl,
    copy_file,
    load_json,
    load_yaml,
    make_run_dir,
    redact_api_config,
    sanitize_run_name_component,
    write_json,
    write_yaml,
)


STAGES: List[Dict[str, Any]] = [
    {
        "id": "stage_A",
        "title": "Preflight, reconstruction, and evidence audit",
        "required": ["stage_A_audit.json", "plan.json"],
        "prompt": (
            "Stage A only: inspect instance.json and every file under seed_evidence; run minimal executable micro-tests; "
            "independently reconstruct and validate the capped oligomer; identify every supported anomaly from file-level evidence without assuming how many exist; "
            "write stage_A_audit.json and plan.json plus the required structure, mapping, accepted-input hashes, and micro-test evidence. "
            "Do not run production formal-solvation, endpoint optimization, or SAPT calculations."
        ),
    },
    {
        "id": "stage_B",
        "title": "Gas conformers and bulk-solvation robustness",
        "required": ["stage_B_bulk.json"],
        "prompt": (
            "Stage B only: use accepted Stage A inputs, generate and deduplicate gas conformers, select R_g, run the common-geometry gas/four-solvent formal calculations, "
            "and perform one evidence-motivated formal sensitivity calculation. Save raw logs, geometry files, hashes, ranking arithmetic, and provenance. "
            "Write stage_B_bulk.json in the canonical schema and run validate_stage_B before expecting Stage C. Do not overwrite Stage A files."
        ),
    },
    {
        "id": "stage_C",
        "title": "Endpoint conformational and local-interaction analysis",
        "required": ["stage_C_local.json"],
        "prompt": (
            "Stage C only: reuse common gas starts for endpoint implicit-solvent optimization; recompute geometry metrics from selected coordinates; "
            "build diverse 1:1 endpoint complexes; use fixed-geometry xTB single points to separate frozen interaction and deformation terms; select representatives; and attempt SAPT within the remaining budget. "
            "Save raw evidence and do not overwrite earlier stage files."
        ),
    },
    {
        "id": "stage_D",
        "title": "Decision and uncertainty",
        "required": ["result.json", "revisions.json"],
        "prompt": (
            "Stage D only: assemble the complete result.json from immutable stage files and raw evidence, validate it against result_schema.json, and write revisions.json. "
            "Distinguish numerical from robust conclusions, answer the scientific-scope questions, and identify one quantitative counterfactual next calculation. "
            "Do not invent missing values or overwrite earlier stages."
        ),
    },
]

STAGE_FILE_NAMES = ["stage_A_audit.json", "stage_B_bulk.json", "stage_C_local.json", "result.json"]


def build_system_message(max_tool_calls: int | None) -> str:
    return (
        "You are an autonomous computational chemistry agent in an evidence-audited staged benchmark. "
        "Use only the provided run_python tool for executable work and work only on the requested stage. "
        "Use benchmark_tools.py for all xTB, PySCF, and Psi4 work; direct access is rejected. "
        "There is no action-count or agent-turn budget; rejected calls remain visible in the audit trace. "
        "A stage is complete only after its required JSON and evidence pass runner validation. Save stage JSON immediately when the stage is complete. "
        "The runner does not create, repair, or downgrade missing final outputs. "
        "Inspect return codes, stderr, created/modified files, and raw outputs after every call. "
        "Earlier stage JSON files become immutable when a later stage begins; corrections belong in revisions.json. "
        "Do not fabricate execution results or treat self-reported JSON as raw evidence."
    )


def public_task_configuration(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return only candidate-visible configuration; hidden generation/truth fields stay runner-side."""
    public: Dict[str, Any] = {
        "benchmark_version": task_cfg.get("benchmark_version"),
        "scoring_policy_id": task_cfg.get("scoring_policy_id"),
        "preferred_tool_call_budget": task_cfg.get("preferred_tool_call_budget"),
        "fixed_tool_call_budget": task_cfg.get("fixed_tool_call_budget"),
        "tool_call_penalties": task_cfg.get("tool_call_penalties", {}),
        "molecule": task_cfg.get("molecule", {}),
        "solvents": task_cfg.get("solvents", {}),
        "methods": task_cfg.get("methods", {}),
        "constants": task_cfg.get("constants", {}),
        "sensitivity_options": task_cfg.get("sensitivity_options", []),
        "resource_guidance": task_cfg.get("resource_guidance", {}),
        "requirements": {
            k: v for k, v in (task_cfg.get("requirements") or {}).items()
            if k != "scoring_reference"
        },
    }
    return public


def build_initial_user_message(question_text: str, task_cfg: Dict[str, Any], public_instance: Dict[str, Any]) -> str:
    return (
        question_text
        + "\n\nPublic task configuration:\n"
        + json.dumps(public_task_configuration(task_cfg), ensure_ascii=False, indent=2)
        + "\n\nRuntime instance summary:\n"
        + json.dumps(public_instance, ensure_ascii=False, indent=2)
        + "\n\nBegin with Stage A only. "
        + STAGES[0]["prompt"]
    )


def stage_validation_errors(workspace: Path, stage_index: int) -> List[str]:
    stage_letter = STAGES[stage_index]["id"].split("_")[-1]
    return validate_stage(workspace, stage_letter)


def stage_complete(workspace: Path, stage_index: int) -> bool:
    return not stage_validation_errors(workspace, stage_index)


def normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if hasattr(tc, "model_dump"):
            normalized.append(tc.model_dump(exclude_none=True))
        else:
            normalized.append(tc)
    return normalized


def resolve_api_config(project_root: Path, configured_value: str | None) -> Path | None:
    if not configured_value:
        return None
    configured = project_root / configured_value
    if configured.exists():
        return configured
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_stage_integrity(path: Path, workspace: Path, tool_index: int, stage_id: str) -> None:
    snapshot: Dict[str, Any] = {}
    for name in STAGE_FILE_NAMES + ["plan.json", "revisions.json"]:
        file_path = workspace / name
        if file_path.exists() and file_path.is_file():
            stat = file_path.stat()
            snapshot[name] = {
                "sha256": _file_sha256(file_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "canonical_json_sha256": _canonical_json_sha256(file_path),
            }
    append_jsonl(
        path,
        {
            "tool_index": tool_index,
            "runner_stage": stage_id,
            "snapshot": snapshot,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--instance-seed", type=int, default=None)
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_yaml(args.config)
    task_dir = project_root / cfg["task_dir"]
    question_path = task_dir / "question.md"
    task_cfg_path = task_dir / "task_config.yaml"
    schema_path = task_dir / "result_schema.json"
    stage_a_schema_path = task_dir / "stage_A_schema.json"
    stage_b_schema_path = task_dir / "stage_B_schema.json"
    stage_c_schema_path = task_dir / "stage_C_schema.json"
    protocol_path = task_dir / "protocol.yaml"
    benchmark_tools_path = task_dir / "benchmark_tools.py"
    stage_validator_path = project_root / "src" / "stage_validation.py"
    question_text = question_path.read_text(encoding="utf-8")
    task_cfg = load_yaml(task_cfg_path)

    candidate_api_path = resolve_api_config(project_root, cfg.get("llm", {}).get("api_config_file"))
    if candidate_api_path is None:
        raise FileNotFoundError("Candidate API config not found. Copy configs/api_config.example.json to configs/api_config.json.")
    candidate_api_cfg = load_json(candidate_api_path)

    evaluator_api_path = resolve_api_config(project_root, cfg.get("evaluation", {}).get("api_config_file"))
    evaluator_api_cfg = load_json(evaluator_api_path) if evaluator_api_path else None

    run_started = time.monotonic()
    max_wall_time_seconds = int(cfg["run"].get("max_wall_time_seconds", 0))
    candidate_model_component = sanitize_run_name_component(candidate_api_cfg.get("model"))
    evaluator_model_component = sanitize_run_name_component((evaluator_api_cfg or {}).get("model"))
    run_name_parts = [cfg["run"]["run_prefix"]]
    if candidate_model_component and candidate_model_component not in sanitize_run_name_component(run_name_parts[0]):
        run_name_parts.append(candidate_model_component)
    if evaluator_model_component:
        run_name_parts.append(f"eval-{evaluator_model_component}")
    run_dir = make_run_dir(cfg["run"]["output_root"], "_".join(run_name_parts))
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    copy_file(question_path, run_dir / "question.md")
    copy_file(task_cfg_path, run_dir / "task_config.yaml")
    copy_file(schema_path, workspace / "result_schema.json")
    copy_file(stage_a_schema_path, workspace / "stage_A_schema.json")
    copy_file(stage_b_schema_path, workspace / "stage_B_schema.json")
    copy_file(stage_c_schema_path, workspace / "stage_C_schema.json")
    copy_file(protocol_path, workspace / "protocol.yaml")
    copy_file(benchmark_tools_path, workspace / "benchmark_tools.py")
    copy_file(stage_validator_path, workspace / "stage_validation.py")
    write_yaml(run_dir / "config_used.yaml", cfg)

    public_instance, instance_truth = prepare_runtime_instance(task_cfg, workspace, seed_override=args.instance_seed)

    safe_candidate = redact_api_config(candidate_api_cfg)
    safe_candidate["api_config_path"] = str(candidate_api_path)
    write_json(run_dir / "candidate_api_config_used.json", safe_candidate)
    if evaluator_api_cfg:
        safe_eval = redact_api_config(evaluator_api_cfg)
        safe_eval["api_config_path"] = str(evaluator_api_path)
        write_json(run_dir / "evaluator_api_config_used.json", safe_eval)

    trajectory_path = run_dir / cfg["logging"]["trajectory_filename"]
    tool_calls_path = run_dir / cfg["logging"]["tool_calls_filename"]
    llm_debug_path = run_dir / cfg["logging"]["llm_debug_filename"]
    conversation_path = run_dir / cfg["logging"]["conversation_markdown_filename"]
    final_answer_path = run_dir / cfg["logging"]["final_answer_filename"]
    final_report_path = run_dir / cfg["logging"]["final_report_filename"]
    stage_integrity_path = run_dir / "stage_integrity.jsonl"

    max_tool_calls = task_cfg.get("fixed_tool_call_budget")
    max_tool_calls = int(max_tool_calls) if max_tool_calls is not None else None
    user_message = build_initial_user_message(question_text, task_cfg, public_instance)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_message(max_tool_calls)},
        {"role": "user", "content": user_message},
    ]
    append_jsonl(trajectory_path, {"turn": 1, "messages_input_to_llm": [{"role": "user", "content": user_message}]})
    write_conversation_markdown(conversation_path, messages)

    configured_turns = cfg["run"].get("max_llm_turns")
    max_turns = int(configured_turns) if configured_turns is not None else 2**31 - 1
    tool_index = 0
    current_stage = 0
    budget_exhausted = False

    for turn in range(1, max_turns + 1):
        elapsed = time.monotonic() - run_started
        if max_wall_time_seconds and elapsed >= max_wall_time_seconds:
            messages.append({"role": "assistant", "content": f"Benchmark wall-time budget exhausted after {elapsed:.1f} seconds."})
            break
        if max_tool_calls is not None and tool_index >= max_tool_calls:
            budget_exhausted = True
            messages.append({"role": "assistant", "content": f"Tool-call budget exhausted at {tool_index} calls."})
            break

        llm_result = call_llm(messages, candidate_api_cfg)
        append_jsonl(llm_debug_path, {"turn": turn, "llm_call": llm_result.raw_response, "ok": llm_result.ok, "error": llm_result.error})
        if not llm_result.ok:
            messages.append({"role": "assistant", "content": f"LLM call failed: {llm_result.error}"})
            break

        assistant_msg = llm_result.message
        messages.append(assistant_msg)
        write_conversation_markdown(conversation_path, messages)
        tool_calls = normalize_tool_calls(assistant_msg.get("tool_calls"))

        if not tool_calls:
            missing = [rel for rel in STAGES[current_stage]["required"] if not (workspace / rel).exists()]
            validation_errors = stage_validation_errors(workspace, current_stage)
            if not validation_errors and current_stage < len(STAGES) - 1:
                current_stage += 1
                next_msg = f"Stage {current_stage + 1}: {STAGES[current_stage]['prompt']}"
            elif not stage_validation_errors(workspace, len(STAGES) - 1):
                break
            else:
                next_msg = (
                    f"No tool call was returned. Continue {STAGES[current_stage]['id']} by calling run_python. "
                    f"Missing required files: {missing}. Validation errors: {validation_errors}. "
                    f"Tool calls used: {tool_index}; action limit: unlimited."
                )
            messages.append({"role": "user", "content": next_msg})
            append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": next_msg}]})
            continue

        for tool_call in tool_calls:
            if max_tool_calls is not None and tool_index >= max_tool_calls:
                budget_exhausted = True
                break
            tool_index += 1
            function = tool_call.get("function") or {}
            name = function.get("name")
            raw_args = function.get("arguments") or "{}"
            try:
                args_obj = json.loads(raw_args)
            except Exception as exc:
                result = {"error": f"Could not parse tool arguments: {exc}", "raw_arguments": raw_args, "returncode": -1}
            else:
                if name != "run_python":
                    result = {"error": f"Unknown tool: {name}", "returncode": -1}
                else:
                    try:
                        script_text = args_obj.get("script", "")
                        requested_timeout = int(args_obj.get("timeout_seconds") or cfg["execution"]["default_script_runtime_seconds"])
                        timeout_class, timeout_floor = timeout_floor_for_script(
                            script_text,
                            cfg["execution"].get("timeout_floor_seconds", {}),
                        )
                        effective_timeout = max(requested_timeout, timeout_floor)
                        if max_wall_time_seconds:
                            remaining = int(max_wall_time_seconds - (time.monotonic() - run_started))
                            if remaining <= 0:
                                raise TimeoutError("Benchmark wall-time budget exhausted before this tool call.")
                            effective_timeout = min(effective_timeout, remaining)
                        result = run_python_tool(
                            script=script_text,
                            timeout_seconds=effective_timeout,
                            workspace=workspace,
                            project_root=project_root,
                            python_executable=cfg["execution"].get("python_executable", "python"),
                            max_script_runtime_seconds=int(cfg["execution"]["max_script_runtime_seconds"]),
                            max_script_size_bytes=int(cfg["execution"]["max_script_size_bytes"]),
                            stdout_stderr_tail_chars=int(cfg["execution"]["stdout_stderr_tail_chars"]),
                            tool_index=tool_index,
                        )
                        result["requested_timeout_seconds"] = requested_timeout
                        result["effective_timeout_seconds"] = effective_timeout
                        result["timeout_class"] = timeout_class
                        result["timeout_floor_seconds"] = timeout_floor
                    except Exception as exc:
                        result = {"error": str(exc), "returncode": -1, "result_json_exists": (workspace / "result.json").exists()}

            append_jsonl(tool_calls_path, {"turn": turn, "stage": STAGES[current_stage]["id"], "tool_call": tool_call, "result": result})
            record_stage_integrity(stage_integrity_path, workspace, tool_index, STAGES[current_stage]["id"])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": json.dumps(result, ensure_ascii=False),
            })
            write_conversation_markdown(conversation_path, messages)

        if budget_exhausted:
            break

        validation_errors = stage_validation_errors(workspace, current_stage)
        if not validation_errors:
            if current_stage < len(STAGES) - 1:
                current_stage += 1
                next_msg = (
                    f"Stage {current_stage + 1}: {STAGES[current_stage]['prompt']} "
                    f"Tool calls used: {tool_index}; action limit: unlimited."
                )
                messages.append({"role": "user", "content": next_msg})
                append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": next_msg}]})
                write_conversation_markdown(conversation_path, messages)
        else:
            required_exist = any((workspace / rel).exists() for rel in STAGES[current_stage]["required"])
            if required_exist:
                next_msg = (
                    f"{STAGES[current_stage]['id']} output exists but did not pass validation. "
                    f"Fix it before advancing. Validation errors: {validation_errors}. "
                    f"Tool calls used: {tool_index}; action limit: unlimited."
                )
                messages.append({"role": "user", "content": next_msg})
                append_jsonl(trajectory_path, {"turn": turn + 1, "messages_input_to_llm": [{"role": "user", "content": next_msg}]})
                write_conversation_markdown(conversation_path, messages)

        stop_on_valid_stage_d = cfg["run"].get(
            "stop_when_stage_D_valid",
            cfg["run"].get("stop_when_result_json_exists", True),
        )
        if stop_on_valid_stage_d and not stage_validation_errors(workspace, len(STAGES) - 1):
            break

    result_path = workspace / "result.json"
    final_stage_errors = stage_validation_errors(workspace, len(STAGES) - 1)
    if result_path.exists():
        try:
            final_result = load_json(result_path)
        except Exception:
            final_result = {}
        write_json(final_answer_path, final_result)
    else:
        final_result = {}

    evaluation = evaluate_run(
        run_dir,
        workspace,
        final_result,
        evaluator_api_cfg=evaluator_api_cfg,
        candidate_model=str(candidate_api_cfg.get("model", "")),
        instance_truth=instance_truth,
        task_cfg=task_cfg,
    )
    final_report = {
        "success": not final_stage_errors,
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "final_answer_file": str(final_answer_path) if result_path.exists() else None,
        "evaluation": evaluation,
        "current_stage": STAGES[current_stage]["id"],
        "tool_calls_used": tool_index,
        "tool_call_budget": max_tool_calls,
        "tool_budget_exhausted": budget_exhausted,
        "elapsed_wall_time_seconds": time.monotonic() - run_started,
        "max_wall_time_seconds": max_wall_time_seconds or None,
    }
    if final_stage_errors:
        final_report["error"] = "Stage D was not completed successfully"
        final_report["final_stage_validation_errors"] = final_stage_errors

    write_json(final_report_path, final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
