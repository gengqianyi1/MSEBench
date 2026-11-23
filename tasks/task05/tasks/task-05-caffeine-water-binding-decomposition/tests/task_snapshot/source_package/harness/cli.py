"""Command-line interface for listing, validating, running and grading tasks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys

import yaml
from pathlib import Path

from .api import aggregate_scores, evaluate_existing_submission, find_task, list_tasks
from .config import TaskConfig
from .orchestrator import run_single_task
from .runtime_checks import sha256_file
from .schema_validation import validate_document
from .utils import load_json, save_json


def _project_root(task_dir: Path) -> Path:
    task_dir = Path(task_dir).resolve()
    if task_dir.parent.name != "tasks":
        raise ValueError(f"task must be located below a tasks/ directory: {task_dir}")
    return task_dir.parents[1]


def _artifact_named_in_prompt(artifact: str, prompt: str) -> bool:
    candidates = {artifact}
    if artifact.endswith("/**"):
        candidates.add(artifact[:-2])
        candidates.add(artifact[:-3])
    return any(candidate and candidate in prompt for candidate in candidates)


def _is_harness_owned_artifact(artifact: str) -> bool:
    """Return whether *artifact* is produced by the Harness, not the model.

    The scientific agent writes the final answer and report.  The Harness owns
    the immutable execution manifest and raw xTB evidence tree, so requiring
    the prompt to tell the model to manufacture those files would blur the
    provenance boundary that the validator is intended to protect.
    """
    normalized = str(artifact).replace("\\", "/").rstrip("*").rstrip("/")
    return normalized == "run_manifest.json" or normalized == "xtb_jobs"


def validate_task_package(task_dir: Path) -> list[dict]:
    """Return structured, auditable findings for one publishable task package."""
    task_dir = Path(task_dir).resolve()
    findings: list[dict] = []
    required_files = ("task.yaml", "prompt.md", "tools.json", "rubric.json", "private_grader/grader.py")
    for name in required_files:
        if not (task_dir / name).is_file():
            findings.append({"severity": "ERROR", "code": "MISSING_FILE", "message": name})
    if findings:
        return findings
    try:
        project_root = _project_root(task_dir)
        schema_dir = project_root / "schemas"
        cfg = TaskConfig.from_yaml(task_dir / "task.yaml")
    except Exception as exc:
        return [{"severity": "ERROR", "code": "INVALID_TASK_CONFIG", "message": str(exc)}]

    if task_dir.name != cfg.task_id:
        findings.append({"severity": "ERROR", "code": "TASK_ID_DIRECTORY_MISMATCH", "message": f"directory={task_dir.name}, task_id={cfg.task_id}"})

    try:
        tools_document = load_json(task_dir / cfg.tools_registry_path)
        validate_document(tools_document, schema_dir / "tools.schema.json")
        tools = tools_document.get("tools", {})
        for name in cfg.required_tools:
            spec = tools.get(name)
            if not isinstance(spec, dict):
                findings.append({"severity": "ERROR", "code": "UNDECLARED_TOOL", "message": name})
                continue
            if spec.get("available") is not True:
                findings.append({"severity": "ERROR", "code": "TOOL_MARKED_UNAVAILABLE", "message": name})
            implementation = spec.get("implementation")
            if implementation:
                relative = str(implementation).split("#", 1)[0]
                if not (project_root / relative).is_file():
                    findings.append({"severity": "ERROR", "code": "MISSING_TOOL_IMPLEMENTATION", "message": f"{name}: {relative}"})
    except Exception as exc:
        findings.append({"severity": "ERROR", "code": "INVALID_TOOLS_JSON", "message": str(exc)})

    try:
        rubric = load_json(task_dir / cfg.rubric_path)
        validate_document(rubric, schema_dir / "rubric.schema.json")
        consistency = {
            "task_id": cfg.task_id,
            "task_version": cfg.version,
            "grader_path": cfg.grader_path,
            "grader_entrypoint": cfg.grader_entrypoint,
            "grader_version": cfg.grader_version,
            "required_artifacts": cfg.required_artifacts,
            "score_semantics": cfg.score_semantics,
        }
        for key, expected in consistency.items():
            if rubric.get(key) != expected:
                findings.append({"severity": "ERROR", "code": "RUBRIC_CONTRACT_MISMATCH", "message": f"{key}: expected {expected!r}, found {rubric.get(key)!r}"})
    except Exception as exc:
        findings.append({"severity": "ERROR", "code": "INVALID_RUBRIC", "message": str(exc)})

    if cfg.grader_path and not (task_dir / cfg.grader_path).is_file():
        findings.append({"severity": "ERROR", "code": "MISSING_GRADER", "message": cfg.grader_path})
    private_root = task_dir / cfg.private_data_path if cfg.private_data_path else task_dir
    if cfg.private_data_path and not private_root.is_dir():
        findings.append({"severity": "ERROR", "code": "MISSING_PRIVATE_DATA", "message": cfg.private_data_path})
    for relative in cfg.private_required_files:
        target = private_root / relative
        if not target.is_file():
            findings.append({"severity": "ERROR", "code": "MISSING_PRIVATE_FILE", "message": f"{cfg.private_data_path}/{relative}"})

    prompt = (task_dir / "prompt.md").read_text(encoding="utf-8")
    for artifact in cfg.required_artifacts:
        if _is_harness_owned_artifact(artifact):
            continue
        if not _artifact_named_in_prompt(artifact, prompt):
            findings.append({"severity": "ERROR", "code": "PROMPT_ARTIFACT_MISMATCH", "message": artifact})
    if ", ," in prompt or re.search(r"report\.md.{0,20}(?:optional|可选)|(?:optional|可选).{0,20}report\.md", prompt, re.I):
        findings.append({"severity": "ERROR", "code": "PROMPT_CONTRACT_CONTRADICTION", "message": "malformed or optional report.md wording"})
    if cfg.optional_artifacts:
        findings.append({"severity": "WARNING", "code": "OPTIONAL_ARTIFACTS_DECLARED", "message": ", ".join(cfg.optional_artifacts)})

    # The scientific source task and the private evaluator mirror are separate
    # copies by design; publication requires byte-identical question/config
    # contracts so that the Agent and grader cannot observe different rules.
    public_question = task_dir / "input" / "original" / "question.md"
    public_scientific_config = task_dir / "input" / "original" / "config" / "task.yaml"
    question_mirrors = sorted((private_root / "evaluator_bundle" / "tasks").glob("*/question.md"))
    config_mirrors = sorted((private_root / "evaluator_bundle" / "tasks").glob("*/config/task.yaml"))
    if not public_question.is_file() or len(question_mirrors) != 1:
        findings.append({"severity": "ERROR", "code": "SCIENTIFIC_QUESTION_MIRROR_MISSING", "message": str(public_question)})
    else:
        public_text = public_question.read_text(encoding="utf-8")
        if public_text != question_mirrors[0].read_text(encoding="utf-8"):
            findings.append({"severity": "ERROR", "code": "SCIENTIFIC_QUESTION_MIRROR_MISMATCH", "message": question_mirrors[0].relative_to(task_dir).as_posix()})
        if re.search(r"report\.md.{0,20}(?:optional|可选)|(?:optional|可选).{0,20}report\.md", public_text, re.I):
            findings.append({"severity": "ERROR", "code": "SCIENTIFIC_QUESTION_CONTRACT_CONTRADICTION", "message": "report.md is required by task.yaml"})
    if not public_scientific_config.is_file() or len(config_mirrors) != 1:
        findings.append({"severity": "ERROR", "code": "SCIENTIFIC_CONFIG_MIRROR_MISSING", "message": str(public_scientific_config)})
    else:
        if public_scientific_config.read_bytes() != config_mirrors[0].read_bytes():
            findings.append({"severity": "ERROR", "code": "SCIENTIFIC_CONFIG_MIRROR_MISMATCH", "message": config_mirrors[0].relative_to(task_dir).as_posix()})
        inner = yaml.safe_load(public_scientific_config.read_text(encoding="utf-8")) or {}
        inner_required = {str(item).rstrip("*/\\") for item in inner.get("required_outputs", [])}
        outer_required = {str(item).rstrip("*/\\") for item in cfg.required_artifacts}
        missing_outer = sorted(item for item in inner_required if item not in outer_required)
        if missing_outer:
            findings.append({"severity": "ERROR", "code": "SCIENTIFIC_OUTPUT_CONTRACT_MISMATCH", "message": ", ".join(missing_outer)})

    for name, relative in cfg.manifest_schemas.items():
        if not (project_root / relative).is_file():
            findings.append({"severity": "ERROR", "code": "MISSING_SCHEMA", "message": f"{name}: {relative}"})
    environment = task_dir / str(cfg.environment_file or "")
    if not environment.is_file():
        findings.append({"severity": "ERROR", "code": "MISSING_ENVIRONMENT", "message": str(cfg.environment_file)})
    lock = task_dir / str(cfg.runtime_checks.get("environment_lock", {}).get("path", ""))
    expected_lock = str(cfg.runtime_checks.get("environment_lock", {}).get("sha256", ""))
    if not lock.is_file():
        findings.append({"severity": "ERROR", "code": "MISSING_ENVIRONMENT_LOCK", "message": str(lock.relative_to(task_dir))})
    elif expected_lock and sha256_file(lock) != expected_lock:
        findings.append({"severity": "ERROR", "code": "ENVIRONMENT_LOCK_HASH_MISMATCH", "message": str(lock.relative_to(task_dir))})

    public_roots = [path for path in (task_dir / "input", task_dir / "public") if path.exists()]
    if not public_roots:
        findings.append({"severity": "ERROR", "code": "NO_PUBLIC_INPUT_DIRECTORY", "message": "Expected input/ or public/"})
    elif not any(path.is_file() for root in public_roots for path in root.rglob("*")):
        findings.append({"severity": "ERROR", "code": "EMPTY_PUBLIC_INPUT", "message": "Public input directory contains no files"})

    grader_text = (task_dir / "private_grader" / "grader.py").read_text(encoding="utf-8")
    rubric_text = (task_dir / "rubric.json").read_text(encoding="utf-8")
    if "result.json" in grader_text or "result.json" in rubric_text:
        findings.append({"severity": "ERROR", "code": "TEMPLATE_GRADER_RESIDUE", "message": "result.json template contract remains"})
    if "private_data/evaluator_bundle" not in grader_text.replace('" / "', '/').replace("' / '", "/") and "evaluator_bundle" not in grader_text:
        findings.append({"severity": "ERROR", "code": "GRADER_NOT_DELEGATED", "message": cfg.grader_path or ""})

    if not findings:
        findings.append({"severity": "OK", "code": "VALID", "message": "Task package is structurally and contractually valid"})
    return findings


def cmd_list(args) -> None:
    records = list_tasks(Path(args.task_dir))
    print(f"{'Task ID':<32} {'Version':<10} {'Domain':<22} Title")
    print("-" * 100)
    for record in records:
        print(f"{record['task_id']:<32} {record['version']:<10} {record['domain']:<22} {record['title']}")


def cmd_validate(args) -> None:
    root = Path(args.task_dir)
    if args.task:
        targets = [find_task(args.task, root)]
    else:
        # --all and the legacy no-argument form both validate every task.
        targets = [d for d in sorted(root.iterdir()) if (d / "task.yaml").is_file()]
    error_count = 0
    results = []
    for task_dir in targets:
        findings = validate_task_package(task_dir)
        if args.json:
            results.append({"task": task_dir.name, "findings": findings})
        else:
            print(f"\n[{task_dir.name}]")
        for finding in findings:
            if not args.json:
                print(f"{finding['severity']:<7} {finding['code']}: {finding['message']}")
            error_count += finding["severity"] == "ERROR"
    if args.json:
        print(
            json.dumps(
                {"passed": error_count == 0, "task_count": len(targets), "results": results},
                ensure_ascii=False,
                indent=2,
            )
        )
    if error_count:
        raise SystemExit(1)


def cmd_doctor(args) -> None:
    task_dir = find_task(args.task, Path(args.task_dir))
    cfg = TaskConfig.from_yaml(task_dir / "task.yaml")
    project_root = _project_root(task_dir)
    findings = validate_task_package(task_dir)
    errors = 0
    for finding in findings:
        print(f"{finding['severity']:<7} {finding['code']}: {finding['message']}")
        errors += finding["severity"] == "ERROR"
    python_available = bool(shutil.which(cfg.python_path))
    print(f"{'OK' if python_available else 'ERROR':<7} PYTHON: {cfg.python_path}")
    errors += not python_available
    tools = load_json(task_dir / cfg.tools_registry_path).get("tools", {})
    for name in cfg.required_tools:
        spec = tools.get(name, {})
        kind = spec.get("kind")
        available = spec.get("available") is True
        details: list[str] = [str(kind or "unknown")]
        if kind == "builtin":
            implementation = str(spec.get("implementation", "")).split("#", 1)[0]
            implementation_path = project_root / implementation
            available = available and implementation_path.is_file()
            details.append(implementation)
        elif kind == "binary":
            binary = str(spec.get("binary", ""))
            available = available and bool(shutil.which(binary))
            details.append(binary)
        elif kind == "python":
            import_name = str(spec.get("import_name", ""))
            probe = subprocess.run([cfg.python_path, "-c", f"import {import_name}"], capture_output=True, text=True, timeout=30)
            available = available and probe.returncode == 0
            details.append(import_name)
        for binary in spec.get("requires_binaries", []):
            found = shutil.which(str(binary))
            available = available and bool(found)
            details.append(f"requires:{binary}={found or 'missing'}")
        print(f"{'OK' if available else 'ERROR':<7} TOOL {name}: {'; '.join(details)}")
        errors += not available
    if errors:
        raise SystemExit(1)


def cmd_run(args):
    task_dir = find_task(args.task, Path(args.task_dir))
    blockers = [item for item in validate_task_package(task_dir) if item["severity"] == "ERROR"]
    if blockers:
        for item in blockers:
            print(f"ERROR {item['code']}: {item['message']}")
        raise SystemExit(1)
    return run_single_task(task_dir, Path(args.config), Path(args.output))


def cmd_batch(args) -> None:
    task_root = Path(args.task_dir)
    requested = {value.strip() for value in args.tasks.split(",")} if args.tasks else None
    tasks = [Path(record["task_dir"]) for record in list_tasks(task_root)]
    if requested:
        tasks = [path for path in tasks if path.name in requested or TaskConfig.from_yaml(path / "task.yaml").task_id in requested]
    results = []
    for task_dir in tasks:
        try:
            result = run_single_task(task_dir, Path(args.config), Path(args.output))
            results.append({"task_id": result["task_id"], "status": result["execution_status"], "score": result["score"].get("score")})
        except Exception as exc:
            results.append({"task_id": task_dir.name, "status": "INFRA_ERROR", "score": None, "error": str(exc)})
    save_json(Path(args.output) / "batch_summary.json", {"results": results})


def cmd_evaluate(args) -> None:
    score = evaluate_existing_submission(
        instance_id=args.task,
        submission_dir=Path(args.submission_dir),
        tasks_root=Path(args.task_dir),
        output_dir=Path(args.output) if args.output else None,
    )
    print(json.dumps(score, indent=2, ensure_ascii=False))


def cmd_aggregate(args) -> None:
    summary = aggregate_scores(Path(args.runs_dir), Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemBench Agent Harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list")
    p.add_argument("--task-dir", default="tasks")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("validate")
    p.add_argument("--task", help="Task ID or directory name; omit for all")
    p.add_argument("--all", action="store_true", help="validate every task package under --task-dir")
    p.add_argument("--task-dir", default="tasks")
    p.add_argument("--json", action="store_true", help="emit structured JSON instead of human-readable text")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("doctor")
    p.add_argument("--task", required=True)
    p.add_argument("--task-dir", default="tasks")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run")
    p.add_argument("--task", required=True, help="Task ID, directory name, or path")
    p.add_argument("--task-dir", default="tasks")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("batch")
    p.add_argument("--task-dir", default="tasks")
    p.add_argument("--tasks", help="Comma-separated exact task IDs")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="runs")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("evaluate")
    p.add_argument("--task", required=True, help="Task ID")
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--task-dir", default="tasks")
    p.add_argument("--output")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("aggregate")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output", default="results/aggregate.json")
    p.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
