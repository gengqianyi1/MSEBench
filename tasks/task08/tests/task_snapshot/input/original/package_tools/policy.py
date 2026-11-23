from __future__ import annotations

from pathlib import Path, PurePosixPath


ROOT_FILES = frozenset(
    {
        ".env.example",
        ".gitignore",
        "LICENSE",
        "LICENSES.md",
        "NOTICE",
        "README.md",
        "TASK.md",
        "environment.yml",
        "requirements.txt",
        "pytest.ini",
        "run_all.ps1",
        "run_all.sh",
    }
)
RECURSIVE_ROOTS = frozenset(
    {
        "baselines",
        "benchmark_case",
        "configs",
        "evaluator",
        "framework",
        "package_tools",
        "prompts",
        "reports",
        "scripts",
        "tests",
        "tools",
        "trajectories",
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "answers",
        "debug",
        "debug_dumps",
        "evaluator_private",
        "htmlcov",
        "outputs",
        "raw_provider_responses",
        "recovery_backup",
        "runs",
        "runs_failed",
        "staging",
        "temporary_validation",
        "venv",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        ".coverage",
        ".env",
        "models.yaml",
        "raw_response.json",
        "raw_responses.json",
        "request_dump.json",
        "response_dump.json",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".bak", ".dump", ".log.bak", ".pyc", ".pyo", ".tmp", ".trace", ".zip"}
)
REQUIRED_REPORTS = frozenset(
    {
        "reports/BENCHMARK_DESIGN_REPORT.md",
        "reports/ORGANIZER_MODEL_COMPARISON.md",
        "reports/ORGANIZER_VALIDATION_REPORT.md",
    }
)
REQUIRED_TRAJECTORY_FILES = frozenset(
    {
        "artifact_manifest.json",
        "dialogue.md",
        "evaluation.json",
        "final_answer.json",
        "metadata.json",
        "trajectory.jsonl",
    }
)

# Gate 8 claims embedded in the committed release snapshot.  The package
# validator independently re-runs these checks; it never treats the claims as
# evidence on their own.
PACKAGE_GATE_CHECKS = (
    "allowlist_staging",
    "flat_root",
    "manifest",
    "clean_extraction",
    "imports",
    "tests",
    "baseline_replay",
    "three_model_replays",
    "secret_scan",
    "absolute_path_scan",
    "forbidden_file_scan",
    "archive_hash",
)


def normalize_relative(path: Path) -> PurePosixPath:
    pure = PurePosixPath(path.as_posix())
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe relative path")
    return pure


def is_allowed(relative: PurePosixPath) -> bool:
    parts = relative.parts
    lowered = tuple(part.casefold() for part in parts)
    if any(part in FORBIDDEN_PARTS for part in lowered):
        return False
    name = parts[-1]
    lowered_name = name.casefold()
    if lowered_name in FORBIDDEN_NAMES:
        return False
    if lowered_name.startswith(".env") and lowered_name != ".env.example":
        return False
    if Path(name).suffix.casefold() in FORBIDDEN_SUFFIXES:
        return False
    if any(part.startswith(".") for part in parts[:-1]):
        return False
    if len(parts) == 1:
        return name in ROOT_FILES
    if parts[0] == "reports":
        return relative.as_posix() in REQUIRED_REPORTS
    if parts[0] == "trajectories":
        if relative.as_posix() == "trajectories/README.md":
            return True
        return len(parts) >= 3 and parts[1] in {"kimi", "deepseek", "glm"}
    return parts[0] in RECURSIVE_ROOTS


def collect_allowlisted_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = normalize_relative(path.relative_to(root))
        if is_allowed(relative):
            files.append(path)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def required_delivery_paths() -> set[str]:
    required = set(ROOT_FILES) | set(REQUIRED_REPORTS)
    required.update(
        {
            "configs/task.yaml",
            "configs/models.example.yaml",
            "configs/tool_contracts.json",
            "configs/oracle_commitment.json",
            "benchmark_case/job_spec.json",
            "benchmark_case/case_manifest.json",
            "framework/agent_runner.py",
            "tools/tool_registry.py",
            "tools/xtb_runner.py",
            "evaluator/evaluate.py",
            "evaluator/scoring.py",
            "baselines/baseline_agent.py",
            "scripts/run_all.py",
            "scripts/package_submission.py",
            "scripts/validate_package.py",
        }
    )
    for model in ("kimi", "deepseek", "glm"):
        required.update(
            {f"trajectories/{model}/{name}" for name in REQUIRED_TRAJECTORY_FILES}
        )
    return required
