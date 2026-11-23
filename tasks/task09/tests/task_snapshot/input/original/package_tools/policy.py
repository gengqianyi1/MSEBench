"""Release allowlist and delivery requirements."""

from __future__ import annotations

from pathlib import Path, PurePath
import re
import stat


EXPECTED_ARCHIVE_FILENAME = (
    "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v2.zip"
)
EXPECTED_SUBMISSION_NAME = (
    "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v2"
)
EXPECTED_TASK_ID = "adaptive_xtb_torsion_profile_agent_v2"
EXPECTED_BENCHMARK_VERSION = "2.0.0"
EXPECTED_CASE_ID = "2_fluorobiphenyl_central_torsion_v1"
EXPECTED_PROTOCOL = "adaptive_torsion_sampling_tool_loop/2.0"
ALLOWED_ROOT_FILES = {
    ".env.example",
    ".gitignore",
    "environment.yml",
    "LICENSE",
    "LICENSES.md",
    "MANIFEST.json",
    "NOTICE",
    "pytest.ini",
    "README.md",
    "requirements.txt",
    "run_all.ps1",
    "run_all.sh",
    "TASK.md",
    "execution_commitment.json",
    "release_commitment.json",
}
ALLOWED_TOP_LEVEL = {
    "baselines",
    "benchmark_case",
    "configs",
    "evaluator",
    "evaluator_private",
    "framework",
    "package_tools",
    "prompts",
    "reports",
    "scripts",
    "tests",
    "tools",
    "trajectories",
}
EXCLUDED_TOP_LEVEL = {
    "audit_tmp",
    ".test_tmp",
    ".pytest_cache",
    "staging",
    "runs",
    "runs_failed",
    "outputs",
    ".git",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".cache",
    ".env",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "venv",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".zip", ".partial", ".tmp", ".bak"}
SOURCE_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".xyz", ".inp",
    ".ps1", ".sh", ".gitignore", ".example", ".mol",
}
FORBIDDEN_RELEASE_NAME_FRAGMENTS = {
    "debug_dump",
    "provider_raw",
    "raw_provider",
    "raw_response",
}
FORBIDDEN_RAW_NATIVE_NAMES = {
    ".xtboptok",
    "attempt_manifest.json",
    "charges",
    "energy",
    "gradient",
    "stderr.log",
    "stdout.log",
    "wbo",
    "xtbopt.log",
    "xtbrestart",
}
SAFE_PROJECTED_NATIVE_LOG_NAMES = frozenset({"stdout.log", "stderr.log"})
SAFE_TRAJECTORY_LABELS = frozenset(
    {"baseline", "kimi", "deepseek", "glm"}
)
SAFE_ATTEMPT_DIRECTORY = re.compile(
    r"batch_[0-9]{2}_angle_[0-9]{3}_attempt_[0-9]{2}"
)

PACKAGE_GATE_CHECKS = (
    "source_validation",
    "integration_tests",
    "security_tests",
    "e2e_tests",
    "secret_scan",
    "old_evidence_preservation",
    "manifest_exact_coverage",
    "flat_root_zip",
    "zip_security",
    "clean_extract",
    "delivery_validation",
    "final_zip_replay",
)


def _safe_projected_native_log(relative: PurePath) -> bool:
    parts = relative.parts
    return bool(
        len(parts) == 6
        and parts[0] == "trajectories"
        and parts[1] in SAFE_TRAJECTORY_LABELS
        and parts[2:4] == ("artifacts", "scan_summaries")
        and SAFE_ATTEMPT_DIRECTORY.fullmatch(parts[4])
        and parts[5] in SAFE_PROJECTED_NATIVE_LOG_NAMES
    )


def _is_link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(path.is_symlink() or attributes & 0x400 or (callable(is_junction) and is_junction()))


def forbidden_release_path_reason(relative: PurePath) -> str | None:
    """Return a stable reason for paths that can never enter a release."""

    parts = tuple(part.casefold() for part in relative.parts)
    if any(
        part.startswith(".env")
        and not (len(parts) == 1 and part == ".env.example")
        for part in parts
    ):
        return "FORBIDDEN_RELEASE_PART"
    if any(part in FORBIDDEN_PARTS for part in parts):
        return "FORBIDDEN_RELEASE_PART"
    if relative.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return "FORBIDDEN_RELEASE_SUFFIX"
    if (
        relative.name.casefold() in FORBIDDEN_RAW_NATIVE_NAMES
        and not _safe_projected_native_log(relative)
    ):
        return "FORBIDDEN_RELEASE_ARTIFACT"
    folded_parts = tuple(part.casefold().replace("-", "_") for part in relative.parts)
    if any(
        fragment in part
        for part in folded_parts
        for fragment in FORBIDDEN_RELEASE_NAME_FRAGMENTS
    ):
        return "FORBIDDEN_RELEASE_ARTIFACT"
    return None


def allowed_release_file(relative: PurePath, *, include_manifest: bool = False) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if len(parts) == 1:
        return (
            relative.name in ALLOWED_ROOT_FILES
            and (include_manifest or relative.name != "MANIFEST.json")
        )
    if parts[0] not in ALLOWED_TOP_LEVEL:
        return False
    if forbidden_release_path_reason(relative) is not None:
        return False
    if _safe_projected_native_log(relative):
        return True
    if relative.suffix.casefold() in SOURCE_SUFFIXES:
        return True
    # Extensionless legal/notices are explicit.
    if relative.suffix == "" and relative.name in {"LICENSE", "NOTICE"}:
        return True
    return False


def collect_allowlisted_files(root: Path) -> list[Path]:
    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for candidate in resolved.rglob("*"):
        relative = candidate.relative_to(resolved)
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if forbidden_release_path_reason(relative) is not None:
            continue
        if relative.as_posix() == "MANIFEST.json":
            continue
        if _is_link_or_reparse(candidate):
            raise RuntimeError(f"REPARSE_OR_LINK_FORBIDDEN:{relative.as_posix()}")
        if candidate.is_file():
            if allowed_release_file(relative):
                files.append(candidate)
            else:
                raise RuntimeError(f"UNEXPECTED_RELEASE_FILE:{relative.as_posix()}")
        elif candidate.exists() and not candidate.is_dir():
            raise RuntimeError(f"SPECIAL_FILE_FORBIDDEN:{relative.as_posix()}")
    return sorted(files, key=lambda item: item.relative_to(resolved).as_posix())


def collect_strict_release_tree(root: Path) -> list[Path]:
    """Collect a staged/delivery tree and reject every non-release member.

    Source collection intentionally ignores caches and other excluded working
    directories. A staged tree has no such exception: anything present would
    become an archive member, so fail closed.
    """

    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for candidate in resolved.rglob("*"):
        relative = candidate.relative_to(resolved)
        if _is_link_or_reparse(candidate):
            raise RuntimeError(f"REPARSE_OR_LINK_FORBIDDEN:{relative.as_posix()}")
        if candidate.is_file():
            reason = forbidden_release_path_reason(relative)
            if reason is not None:
                raise RuntimeError(f"{reason}:{relative.as_posix()}")
            if not allowed_release_file(relative, include_manifest=True):
                raise RuntimeError(f"UNEXPECTED_RELEASE_FILE:{relative.as_posix()}")
            files.append(candidate)
        elif candidate.exists() and not candidate.is_dir():
            raise RuntimeError(f"SPECIAL_FILE_FORBIDDEN:{relative.as_posix()}")
    return sorted(files, key=lambda item: item.relative_to(resolved).as_posix())


def required_delivery_paths() -> set[str]:
    required = {
        "README.md",
        "TASK.md",
        "LICENSE",
        "LICENSES.md",
        "NOTICE",
        ".env.example",
        ".gitignore",
        "environment.yml",
        "requirements.txt",
        "pytest.ini",
        "run_all.ps1",
        "run_all.sh",
        "configs/task.yaml",
        "configs/tool_contracts.json",
        "configs/models.example.yaml",
        "configs/release.json",
        "benchmark_case/reference.xyz",
        "benchmark_case/reference_structure.json",
        "benchmark_case/case_spec.json",
        "evaluator_private/oracle_profile.json",
        "evaluator_private/oracle_commitment.json",
        "evaluator_private/oracle_provenance.json",
        "package_tools/historical_integrity.py",
        "scripts/release_patch_rehearsal.py",
        "scripts/verify_old_evidence.py",
        "reports/INHERITANCE_AUDIT.md",
        "reports/HISTORICAL_RELEASE_INTEGRITY.md",
        "reports/HISTORICAL_RELEASE_INTEGRITY_POST.json",
        "reports/V1_1_PACKAGE_ROOT_CAUSE.md",
        "reports/V1_1_DEEPSEEK_ELIGIBILITY_ROOT_CAUSE.md",
        "reports/V1_1_EXTREMA_BARRIER_SCORING_AUDIT.md",
        "reports/V2_PROTOCOL_CHANGELOG.md",
        "reports/SCIENTIFIC_PROTOCOL.md",
        "reports/SCIENTIFIC_ASSET_VALIDATION.md",
        "reports/BENCHMARK_DESIGN_REPORT.md",
        "reports/RESOURCE_USAGE_COMPARISON.md",
        "reports/RELEASE_PATCH_CHANGELOG.md",
        "reports/PRE_FREEZE_PACKAGE_REHEARSAL.md",
        "reports/EXECUTION_FREEZE_REPORT.md",
        "reports/ORGANIZER_VALIDATION_REPORT.md",
        "reports/ORGANIZER_MODEL_COMPARISON.md",
        "reports/FINAL_ACCEPTANCE_REPORT.md",
        "reports/baseline_live_validation.json",
        "reports/baseline_replay_validation.json",
        "reports/gate1_scientific_smoke.json",
        "reports/oracle_independent_validation.json",
        "reports/pre_freeze_package_rehearsal.json",
        "reports/pre_freeze_package_rehearsal_release_patch_01.json",
        "reports/pre_freeze_package_rehearsal_release_patch_03.json",
        "reports/PROVIDER_PREFLIGHT.json",
        "reports/PROVIDER_PREFLIGHT_RECHARGE_01.json",
        "reports/kimi_formal_validation.json",
        "reports/kimi_formal_validation_recharge_01.json",
        "reports/deepseek_formal_validation.json",
        "reports/deepseek_formal_validation_recharge_01.json",
        "reports/glm_formal_validation.json",
        "reports/glm_formal_validation_recharge_01.json",
        "reports/replay_validation.json",
        "reports/replay_validation_recharge_01.json",
        "execution_commitment.json",
        "release_commitment.json",
    }
    for label in ("baseline", "kimi", "deepseek", "glm"):
        for name in (
            "trajectory.jsonl",
            "final_answer.json",
            "evaluation.json",
            "run_manifest.json",
            "usage_report.json",
        ):
            required.add(f"trajectories/{label}/{name}")
    return required


__all__ = [
    "EXPECTED_ARCHIVE_FILENAME",
    "EXPECTED_BENCHMARK_VERSION",
    "EXPECTED_CASE_ID",
    "EXPECTED_PROTOCOL",
    "EXPECTED_SUBMISSION_NAME",
    "EXPECTED_TASK_ID",
    "PACKAGE_GATE_CHECKS",
    "allowed_release_file",
    "collect_allowlisted_files",
    "collect_strict_release_tree",
    "forbidden_release_path_reason",
    "required_delivery_paths",
]
