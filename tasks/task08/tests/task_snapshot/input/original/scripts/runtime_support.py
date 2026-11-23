"""Shared, non-model runtime helpers for isolated runs and frozen promotion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CASE_TEMPLATE = ROOT / "benchmark_case"
TRAJECTORY_REQUIRED = (
    "trajectory.jsonl",
    "dialogue.md",
    "metadata.json",
    "final_answer.json",
    "evaluation.json",
)


class RuntimeIntegrityError(RuntimeError):
    """An immutable run, case, or promotion invariant was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def pre_model_source_fingerprint(root: Path = ROOT) -> str:
    """Bind a gate receipt to immutable pre-model source and evidence bytes.

    Formal trajectories and generated reports are deliberately excluded so an
    already-approved runner can execute the remaining models without making
    its own receipt stale.  The receipt file itself is excluded to avoid a
    self-reference.
    """

    from package_tools.policy import collect_allowlisted_files

    resolved = root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for path in collect_allowlisted_files(resolved):
        relative = path.relative_to(resolved).as_posix()
        if (
            relative == "configs/release.json"
            or relative.startswith("reports/")
            or relative.startswith("trajectories/")
        ):
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise RuntimeIntegrityError("PRE_MODEL_SOURCE_SET_EMPTY")
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeIntegrityError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrityError("required JSON is unreadable") from exc


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    reparse = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def regular_files(root: Path) -> list[Path]:
    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for candidate in resolved.rglob("*"):
        if _is_reparse_or_symlink(candidate):
            raise RuntimeIntegrityError("reparse points are forbidden in run artifacts")
        if candidate.is_file():
            files.append(candidate)
        elif candidate.exists() and not candidate.is_dir():
            raise RuntimeIntegrityError("special files are forbidden in run artifacts")
    return sorted(files, key=lambda item: item.relative_to(resolved).as_posix())


def verify_case_template(case_root: Path = CASE_TEMPLATE) -> dict[str, Any]:
    case = case_root.resolve(strict=True)
    if not case.is_dir() or _is_reparse_or_symlink(case):
        raise RuntimeIntegrityError("case template is not a safe directory")
    manifest = load_json(case / "case_manifest.json")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), list):
        raise RuntimeIntegrityError("case manifest is invalid")
    declared: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping):
            raise RuntimeIntegrityError("case manifest entry is invalid")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise RuntimeIntegrityError("case manifest path is invalid")
        pure = Path(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in declared:
            raise RuntimeIntegrityError("case manifest path is unsafe or duplicated")
        declared.add(relative)
        target = (case / pure).resolve(strict=True)
        try:
            target.relative_to(case)
        except ValueError as exc:
            raise RuntimeIntegrityError("case manifest path escapes the template") from exc
        if not target.is_file() or _is_reparse_or_symlink(target):
            raise RuntimeIntegrityError("case manifest member is not a regular file")
        if entry.get("size_bytes") != target.stat().st_size:
            raise RuntimeIntegrityError("case manifest size mismatch")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or sha256_file(target) != expected.upper():
            raise RuntimeIntegrityError("case manifest SHA-256 mismatch")
    actual = {
        path.relative_to(case).as_posix()
        for path in regular_files(case)
        if path.name != "case_manifest.json"
    }
    if actual != declared:
        raise RuntimeIntegrityError("case template contains undeclared or missing files")
    return dict(manifest)


def clone_frozen_case(destination: Path) -> Path:
    """Clone and re-verify the frozen case without permitting overwrite."""

    verify_case_template(CASE_TEMPLATE)
    if destination.exists() or destination.is_symlink():
        raise RuntimeIntegrityError("workspace destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CASE_TEMPLATE, destination, copy_function=shutil.copy2)
    verify_case_template(destination)
    generated = [
        child.name
        for child in (destination / "attempts").iterdir()
        if child.is_dir() and child.name.startswith("attempt_")
        and child.name != "attempt_000_initial_failure"
    ]
    if generated:
        raise RuntimeIntegrityError("cloned workspace contains generated attempts")
    return destination


def build_file_manifest(root: Path, *, excluded: Iterable[str] = ()) -> dict[str, Any]:
    excluded_set = {str(value).replace("\\", "/") for value in excluded}
    entries: list[dict[str, Any]] = []
    for path in regular_files(root):
        relative = path.relative_to(root.resolve(strict=True)).as_posix()
        if relative in excluded_set:
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": "1.0", "algorithm": "SHA-256", "files": entries}


def verify_file_manifest(root: Path, manifest_path: Path) -> bool:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), list):
        return False
    expected = build_file_manifest(
        root,
        excluded=(manifest_path.relative_to(root).as_posix(),),
    )
    return manifest == expected


def scan_tree(root: Path) -> dict[str, Any]:
    """Run non-reflective secret and runtime-path scans over an isolated tree."""

    from package_tools.scanners import scan_runtime_absolute_paths, scan_secrets

    files = regular_files(root)
    secrets = scan_secrets(root, files)
    absolute, expected_test_literals = scan_runtime_absolute_paths(root, files)
    return {
        "passed": not secrets and not absolute,
        "file_count": len(files),
        "secret_finding_count": len(secrets),
        "runtime_absolute_path_finding_count": len(absolute),
        "expected_security_test_path_literal_count": len(expected_test_literals),
        "secret_findings": secrets,
        "runtime_absolute_path_findings": absolute,
    }


def deterministic_evaluation(
    *,
    final_answer: Path,
    trajectory: Path,
    artifacts: Path,
    output: Path | None = None,
    config_path: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Evaluate twice in memory and optionally persist the byte-identical result."""

    from evaluator.evaluate import evaluate_files, evaluation_bytes

    arguments = {
        "final_answer_path": final_answer,
        "trajectory_path": trajectory,
        "artifacts_path": artifacts,
        "config_path": config_path or ROOT / "configs" / "task.yaml",
    }
    first = evaluate_files(**arguments)
    second = evaluate_files(**arguments)
    first_bytes = evaluation_bytes(first, indent=2)
    if first_bytes != evaluation_bytes(second, indent=2):
        raise RuntimeIntegrityError("evaluator output is not byte deterministic")
    if output is not None:
        if output.exists() or output.is_symlink():
            raise RuntimeIntegrityError("evaluation output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(first_bytes)
    return first, first_bytes


def promote_formal_trajectory(
    *,
    logical_model: str,
    agent_output: Path,
    workspace: Path,
    evaluation_path: Path,
    candidate_root: Path,
) -> Path:
    """Build, re-evaluate, scan, and atomically promote an untouched formal run."""

    if logical_model not in {"kimi", "deepseek", "glm"}:
        raise RuntimeIntegrityError("logical model is not formal")
    target = ROOT / "trajectories" / logical_model
    if target.exists() or target.is_symlink():
        raise RuntimeIntegrityError("formal trajectory already exists; overwrite is forbidden")
    if candidate_root.exists() or candidate_root.is_symlink():
        raise RuntimeIntegrityError("promotion candidate already exists")
    candidate_root.mkdir(parents=True)
    for name in TRAJECTORY_REQUIRED[:-1]:
        source = agent_output / name
        if not source.is_file():
            raise RuntimeIntegrityError("runner output is incomplete")
        shutil.copy2(source, candidate_root / name)
    shutil.copy2(evaluation_path, candidate_root / "evaluation.json")
    shutil.copytree(workspace, candidate_root / "artifacts", copy_function=shutil.copy2)

    frozen_evaluation = (candidate_root / "evaluation.json").read_bytes()
    replay, replay_bytes = deterministic_evaluation(
        final_answer=candidate_root / "final_answer.json",
        trajectory=candidate_root / "trajectory.jsonl",
        artifacts=candidate_root / "artifacts",
    )
    if replay_bytes != frozen_evaluation:
        raise RuntimeIntegrityError("promoted evaluator replay differs from frozen evaluation")
    if not bool(replay.get("eligibility", {}).get("eligible")):
        raise RuntimeIntegrityError("candidate is not evaluator-eligible")

    manifest_path = candidate_root / "artifact_manifest.json"
    write_json_exclusive(
        manifest_path,
        build_file_manifest(candidate_root, excluded=("artifact_manifest.json",)),
    )
    if not verify_file_manifest(candidate_root, manifest_path):
        raise RuntimeIntegrityError("trajectory artifact manifest is not reproducible")
    scan = scan_tree(candidate_root)
    if not scan["passed"]:
        raise RuntimeIntegrityError("trajectory secret or absolute-path scan failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate_root.rename(target)
    return target


def safe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    system_root = environment.get("SystemRoot")
    if system_root:
        system32 = str(Path(system_root) / "System32")
        environment["PATH"] = system32 + os.pathsep + environment.get("PATH", "")
    return environment


__all__ = [
    "CASE_TEMPLATE",
    "ROOT",
    "RuntimeIntegrityError",
    "build_file_manifest",
    "clone_frozen_case",
    "deterministic_evaluation",
    "load_json",
    "promote_formal_trajectory",
    "pre_model_source_fingerprint",
    "regular_files",
    "safe_environment",
    "scan_tree",
    "sha256_file",
    "verify_case_template",
    "verify_file_manifest",
    "write_json_exclusive",
]
