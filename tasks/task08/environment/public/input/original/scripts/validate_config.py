from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "kimi": "Pro/moonshotai/Kimi-K2.6",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "glm": "zai-org/GLM-5.2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("non-finite JSON value")
        ),
    )


def _yaml(path: Path) -> Any:
    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[str, Any]:
        loader.flatten_mapping(node)
        result: dict[str, Any] = {}
        for key, value in loader.construct_pairs(node, deep=deep):
            if key in result:
                raise ValueError("duplicate YAML key")
            result[key] = value
        return result

    UniqueSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueSafeLoader)


def validate_configuration(root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    task_path = root / "configs" / "task.yaml"
    try:
        task = _yaml(task_path)
        if not isinstance(task, dict):
            raise TypeError("task config must be a mapping")
    except Exception:
        return {
            "schema_version": "1.0",
            "passed": False,
            "status": "BLOCKED_CONFIG_CONFLICT",
            "errors": ["TASK_CONFIG_INVALID"],
        }

    def exact_keys(value: Any, expected: set[str], label: str) -> None:
        if not isinstance(value, dict) or set(value) != expected:
            errors.append(f"{label}_SCHEMA_MISMATCH")

    exact_keys(
        task,
        {
            "schema_version",
            "task",
            "budget",
            "runtime",
            "models",
            "tools",
            "evaluator",
            "scoring",
            "security",
            "determinism",
            "packaging",
        },
        "TASK_ROOT",
    )
    required_mapping_sections = (
        "task",
        "budget",
        "runtime",
        "models",
        "tools",
        "evaluator",
        "scoring",
        "security",
        "determinism",
        "packaging",
    )
    malformed_sections = [
        name for name in required_mapping_sections if not isinstance(task.get(name), dict)
    ]
    if malformed_sections:
        return {
            "schema_version": "1.0",
            "passed": False,
            "status": "BLOCKED_CONFIG_CONFLICT",
            "errors": [f"{name.upper()}_SCHEMA_MISMATCH" for name in malformed_sections],
        }
    exact_keys(
        task.get("task"),
        {"id", "case_id", "protocol", "job_spec", "case_manifest"},
        "TASK_IDENTITY",
    )
    exact_keys(task.get("task", {}).get("protocol"), {"name", "version"}, "PROTOCOL")
    exact_keys(task.get("task", {}).get("job_spec"), {"path", "sha256"}, "JOB_REFERENCE")
    exact_keys(
        task.get("task", {}).get("case_manifest"),
        {"path", "sha256"},
        "CASE_MANIFEST_REFERENCE",
    )
    expected_budget = {
        "max_tool_calls",
        "min_successful_tool_calls",
        "max_agent_turns",
        "max_execution_attempts",
        "max_provider_retries",
        "total_timeout_seconds",
        "single_xtb_timeout_seconds",
        "max_text_read_lines_per_call",
        "max_text_read_bytes_per_call",
        "require_read_text_evidence",
        "require_native_xtb_execution",
        "require_geometry_optimization",
        "require_hessian",
    }
    exact_keys(task.get("budget"), expected_budget, "BUDGET")
    budget_value = task.get("budget") if isinstance(task.get("budget"), dict) else {}
    integer_budget = expected_budget - {
        "require_read_text_evidence",
        "require_native_xtb_execution",
        "require_geometry_optimization",
        "require_hessian",
    }
    if any(
        isinstance(budget_value.get(name), bool)
        or not isinstance(budget_value.get(name), int)
        or int(budget_value.get(name, 0)) < (0 if name in {"min_successful_tool_calls", "max_provider_retries"} else 1)
        for name in integer_budget
    ):
        errors.append("BUDGET_TYPE_OR_RANGE_INVALID")
    if any(budget_value.get(name) is not True for name in expected_budget - integer_budget):
        errors.append("BUDGET_REQUIRED_GATE_DISABLED")
    exact_keys(
        task.get("models"),
        {
            "registry_path",
            "formal_order",
            "formal_execution_mode",
            "formal_native_mode",
            "require_exact_response_model",
            "allow_fallback",
            "allow_environment_model_override",
        },
        "MODEL_POLICY",
    )
    runtime = task.get("runtime") if isinstance(task.get("runtime"), dict) else {}
    exact_keys(runtime, {"geometry_generation", "replay", "live_xtb", "xtb"}, "RUNTIME")
    exact_keys(
        runtime.get("geometry_generation"),
        {"rdkit_version", "embed_algorithm", "embed_seed", "preoptimization"},
        "GEOMETRY_GENERATION",
    )
    exact_keys(runtime.get("replay"), {"call_model_api", "run_native_xtb"}, "REPLAY_MODE")
    exact_keys(
        runtime.get("live_xtb"),
        {"call_model_api_for_formal_runs", "run_native_xtb"},
        "LIVE_MODE",
    )
    exact_keys(
        runtime.get("xtb"),
        {
            "environment_manager",
            "conda_environment",
            "executable_alias",
            "version",
            "identity_policy",
            "required_help_tokens",
            "platform_reference_hashes",
            "initial_failure_cycle_limit",
            "restart_cycle_limit_default",
            "restart_cycle_limit_min",
            "restart_cycle_limit_max",
            "thread_count_default",
            "thread_count_min",
            "thread_count_max",
            "walltime_seconds_default",
            "walltime_seconds_max",
            "environment",
        },
        "XTB_RUNTIME",
    )
    exact_keys(
        runtime.get("xtb", {}).get("environment"),
        {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"},
        "XTB_ENVIRONMENT",
    )
    exact_keys(task.get("tools"), {"contract_path", "allowed"}, "TOOLS")
    exact_keys(
        task.get("evaluator"),
        {
            "public_independent_replay",
            "private_oracle_required_for_public_score",
            "oracle_commitment",
        },
        "EVALUATOR",
    )
    exact_keys(
        task.get("evaluator", {}).get("oracle_commitment"),
        {"path", "sha256"},
        "ORACLE_COMMITMENT",
    )
    exact_keys(
        task.get("scoring"),
        {"total_points", "rounding_decimal_places", "categories", "diagnosis_reference", "eligibility_gates"},
        "SCORING",
    )
    exact_keys(
        task.get("determinism"),
        {"json_sort_keys", "json_indent", "json_ensure_ascii", "newline", "allow_nan", "trajectory_genesis_hash"},
        "DETERMINISM",
    )
    exact_keys(
        task.get("packaging"),
        {"archive_layout", "archive_filename", "exclude_private_oracle"},
        "PACKAGING",
    )
    exact_keys(
        task.get("security"),
        {
            "workspace_only",
            "reject_absolute_paths",
            "reject_parent_segments",
            "reject_symlink_escape",
            "reject_shell_input",
            "native_shell",
            "model_network_access",
            "redact_provider_exceptions",
            "forbidden_read_roots",
        },
        "SECURITY",
    )

    if task.get("task", {}).get("id") != "xtb_failure_diagnosis_restart_agent_v2":
        errors.append("TASK_ID_MISMATCH")
    protocol = task.get("task", {}).get("protocol", {})
    if protocol != {"name": "xtb_failure_recovery_tool_loop", "version": "1.0"}:
        errors.append("PROTOCOL_MISMATCH")

    categories = task.get("scoring", {}).get("categories", {})
    if not isinstance(categories, dict) or sum(categories.values()) != task.get("scoring", {}).get("total_points"):
        errors.append("SCORING_TOTAL_MISMATCH")
    if task.get("scoring", {}).get("total_points") != 100:
        errors.append("SCORING_TOTAL_NOT_100")
    budget = task.get("budget", {})
    if int(budget.get("min_successful_tool_calls", -1)) > int(budget.get("max_tool_calls", -1)):
        errors.append("TOOL_BUDGET_INCONSISTENT")
    if int(budget.get("max_execution_attempts", 0)) < 1:
        errors.append("EXECUTION_ATTEMPT_BUDGET_INVALID")

    def checked_reference(spec: dict[str, Any], label: str) -> Path | None:
        try:
            relative = Path(str(spec["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            path = (root / relative).resolve()
            path.relative_to(root.resolve())
            if not path.is_file():
                raise FileNotFoundError
            if _sha256(path) != str(spec["sha256"]).upper():
                errors.append(f"{label}_HASH_MISMATCH")
            return path
        except Exception:
            errors.append(f"{label}_REFERENCE_INVALID")
            return None

    job_path = checked_reference(task["task"]["job_spec"], "JOB_SPEC")
    case_manifest_path = checked_reference(task["task"]["case_manifest"], "CASE_MANIFEST")
    commitment_path = checked_reference(task["evaluator"]["oracle_commitment"], "ORACLE_COMMITMENT")

    job = _json(job_path) if job_path else {}
    if job.get("case_id") != task.get("task", {}).get("case_id"):
        errors.append("CASE_ID_MISMATCH")
    if job.get("method") != "GFN2-xTB":
        errors.append("JOB_METHOD_CONFLICT")
    molecule = job.get("molecule", {})
    if molecule.get("charge") != 0 or molecule.get("uhf") != 0:
        errors.append("JOB_ELECTRONIC_STATE_CONFLICT")
    if job.get("solvation", "missing") is not None:
        errors.append("JOB_SOLVATION_CONFLICT")
    if job.get("required_phases") != ["geometry_optimization", "hessian"]:
        errors.append("JOB_PHASE_CONFLICT")
    if int(molecule.get("atom_count", 0)) != len(molecule.get("element_sequence", [])):
        errors.append("JOB_ATOM_SEQUENCE_CONFLICT")

    if case_manifest_path:
        manifest = _json(case_manifest_path)
        case_root = case_manifest_path.parent
        listed = set()
        for item in manifest.get("files", []):
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                errors.append("CASE_MANIFEST_UNSAFE_PATH")
                continue
            path = case_root / relative
            listed.add(relative.as_posix())
            if not path.is_file():
                errors.append("CASE_MANIFEST_FILE_MISSING")
                continue
            if path.stat().st_size != int(item.get("size_bytes", -1)):
                errors.append("CASE_MANIFEST_SIZE_MISMATCH")
            if _sha256(path) != str(item.get("sha256", "")).upper():
                errors.append("CASE_MANIFEST_FILE_HASH_MISMATCH")
        actual = {
            path.relative_to(case_root).as_posix()
            for path in case_root.rglob("*")
            if path.is_file() and path.name != "case_manifest.json"
        }
        if listed != actual:
            errors.append("CASE_MANIFEST_INVENTORY_MISMATCH")
        configured_cycle = task["runtime"]["xtb"]["initial_failure_cycle_limit"]
        if manifest.get("initial_failure", {}).get("optimization_cycle_limit") != configured_cycle:
            errors.append("INITIAL_CYCLE_LIMIT_CONFLICT")
        if manifest.get("source_type") != "NATIVE_XTB":
            errors.append("INITIAL_FIXTURE_NOT_NATIVE")

    model_path = root / str(task["models"]["registry_path"])
    try:
        models = _yaml(model_path)
        actual_models = {
            name: data["model"]
            for name, data in models["models"].items()
        }
        if actual_models != EXPECTED_MODELS:
            errors.append("MODEL_REGISTRY_MISMATCH")
        defaults = models["provider_defaults"]
        if defaults.get("sdk_max_retries") != 0:
            errors.append("SDK_RETRY_NOT_DISABLED")
        if defaults.get("api_key_env") != "GJLD_API_KEY":
            errors.append("API_KEY_ENV_MISMATCH")
    except Exception:
        errors.append("MODEL_REGISTRY_INVALID")

    contract_path = root / str(task["tools"]["contract_path"])
    try:
        contract = _json(contract_path)
        if contract.get("protocol") != protocol:
            errors.append("TOOL_CONTRACT_PROTOCOL_MISMATCH")
        names = [item["function"]["name"] for item in contract["tools"]]
        if names != task["tools"]["allowed"]:
            errors.append("TOOL_CONTRACT_NAME_MISMATCH")
        if len(names) != len(set(names)):
            errors.append("TOOL_CONTRACT_DUPLICATE_NAME")
    except Exception:
        errors.append("TOOL_CONTRACT_INVALID")

    if commitment_path:
        commitment = _json(commitment_path)
        if commitment.get("task_id") != task["task"]["id"]:
            errors.append("ORACLE_COMMITMENT_TASK_MISMATCH")
        if not isinstance(commitment.get("case_count"), int) or commitment["case_count"] < 1:
            errors.append("ORACLE_COMMITMENT_CASE_MISMATCH")
        if commitment.get("public_evaluator_dependency") is not False:
            errors.append("PUBLIC_EVALUATOR_ORACLE_DEPENDENCY")

    unique_errors = sorted(set(errors))
    return {
        "schema_version": "1.0",
        "passed": not unique_errors,
        "status": "PASS" if not unique_errors else "BLOCKED_CONFIG_CONFLICT",
        "errors": unique_errors,
        "checks": {
            "score_total": 100,
            "job_spec_scientific_authority": True,
            "task_config_policy_authority": True,
            "case_manifest_inventory_checked": case_manifest_path is not None,
            "model_registry_checked": True,
            "tool_contract_checked": True,
            "private_oracle_required_for_public_score": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen benchmark configuration")
    parser.parse_args(argv)
    report = validate_configuration()
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
