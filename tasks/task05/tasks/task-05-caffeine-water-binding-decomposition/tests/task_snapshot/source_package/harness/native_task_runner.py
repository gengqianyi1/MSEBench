"""Run the verified five-tool xTB Agent outside the model's file sandbox."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .budget import BudgetLedger
from .model_client import RobustModelClient
from .runtime_checks import evidence_tree_sha256, run_runtime_checks, sha256_file
from .schema_validation import validate_budget_ledger, validate_document, validate_scientific_manifest
from .release_contract import load_release_contract


HIDDEN_REASONING_FIELDS = {"reasoning", "reasoning_content", "chain_of_thought", "analysis"}


def _artifact_manifest(submission_dir: Path) -> list[dict[str, str]]:
    result = []
    for path in sorted(item for item in Path(submission_dir).rglob("*") if item.is_file()):
        relative = path.relative_to(submission_dir).as_posix()
        if relative == "run_manifest.json":
            continue
        result.append({"path": relative, "sha256": sha256_file(path)})
    return result


def submission_payload_sha256(submission_dir: Path) -> str:
    """Hash every payload file except the self-referential run manifest."""
    digest = hashlib.sha256()
    for item in _artifact_manifest(Path(submission_dir)):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _protocol_from_job(job: Mapping[str, Any]) -> str:
    command = [str(value).lower() for value in (job.get("command") or [])]
    namespace = str(job.get("namespace") or "").lower()
    if "--hess" in command or "hess" in namespace:
        return "hess"
    if "--opt" in command or "opt" in namespace:
        return "opt"
    if "scan" in namespace or any("scan" in value for value in command):
        return "scan"
    return "sp"


def upgrade_scientific_manifest(
    old_manifest: Mapping[str, Any],
    *,
    task_id: str,
    scientific_problem_id: str,
    run_id: str,
    task_version: str,
    grader_version: str,
    package_release: str,
    runtime_checks: Mapping[str, Any],
    budget_ledger: Mapping[str, Any],
    environment: Mapping[str, str],
    submission_dir: Path,
) -> dict[str, Any]:
    """Add the formal v2 fields while retaining all raw xTB evidence fields."""
    upgraded = deepcopy(dict(old_manifest))
    upgraded.update(
        {
            "schema_name": "xtb_submission_manifest_v2",
            "schema_version": "2.1.0",
            "manifest_role": "scientific_submission",
            "run_id": run_id,
            "task_id": task_id,
            "scientific_problem_id": scientific_problem_id,
            "task_version": task_version,
            "grader_version": grader_version,
            "package_release": package_release,
            "runtime_checks": deepcopy(dict(runtime_checks)),
            "budget_ledger": deepcopy(dict(budget_ledger)),
            "environment": dict(environment),
            **dict(environment),
        }
    )
    jobs = []
    for raw_value in old_manifest.get("jobs", []) or []:
        if not isinstance(raw_value, dict):
            continue
        raw = deepcopy(raw_value)
        workdir = str(raw.get("workdir") or raw.get("working_directory") or "")
        inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
        geometry = inputs.get("geometry") if isinstance(inputs.get("geometry"), dict) else {}
        detailed = inputs.get("detailed_input") if isinstance(inputs.get("detailed_input"), dict) else {}
        files_sha256 = raw.get("files_sha256") if isinstance(raw.get("files_sha256"), dict) else {}
        evidence = [f"{workdir.rstrip('/')}/{name}" for name in sorted(files_sha256)] if workdir else []
        raw.update(
            {
                "protocol": _protocol_from_job(raw),
                "working_directory": workdir,
                "input_geometry_sha256": str(
                    raw.get("input_geometry_sha256")
                    or geometry.get("job_copy_sha256")
                    or geometry.get("source_sha256")
                    or "0" * 64
                ),
                "detailed_input_sha256": (
                    raw.get("detailed_input_sha256")
                    or detailed.get("job_copy_sha256")
                    or detailed.get("source_sha256")
                ),
                "output_sha256": raw.get("output_sha256") or raw.get("evidence_tree_sha256"),
                "parent_job_id": raw.get("parent_job_id"),
                "retry_of_job_id": raw.get("retry_of_job_id") or raw.get("retry_of"),
                "workdir": workdir,
                "evidence_tree_sha256": evidence_tree_sha256(files_sha256) if files_sha256 else None,
                "evidence": evidence,
            }
        )
        jobs.append(raw)
    upgraded["jobs"] = jobs
    upgraded["artifacts"] = _artifact_manifest(Path(submission_dir))
    upgraded["submission_sha256"] = submission_payload_sha256(Path(submission_dir))
    return upgraded


class ObservableModelAdapter:
    """Expose the old AgentRun client API while using the frozen robust client."""

    def __init__(
        self,
        client: RobustModelClient,
        *,
        resumed: bool = False,
        hidden_context: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.logical_turn = 0
        self.resumed = resumed
        # Provider-private thinking state is retained only in process memory.
        # Keys are stable message indexes in the Agent's append-only history.
        self.hidden_context = hidden_context if hidden_context is not None else {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.client.config["provider"],
            "model": self.client.config["requested_model_id"],
            "temperature": self.client.config["temperature"],
            "top_p": self.client.config["top_p"],
            "seed": self.client.config["seed"],
            "request_timeout_seconds": self.client.config["request_timeout_seconds"],
        }

    def create_chat_completion(self, *, messages, tools=None, seed=None, max_tokens=None) -> dict[str, Any]:
        self.logical_turn += 1
        self.client.config["tools"] = list(tools or [])
        provider_messages = deepcopy(list(messages))
        for index, hidden in self.hidden_context.items():
            if 0 <= index < len(provider_messages) and provider_messages[index].get("role") == "assistant":
                provider_messages[index].update(hidden)
        response = self.client.complete(provider_messages, logical_turn=self.logical_turn, resumed=self.resumed)
        choice = response["choices"][0]
        message = choice["message"]
        hidden = {key: value for key, value in message.items() if key in HIDDEN_REASONING_FIELDS}
        if hidden:
            self.hidden_context[len(messages)] = hidden
        clean_message = {key: value for key, value in message.items() if key not in HIDDEN_REASONING_FIELDS}
        return {
            "id": response.get("id"),
            "model": response.get("model") or self.client.config["requested_model_id"],
            "created": response.get("created"),
            "message": clean_message,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {},
            "credential_env": None,
        }


def _copy_submission(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _stage_scientific_submission(final_dir: Path, staging: Path) -> None:
    """Freeze exactly the agent's contracted final payload for outer collection."""
    final_dir = Path(final_dir)
    staging = Path(staging)
    if not final_dir.is_dir():
        raise FileNotFoundError(f"agent final submission is missing: {final_dir}")
    if staging.exists():
        raise FileExistsError(f"native submission staging already exists: {staging}")
    staging.mkdir(parents=True)
    _copy_submission(final_dir, staging)


def _load_model_config(bundle_root: Path) -> dict[str, Any]:
    explicit = os.environ.get("CHEMBENCH_MODEL_CONFIG_FILE")
    if explicit:
        path = Path(explicit).resolve()
    else:
        display_name = os.environ.get("CHEMBENCH_MODEL_NAME") or os.environ.get("CHEMBENCH_LLM_MODEL")
        path = bundle_root / "configs" / "models" / f"{display_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"frozen model config is missing: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config["url"] = config.pop("base_url")
    return config


def _archive_native_artifacts(
    run_dir: Path,
    agent: Any,
    clients: list[Any],
    budget_document: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Path:
    """Persist observable evidence before any fallible submission postprocessing."""
    artifacts = Path(run_dir) / "native_run_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory.jsonl", "tool_calls.jsonl", "conversation.md", "run_manifest.json", "run_failure.json"):
        source = Path(agent.run_dir) / name
        if source.is_file():
            shutil.copy2(source, artifacts / name)
    (artifacts / "api_events.jsonl").write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for client in clients
            for event in client.events
        ),
        encoding="utf-8",
    )
    (artifacts / "budget_ledger.json").write_text(
        json.dumps(dict(budget_document), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifacts / "runtime_checks.json").write_text(
        json.dumps(dict(runtime), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifacts


def main_for_task(entrypoint: str) -> int:
    task_dir = Path(entrypoint).resolve().parent
    bundle_root = task_dir.parents[1]
    run_dir = Path(os.environ["CHEMBENCH_RUN_DIR"]).resolve()
    workspace = Path(os.environ["CHEMBENCH_WORKSPACE"]).resolve()
    schema_dir = bundle_root / "schemas"
    task_config = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    release_contract = load_release_contract(bundle_root)
    runtime = run_runtime_checks(task_dir, task_config["runtime_checks"], schema_dir)
    if runtime["status"] != "passed":
        raise RuntimeError("task-declared runtime checks failed")

    model_config = _load_model_config(bundle_root)
    keys = {
        slot: os.environ.get(env_name, "")
        for slot, env_name in model_config.pop("api_key_slots").items()
    }
    ledger = BudgetLedger(
        task_config["budget_limits"],
        compute_budget=task_config.get("compute_budget"),
        schema_dir=schema_dir,
    )
    robust_client = RobustModelClient(model_config, keys=keys, ledger=ledger)
    clients = [robust_client]
    adapter = ObservableModelAdapter(robust_client)

    runtime_root = bundle_root / "runtime_common"
    sys.path.insert(0, str(runtime_root))
    from framework.agent_loop import AgentRun

    public_science = task_dir / "input" / "original"
    internal_root = run_dir / "native_internal"
    internal_root.mkdir(parents=True, exist_ok=True)
    agent = AgentRun(
        task_dir=public_science,
        output_root=internal_root,
        model_client=adapter,
        xtb_executable="xtb",
        seed=int(model_config["seed"]),
        budget_ledger=ledger,
    )
    failure = None
    try:
        final_dir = agent.run()
    except Exception as exc:
        first_failure = f"{type(exc).__name__}: {exc}"
        resumable = "transport retries exhausted" in first_failure
        if resumable and int(model_config.get("max_process_resumes", 1)) >= 1:
            ledger.record_process_resume(max(1, agent._turns), first_failure)
            resumed_client = RobustModelClient(model_config, keys=keys, ledger=ledger)
            clients.append(resumed_client)
            resumed_adapter = ObservableModelAdapter(
                resumed_client,
                resumed=True,
                hidden_context=adapter.hidden_context,
            )
            try:
                agent = AgentRun.resume_existing(
                    task_dir=public_science,
                    run_dir=agent.run_dir,
                    model_client=resumed_adapter,
                    xtb_executable="xtb",
                    seed=int(model_config["seed"]),
                    budget_ledger=ledger,
                )
                resumed_adapter.logical_turn = agent._turns
                final_dir = agent.run()
            except Exception as resume_exc:
                failure = f"{type(resume_exc).__name__}: {resume_exc}"
                final_dir = agent.run_dir / "final_submission"
        else:
            failure = first_failure
            final_dir = agent.run_dir / "final_submission"
    budget_document = ledger.to_document()
    artifacts = _archive_native_artifacts(run_dir, agent, clients, budget_document, runtime)
    submission_staging = run_dir / "native_submission"
    try:
        validate_budget_ledger(budget_document, schema_dir)
        _stage_scientific_submission(final_dir, submission_staging)
        manifest_path = submission_staging / "run_manifest.json"
        if manifest_path.is_file():
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            environment = {
                "environment_lock_sha256": runtime["environment_lock"]["actual_sha256"],
                "task_config_sha256": runtime["hashes"]["task_config_sha256"],
                "public_inputs_sha256": runtime["hashes"]["public_inputs_sha256"],
                "xtb_executable_sha256": runtime["binaries"][0]["sha256"],
            }
            upgraded = upgrade_scientific_manifest(
                old_manifest,
                task_id=task_config["task_id"],
                scientific_problem_id=task_config["scientific_problem_id"],
                run_id=os.environ.get("CHEMBENCH_RUN_ID", run_dir.name),
                task_version=task_config["version"],
                grader_version=release_contract.task_grader_version(task_config["task_id"]),
                package_release=release_contract.task_package_release(task_config["task_id"]),
                runtime_checks=runtime,
                budget_ledger=budget_document,
                environment=environment,
                submission_dir=submission_staging,
            )
            validate_scientific_manifest(upgraded, schema_dir)
            manifest_path.write_text(
                json.dumps(upgraded, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        postprocess_failure = f"{type(exc).__name__}: {exc}"
        failure = f"{failure}; postprocessing failed: {postprocess_failure}" if failure else postprocess_failure
    result = {
        "task_id": task_config["task_id"],
        "requested_model_id": model_config["requested_model_id"],
        "returned_model_ids": sorted({event["returned_model_id"] for client in clients for event in client.events if event["returned_model_id"]}),
        "failure": failure,
    }
    (artifacts / "native_runner_result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if failure is None else 1
