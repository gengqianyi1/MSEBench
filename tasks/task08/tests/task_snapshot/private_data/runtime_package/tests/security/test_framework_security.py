from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import framework.siliconflow_adapter as adapter_module
from framework import (
    MISSING_GJLD_API_KEY,
    AgentRunner,
    MissingCredentialError,
    ModelConfig,
    ScriptedTransport,
    has_absolute_local_path,
    has_hidden_reasoning,
    load_runtime_credential,
    sanitize_for_persistence,
    validate_hash_chain,
)


ROOT = Path(__file__).resolve().parents[2]


def _task() -> dict:
    value = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    value = copy.deepcopy(value)
    value["budget"]["max_provider_retries"] = 0
    value["budget"]["min_successful_tool_calls"] = 0
    return value


def _model() -> ModelConfig:
    return ModelConfig.load_registry(ROOT / "configs" / "models.example.yaml", "kimi")


def test_persistence_sanitizer_removes_secrets_paths_and_hidden_reasoning() -> None:
    secret = "test-super-secret-value-123456789"
    unsafe = {
        "reasoning_content": "private chain of thought",
        "content": (
            f"<think>hidden {secret}</think> inspect "
            f"D:\\private\\case\\input.xyz using {secret}"
        ),
    }
    safe = sanitize_for_persistence(unsafe, secrets=(secret,))
    encoded = json.dumps(safe, ensure_ascii=False)
    assert secret not in encoded
    assert "private chain of thought" not in encoded
    assert "reasoning_content" not in safe
    assert "D:\\private" not in encoded
    assert not has_hidden_reasoning(safe)
    assert not has_absolute_local_path(safe)


def test_credential_loader_uses_only_parent_dotenv_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(*, dotenv_path, override):
        calls.append((Path(dotenv_path), override))
        return False

    monkeypatch.setattr(adapter_module, "load_dotenv", fake_load_dotenv)
    monkeypatch.delenv("GJLD_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError) as caught:
        load_runtime_credential(project)
    assert str(caught.value) == MISSING_GJLD_API_KEY
    assert calls == [(project.parent / ".env", False)]

    monkeypatch.setenv("GJLD_API_KEY", "process-value-wins")
    assert load_runtime_credential(project) == "process-value-wins"
    assert calls[-1] == (project.parent / ".env", False)


def test_provider_exception_persists_only_safe_type_and_status(tmp_path: Path) -> None:
    secret = "provider-secret-never-persist-987654321"
    host_path = r"D:\private\provider\payload.json"
    destination = tmp_path / "safe-failure"
    transport = ScriptedTransport([RuntimeError(f"{secret} at {host_path}")])
    result = AgentRunner(
        task_config=_task(),
        model_config=_model(),
        logical_model="kimi",
        output_dir=destination,
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={"list_case_files": lambda: {"ok": True}},
        transport=transport,
        run_id="security-provider-failure",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
    ).run()
    assert result.status == "FAILED"
    assert {path.name for path in destination.iterdir()} == {
        "trajectory.jsonl",
        "dialogue.md",
        "metadata.json",
        "final_answer.json",
    }
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in destination.iterdir() if path.is_file()
    )
    assert secret not in persisted
    assert host_path not in persisted
    assert "RuntimeError" in persisted
    assert "PROVIDER_REQUEST_FAILED" in persisted
    events = [
        json.loads(line)
        for line in (destination / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_hash_chain(events)
    failures = [event["payload"] for event in events if event["event"] == "provider_error"]
    assert failures[0]["error"] == {
        "type": "RuntimeError",
        "status": "PROVIDER_REQUEST_FAILED",
    }
