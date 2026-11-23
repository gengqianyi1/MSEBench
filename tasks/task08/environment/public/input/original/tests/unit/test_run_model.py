from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_model as run_model
from scripts.runtime_support import RuntimeIntegrityError


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_model, "ROOT", tmp_path)
    monkeypatch.setattr(run_model, "_pre_model_gate", lambda: None)

    def clone(destination: Path) -> Path:
        destination.mkdir(parents=True)
        return destination

    monkeypatch.setattr(run_model, "clone_frozen_case", clone)


def _assert_preserved(tmp_path: Path, run_id: str, stage: str) -> None:
    source = tmp_path / "runs" / "kimi" / run_id
    failed = tmp_path / "runs_failed" / "kimi" / run_id
    assert not source.exists()
    assert failed.is_dir()
    summary = json.loads((failed / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED_NOT_PROMOTED"
    assert summary["failure_stage"] == stage
    assert summary["local_disposition"] == f"runs_failed/kimi/{run_id}"


def test_configuration_exception_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)

    class BrokenTask:
        @staticmethod
        def load(_path: Path) -> object:
            raise ValueError("private detail must not be persisted")

    monkeypatch.setattr(run_model, "TaskConfig", BrokenTask)
    with pytest.raises(RuntimeIntegrityError, match="AFTER_PRESERVATION"):
        run_model.run_formal_model("kimi", run_id="unit-config-failure")
    _assert_preserved(tmp_path, "unit-config-failure", "LOAD_FORMAL_CONFIGURATION")


def _stub_successful_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    model = SimpleNamespace(model="Pro/moonshotai/Kimi-K2.6")
    monkeypatch.setattr(
        run_model,
        "TaskConfig",
        SimpleNamespace(load=lambda _path: SimpleNamespace()),
    )
    monkeypatch.setattr(
        run_model,
        "ModelConfig",
        SimpleNamespace(load_registry=lambda _path, _logical: model),
    )
    monkeypatch.setattr(
        run_model,
        "ToolContractSet",
        SimpleNamespace(load=lambda _path: SimpleNamespace()),
    )
    monkeypatch.setattr(run_model, "ToolRegistry", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_model, "_formal_prompt", lambda: "complete public task")

    class Runner:
        def __init__(self, **kwargs: object) -> None:
            self.output = Path(kwargs["output_dir"])

        def run(self, **_kwargs: object) -> object:
            self.output.mkdir(parents=True)
            return SimpleNamespace(complete=True, status="COMPLETE")

    monkeypatch.setattr(run_model, "AgentRunner", Runner)


def test_evaluator_exception_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    _stub_successful_runner(monkeypatch)
    monkeypatch.setattr(
        run_model,
        "deterministic_evaluation",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unsafe evaluator detail")),
    )
    with pytest.raises(RuntimeIntegrityError, match="AFTER_PRESERVATION"):
        run_model.run_formal_model("kimi", run_id="unit-evaluator-failure")
    _assert_preserved(tmp_path, "unit-evaluator-failure", "EVALUATE_AND_REPLAY")


def test_promotion_exception_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    _stub_successful_runner(monkeypatch)

    def evaluate(*, output: Path, **_kwargs: object) -> tuple[dict[str, object], bytes]:
        payload = b"{}\n"
        output.write_bytes(payload)
        return {"eligibility": {"eligible": True}, "official_score": 100}, payload

    monkeypatch.setattr(run_model, "deterministic_evaluation", evaluate)
    monkeypatch.setattr(
        run_model,
        "load_json",
        lambda _path: {
            "requested_model": "Pro/moonshotai/Kimi-K2.6",
            "actual_model": "Pro/moonshotai/Kimi-K2.6",
            "execution_mode": "REAL_API",
            "native_mode": "LIVE_XTB",
        },
    )
    monkeypatch.setattr(
        run_model,
        "scan_tree",
        lambda _path: {
            "passed": True,
            "secret_finding_count": 0,
            "runtime_absolute_path_finding_count": 0,
        },
    )
    monkeypatch.setattr(
        run_model,
        "promote_formal_trajectory",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("promotion failed")),
    )
    with pytest.raises(RuntimeIntegrityError, match="AFTER_PRESERVATION"):
        run_model.run_formal_model("kimi", run_id="unit-promotion-failure")
    _assert_preserved(tmp_path, "unit-promotion-failure", "PROMOTE_FORMAL_TRAJECTORY")
