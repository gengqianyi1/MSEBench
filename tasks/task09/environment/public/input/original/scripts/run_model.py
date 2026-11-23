"""Run exactly one immutable protocol-2.0 SiliconFlow formal trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from evaluator.evaluate_v2 import EvaluationError, evaluate_run
from framework.agent_runner import run_formal_model
from framework.execution_freeze import (
    ExecutionFreezeError,
    load_execution_freeze_binding,
)
from framework.schemas import (
    ACTIVE_PREFLIGHT_REPORT,
    BENCHMARK_VERSION,
    FORMAL_CAMPAIGN_ID,
    FORMAL_MODELS,
    FORMAL_VALIDATION_REPORTS,
    PROTOCOL,
    TASK_ID,
)
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ORDER = tuple(FORMAL_MODELS)
SAFE_STATUS = re.compile(r"^[A-Z0-9_.:-]{1,160}$")


def _strict_json(path: Path) -> dict[str, Any]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
    )
    if duplicates or not isinstance(value, dict):
        raise ValueError(f"invalid closed JSON object: {path.name}")
    return value


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_all_pass_preflight() -> dict[str, Any]:
    path = ROOT / ACTIVE_PREFLIGHT_REPORT
    receipt = _strict_json(path)
    models = receipt.get("models")
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("benchmark_version") != BENCHMARK_VERSION
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != "PASS"
        or receipt.get("formal_campaign_allowed") is not True
        or receipt.get("native_xtb_calls") != 0
        or receipt.get("preflight_provider_turns_per_model") != 1
        or receipt.get("raw_provider_responses_persisted") is not False
        or not isinstance(models, dict)
        or set(models) != set(FORMAL_MODELS)
    ):
        raise RuntimeError("ALL_OR_NONE_PROVIDER_PREFLIGHT_NOT_PASS")
    for label, expected in FORMAL_MODELS.items():
        item = models[label]
        if (
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or item.get("requested_model") != expected
            or item.get("actual_model") != expected
            or item.get("exact_identity") is not True
            or item.get("structured_json_valid") is not True
            or item.get("fallback_used") is not False
            or item.get("model_override_used") is not False
            or item.get("formal_trajectory") is not False
            or item.get("native_xtb_calls") != 0
            or item.get("raw_response_persisted") is not False
        ):
            raise RuntimeError("ALL_OR_NONE_PROVIDER_PREFLIGHT_NOT_PASS")
    return receipt


def _validate_root_execution_commitment() -> tuple[dict[str, Any], dict[str, Any]]:
    independent_binding = load_execution_freeze_binding(ROOT)
    payload = _strict_json(ROOT / "execution_commitment.json")
    return payload, {
        "passed": True,
        **independent_binding,
    }


def _formal_run_valid(label: str) -> bool:
    run_root = ROOT / "trajectories" / label
    required = {
        "trajectory.jsonl",
        "final_answer.json",
        "evaluation.json",
        "run_manifest.json",
        "usage_report.json",
    }
    if any(not (run_root / name).is_file() for name in required):
        return False
    try:
        evaluation = evaluate_run(run_root, project_root=ROOT, write=False)
    except (EvaluationError, OSError, UnicodeError, ValueError, TypeError):
        return False
    replay_bound = (
        _compact_json(evaluation)
        == (run_root / "evaluation.json").read_bytes()
    )
    if (
        evaluation.get("run_status") in {"FAIL", "BLOCKED"}
        and evaluation.get("legal_failure_envelope") is True
    ):
        # A model/protocol failure or provider block with a closed,
        # execution-freeze-bound terminal envelope is a completed formal
        # outcome for campaign ordering; it must not suppress later models.
        return bool(
            replay_bound
            and evaluation.get("execution_freeze_binding", {}).get("passed")
            is True
        )
    return bool(
        replay_bound
        and evaluation.get("run_status") in {"PASS", "FAIL"}
        and evaluation.get("reconstructed", {}).get("actual_model_identity")
        == FORMAL_MODELS[label]
        and evaluation.get("eligibility", {})
        .get("gates", {})
        .get("exact_model_identity")
        is True
    )


def _validate_formal_order(label: str) -> None:
    index = FORMAL_ORDER.index(label)
    for earlier in FORMAL_ORDER[:index]:
        if not _formal_run_valid(earlier):
            raise RuntimeError(f"FORMAL_ORDER_PREDECESSOR_NOT_COMPLETE:{earlier}")
    for current_or_later in FORMAL_ORDER[index:]:
        directory = ROOT / "trajectories" / current_or_later
        if directory.exists() and any(
            item.is_file() or item.is_symlink()
            for item in directory.rglob("*")
        ):
            raise RuntimeError(
                f"FORMAL_TRAJECTORY_ALREADY_STARTED:{current_or_later}"
            )


def _atomic_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("formal validation report already exists")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("formal validation partial report exists")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
    temporary.replace(path)


def _pre_model_gates(
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = _validate_all_pass_preflight()
    execution, execution_validation = _validate_root_execution_commitment()
    _validate_formal_order(label)
    return preflight, execution, execution_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=list(FORMAL_ORDER))
    args = parser.parse_args()
    try:
        _, _execution_commitment, execution_validation = _pre_model_gates(
            args.label
        )
    except (
        EvaluationError,
        ExecutionFreezeError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
        TypeError,
    ) as exc:
        raw_reason = str(exc)
        reason = (
            raw_reason
            if SAFE_STATUS.fullmatch(raw_reason)
            else type(exc).__name__.upper()
        )
        print(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "label": args.label,
                    "status": "BLOCKED",
                    "formal_trajectory_started": False,
                    "native_xtb_attempts": 0,
                    "reason": reason,
                },
                sort_keys=True,
            )
        )
        return 2
    destination = ROOT / "trajectories" / args.label

    # The runner independently reloads the root execution freeze and binds its
    # compact identity into the trajectory and trusted final envelope.
    run_formal_model(args.label, destination)

    first = evaluate_run(destination, project_root=ROOT, write=False)
    second = evaluate_run(destination, project_root=ROOT, write=False)
    if first != second:
        raise RuntimeError("V2_EVALUATOR_NONDETERMINISTIC")
    written = evaluate_run(destination, project_root=ROOT, write=True)
    if first != written:
        raise RuntimeError("V2_WRITTEN_EVALUATION_CHANGED")

    actual = first.get(
        "actual_model_identity",
        first.get("reconstructed", {}).get("actual_model_identity"),
    )
    exact_identity = bool(
        actual == FORMAL_MODELS[args.label]
        and (
            first.get("exact_model_identity") is True
            or first.get("eligibility", {})
            .get("gates", {})
            .get("exact_model_identity")
            is True
        )
    )
    usage = _strict_json(destination / "usage_report.json")
    previous_blocked_report = (
        ROOT / "reports" / f"{args.label}_formal_validation.json"
    )
    previous_report_sha256 = (
        sha256_file(previous_blocked_report)
        if previous_blocked_report.is_file()
        else None
    )
    validation_report = ROOT / FORMAL_VALIDATION_REPORTS[args.label]
    summary = {
        "schema_version": "2.0",
        "benchmark_version": BENCHMARK_VERSION,
        "protocol": PROTOCOL,
        "formal_campaign_id": FORMAL_CAMPAIGN_ID,
        "label": args.label,
        "requested_model_id": FORMAL_MODELS[args.label],
        "actual_model_id": actual,
        "status": first["run_status"],
        "eligible": first["eligible"],
        "total_score": first["total_score"],
        "metrics": first.get("metrics", {}),
        "exact_identity": exact_identity,
        "fallback_used": False,
        "model_override_used": False,
        "state": first.get("reconstructed", {}).get("state", {}),
        "usage": {
            "provider_turns": usage.get("provider_turns"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "native_xtb_attempts": usage.get("native_xtb_attempts"),
        },
        "preflight_report_sha256": sha256_file(
            ROOT / ACTIVE_PREFLIGHT_REPORT
        ),
        "root_execution_commitment_sha256": sha256_file(
            ROOT / "execution_commitment.json"
        ),
        "root_execution_content_sha256": execution_validation[
            "content_sha256"
        ],
        "trajectory_sha256": sha256_file(
            destination / "trajectory.jsonl"
        ),
        "final_answer_sha256": sha256_file(
            destination / "final_answer.json"
        ),
        "evaluation_sha256": sha256_file(
            destination / "evaluation.json"
        ),
        "usage_report_sha256": sha256_file(
            destination / "usage_report.json"
        ),
        "previous_blocked_report_sha256": previous_report_sha256,
        "raw_provider_response_persisted": False,
    }
    _atomic_report(validation_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Scientific ineligibility is a valid formal result. Provider/identity or
    # trajectory blocking is not.
    return (
        0
        if summary["status"] in {"PASS", "FAIL"}
        and summary["exact_identity"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
