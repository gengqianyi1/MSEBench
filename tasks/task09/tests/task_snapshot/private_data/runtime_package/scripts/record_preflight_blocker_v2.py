"""Freeze truthful all-or-none campaign blocker receipts after preflight."""

from __future__ import annotations

import json
from pathlib import Path

from evaluator.evaluate_v2 import evaluate_run
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def main() -> int:
    preflight_path = ROOT / "reports" / "PROVIDER_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "BLOCKED":
        raise RuntimeError("this recorder is only valid for a blocked preflight")
    formal_dirs = {
        label: ROOT / "trajectories" / label
        for label in ("kimi", "deepseek", "glm")
    }
    if any(
        any(path.is_file() for path in directory.rglob("*"))
        for directory in formal_dirs.values()
    ):
        raise RuntimeError("formal trajectory data exists; blocker recorder refused")
    baseline = evaluate_run(
        ROOT / "trajectories" / "baseline", write=False
    )
    frozen_baseline = json.loads(
        (ROOT / "trajectories" / "baseline" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    if baseline != frozen_baseline:
        raise RuntimeError("baseline replay changed before blocker freeze")
    baseline_replay = {
        "schema_version": "2.0",
        "status": "PASS",
        "run": "baseline",
        "eligible": baseline["eligible"],
        "total_score": baseline["total_score"],
        "metrics": baseline["metrics"],
        "trajectory_sha256": sha256_file(
            ROOT / "trajectories" / "baseline" / "trajectory.jsonl"
        ),
        "evaluation_sha256": sha256_file(
            ROOT / "trajectories" / "baseline" / "evaluation.json"
        ),
        "byte_deterministic_replay": True,
    }
    _write(
        ROOT / "reports" / "baseline_replay_validation.json",
        baseline_replay,
    )
    model_receipts = {}
    for label, model in (
        ("kimi", "Pro/moonshotai/Kimi-K2.6"),
        ("deepseek", "deepseek-ai/DeepSeek-V4-Pro"),
        ("glm", "zai-org/GLM-5.2"),
    ):
        probe = preflight["models"][label]
        receipt = {
            "schema_version": "2.0",
            "label": label,
            "requested_model": model,
            "status": "BLOCKED",
            "formal_trajectory_started": False,
            "formal_provider_turns": 0,
            "native_xtb_attempts": 0,
            "exact_identity": None,
            "eligible": False,
            "total_score": None,
            "preflight_status": probe["status"],
            "preflight_exact_identity": probe["exact_identity"],
            "preflight_error_category": probe["error_category"],
            "blocker": "ALL_OR_NONE_PROVIDER_PREFLIGHT_GATE",
            "raw_provider_response_persisted": False,
            "fallback_used": False,
        }
        _write(
            ROOT / "reports" / f"{label}_formal_validation.json", receipt
        )
        model_receipts[label] = receipt
    blocker = {
        "schema_version": "2.0",
        "status": "BLOCKED",
        "gate": 7,
        "category": "NETWORK_CONNECTION_FAILED",
        "affected_model": "zai-org/GLM-5.2",
        "preflight_report_sha256": sha256_file(preflight_path),
        "preflight_provider_turns_per_model": 1,
        "formal_campaign_started": False,
        "formal_trajectories_created": 0,
        "formal_provider_turns": 0,
        "formal_native_xtb_attempts": 0,
        "retry_permitted_by_frozen_preflight_policy": False,
        "minimum_recovery": (
            "Run a new organizer-authorized preflight after external network "
            "availability changes; all three models must pass before any formal run."
        ),
    }
    _write(ROOT / "reports" / "FORMAL_CAMPAIGN_BLOCKER.json", blocker)
    replay = {
        "schema_version": "2.0",
        "status": "BLOCKED",
        "baseline": baseline_replay,
        "formal_models": {
            label: {
                "status": "BLOCKED",
                "trajectory_present": False,
                "reason": "ALL_OR_NONE_PROVIDER_PREFLIGHT_GATE",
            }
            for label in formal_dirs
        },
        "all_existing_trajectories_replay": True,
        "formal_replay_available": False,
    }
    _write(ROOT / "reports" / "replay_validation.json", replay)
    release = {
        "schema_version": "2.0",
        "task_id": "adaptive_xtb_torsion_profile_agent_v2",
        "benchmark_version": "2.0.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "status": "BLOCKED",
        "release_ready": False,
        "benchmark_framework_status": "PASS",
        "formal_campaign_execution_status": "BLOCKED",
        "models": {
            label: {
                "requested_model": receipt["requested_model"],
                "status": "BLOCKED",
                "formal_trajectory_started": False,
            }
            for label, receipt in model_receipts.items()
        },
        "final_zip_created": False,
        "blocker_report": "reports/FORMAL_CAMPAIGN_BLOCKER.json",
    }
    _write(ROOT / "configs" / "release.json", release)
    print(json.dumps(blocker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
