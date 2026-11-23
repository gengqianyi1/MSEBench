#!/usr/bin/env python3
"""Normalize Harbor trial artifacts into an auditable scoring diagnostic table."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def numeric_reward(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def reward_from_logs(trial_dir: Path) -> float | None:
    candidates = list(trial_dir.glob("logs/verifier/reward.json")) + list(trial_dir.glob("**/verifier/reward.json"))
    for path in candidates:
        payload = load_json(path)
        if payload:
            for key in ("reward", "final_score", "raw_score"):
                result = numeric_reward(payload.get(key))
                if result is not None:
                    return result
    for path in trial_dir.glob("**/verifier/reward.txt"):
        try:
            return numeric_reward(float(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
    return None


def classify(payload: dict[str, Any], trial_dir: Path) -> tuple[str, float | None, str | None]:
    config = payload.get("config") or {}
    if isinstance(config, dict) and config.get("install_only"):
        return "install_only_not_scored", None, None
    exception = payload.get("exception_info")
    if exception:
        return "framework_exception", None, str(exception)
    verifier = payload.get("verifier_result") or {}
    reward = None
    if isinstance(verifier, dict):
        reward = numeric_reward(verifier.get("reward"))
    reward = reward if reward is not None else reward_from_logs(trial_dir)
    agent = payload.get("agent_result") or {}
    metadata = agent.get("metadata") if isinstance(agent, dict) else {}
    failure = metadata.get("failure") if isinstance(metadata, dict) else None
    # Harbor's top-level summary may default an interrupted trial to zero.
    # A declared Agent/sidecar failure happened before a trustworthy verifier
    # submission, so it is infrastructure evidence even if that placeholder
    # value is present in the result object.
    if failure:
        return "agent_or_sidecar_failure", None, str(failure)
    if reward is None:
        return "no_numeric_reward", None, None
    if reward == 0:
        return "valid_zero_reward", reward, None
    return "numeric_reward", reward, None


def row_for(path: Path, root: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if not payload or not payload.get("task_name"):
        return None
    status, reward, note = classify(payload, path.parent)
    return {
        "task_name": payload.get("task_name"),
        "trial_name": payload.get("trial_name"),
        "status": status,
        "reward": reward,
        "note": note,
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "source_root": str(root),
        "result_path": str(path),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Harbor scoring diagnostics",
        "",
        "This table is artifact-derived. `valid_zero_reward` means a numeric zero was found; it does not by itself attribute the zero to model capability.",
        "",
        "| Task | Trial | Classification | Reward | Note | Result |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        reward = "" if row["reward"] is None else f"{row['reward']:.8g}"
        note = str(row["note"] or "").replace("|", "\\|").replace("\n", " ")[:180]
        lines.append(
            f"| {row['task_name']} | {row['trial_name'] or ''} | {row['status']} | {reward} | {note} | `{row['result_path']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in args.jobs_dir:
        root = root.resolve()
        if not root.is_dir():
            continue
        for result_path in root.rglob("result.json"):
            row = row_for(result_path, root)
            if row and row["result_path"] not in seen:
                seen.add(row["result_path"])
                rows.append(row)
    rows.sort(key=lambda row: (str(row["task_name"]), str(row["finished_at"] or ""), str(row["result_path"])))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"schema_version": "chembench-harbor-diagnostics/1.0", "trials": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(rows), encoding="utf-8")
    print(json.dumps({"status": "PASS", "trial_count": len(rows), "json": str(args.output_json), "markdown": str(args.output_markdown)}))


if __name__ == "__main__":
    main()
