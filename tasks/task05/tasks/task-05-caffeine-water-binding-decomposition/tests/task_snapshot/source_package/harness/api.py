"""Stable Python API for third-party runners and submission-only evaluation."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from .config import TaskConfig
from .evaluator import evaluate_submission
from .utils import load_json, save_json


def list_tasks(tasks_root: Path | str = "tasks") -> list[dict]:
    root = Path(tasks_root).resolve()
    records = []
    for task_dir in sorted(root.iterdir()):
        config_file = task_dir / "task.yaml"
        if task_dir.is_dir() and config_file.is_file():
            cfg = TaskConfig.from_yaml(config_file)
            records.append({
                "task_id": cfg.task_id,
                "title": cfg.title,
                "version": cfg.version,
                "domain": cfg.domain,
                "author": cfg.author,
                "task_dir": str(task_dir),
            })
    return records


def find_task(instance_id_or_path: str, tasks_root: Path | str = "tasks") -> Path:
    direct = Path(instance_id_or_path)
    if direct.is_dir() and (direct / "task.yaml").is_file():
        return direct.resolve()
    root = Path(tasks_root).resolve()
    named = root / instance_id_or_path
    if named.is_dir() and (named / "task.yaml").is_file():
        return named.resolve()
    for record in list_tasks(root):
        if record["task_id"] == instance_id_or_path:
            return Path(record["task_dir"])
    raise FileNotFoundError(f"Task not found: {instance_id_or_path}")


def evaluate_existing_submission(
    instance_id: str,
    submission_dir: Path | str,
    tasks_root: Path | str = "tasks",
    output_dir: Path | str | None = None,
) -> dict:
    task_dir = find_task(instance_id, tasks_root)
    cfg = TaskConfig.from_yaml(task_dir / "task.yaml")
    rubric_file = task_dir / cfg.rubric_path
    rubric = load_json(rubric_file) if rubric_file.is_file() else None
    run_dir = Path(output_dir).resolve() if output_dir else Path(submission_dir).resolve().parent
    score = evaluate_submission(
        run_dir=run_dir,
        submission_dir=Path(submission_dir),
        task_cfg=cfg,
        rubric=rubric,
        task_dir=task_dir,
        execution_status="EXTERNAL_SUBMISSION",
        run_manifest={"run_id": None},
    )
    if output_dir:
        save_json(Path(output_dir) / "score.json", score)
    return score


def aggregate_scores(runs_dir: Path | str, output_path: Path | str | None = None) -> dict:
    """Aggregate only scientifically scored, valid submissions.

    The task-local score is retained for per-task reporting, while
    ``comparison_score`` is used for cross-task summaries.  Invalid
    submissions, infrastructure failures and unscored runs are counted by
    status and never silently inserted as numeric zeroes.
    """
    records: list[dict] = []
    excluded: list[dict] = []
    status_counts: dict[str, int] = {}
    score_files = sorted(Path(runs_dir).rglob("score.json"))
    for score_file in score_files:
        try:
            score = load_json(score_file)
        except Exception:
            excluded.append({"path": str(score_file), "reason": "UNREADABLE"})
            status_counts["UNREADABLE"] = status_counts.get("UNREADABLE", 0) + 1
            continue
        status_key = "/".join(str(score.get(key, "UNKNOWN")) for key in ("execution_status", "submission_status", "grading_status"))
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        comparison = score.get("comparison_score")
        local = score.get("task_local_score", score.get("score"))
        valid = (
            score.get("submission_status") == "SUBMITTED"
            and score.get("grading_status") == "SCORED"
            and isinstance(comparison, (int, float))
            and isinstance(local, (int, float))
        )
        if valid:
            records.append(score)
        else:
            excluded.append({
                "path": str(score_file),
                "reason": status_key,
                "failure_codes": score.get("failure_codes", []),
            })

    grouped: dict[tuple[str, str], list[dict]] = {}
    for score in records:
        key = (str(score.get("task_id", "unknown")), str(score.get("task_version", "unknown")))
        grouped.setdefault(key, []).append(score)

    submitted_count = sum(count for key, count in status_counts.items() if "/SUBMITTED/" in key)
    total_count = len(score_files)
    summary = {
        "schema_version": "1.2",
        "aggregation_field": "comparison_score",
        "total_runs": total_count,
        "valid_scientific_runs": len(records),
        "valid_runs": len(records),
        "excluded_runs": len(excluded),
        "valid_submission_rate": round(submitted_count / total_count, 8) if total_count else None,
        "scientific_scoring_rate": round(len(records) / total_count, 8) if total_count else None,
        "mean_score": round(mean(float(item["comparison_score"]) for item in records), 6) if records else None,
        "mean_comparison_score": round(mean(float(item["comparison_score"]) for item in records), 6) if records else None,
        "mean_task_local_score": round(mean(float(item.get("task_local_score", item["score"])) for item in records), 6) if records else None,
        "status_counts": dict(sorted(status_counts.items())),
        "by_task_version": {
            f"{task_id}@{version}": {
                "count": len(items),
                "comparison_mean": round(mean(float(item["comparison_score"]) for item in items), 6),
                "comparison_min": min(float(item["comparison_score"]) for item in items),
                "comparison_max": max(float(item["comparison_score"]) for item in items),
                "task_local_mean": round(mean(float(item.get("task_local_score", item["score"])) for item in items), 6),
            }
            for (task_id, version), items in sorted(grouped.items())
        },
        "excluded": excluded,
    }
    if output_path:
        save_json(output_path, summary)
    return summary
