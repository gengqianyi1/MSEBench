#!/usr/bin/env python3
"""Formally evaluate TextQA v001 (positions 01--61) with DeepSeek and its graders.

The runner never overwrites an existing run directory.  Every item gets its own
raw provider response, extracted submission, verifier logs, and summary.  A
successful API call is graded once only; retries are reserved for transient
transport/provider errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import time
import traceback
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from textqa_deepseek_smoke import ROOT, extract_submission, model_call
from textqa_shared_verifier import SharedTextQAVerifier, make_private_index


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def common_verifier_image(dataset: Path, tasks: list[dict]) -> str:
    """Return a cached verifier image after proving all task runtimes match.

    References remain private per task and are mounted at runtime.  Reusing an
    image is therefore safe only when Dockerfile, grader and launcher bytes are
    identical across the immutable generation.
    """
    asset_names = ("Dockerfile", "grader.py", "test.sh")
    fingerprints: dict[str, set[str]] = {name: set() for name in asset_names}
    for entry in tasks:
        tests = dataset / entry["task_directory"] / "tests"
        for name in asset_names:
            fingerprints[name].add(hashlib.sha256((tests / name).read_bytes()).hexdigest())
    divergent = {name: sorted(values) for name, values in fingerprints.items() if len(values) != 1}
    if divergent:
        raise RuntimeError(f"TextQA verifier assets differ; cannot share environment: {divergent}")
    runtime_hash = hashlib.sha256("".join(next(iter(fingerprints[name])) for name in asset_names).encode()).hexdigest()[:16]
    image = f"chembench/textqa-verifier:{runtime_hash}"
    inspected = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspected.returncode:
        canonical = dataset / tasks[0]["task_directory"] / "tests"
        subprocess.run(["docker", "build", "-t", image, str(canonical)], check=True)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    dataset = ROOT / "textqa" / "generated" / "textqa_v001"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    tasks = sorted(manifest["tasks"], key=lambda entry: entry["position"])
    expected_positions = list(range(1, 62))
    positions = [entry["position"] for entry in tasks]
    if positions != expected_positions:
        raise RuntimeError(f"TextQA manifest positions are not 01--61: {positions}")

    output.mkdir(parents=True)
    # The private answer index is mounted only into one persistent verifier
    # container. The model-facing process never receives this file.
    private_dir = Path(tempfile.mkdtemp(prefix="textqa-private-index-"))
    private_index = make_private_index(dataset, private_dir / "reference_index.json")
    verifier = SharedTextQAVerifier(private_index)
    run_meta = {
        "schema_version": "textqa-formal-run/1.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "thinking_mode": "disabled",
        "response_format": "json_object",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "request_timeout_seconds": 120,
        "retry_policy": "at most 2 attempts; only HTTP 429/5xx or network timeout/reset",
        "task_count": len(tasks),
        "shared_verifier_image": verifier.image,
    }
    write_json(output / "run_config.json", run_meta)
    results: list[dict] = []

    for entry in tasks:
        position = entry["position"]
        task = dataset / entry["task_directory"]
        item_dir = output / f"{position:02d}_{entry['task_name']}"
        workspace, logs = item_dir / "workspace", item_dir / "verifier_logs"
        started = datetime.now(timezone.utc)
        result = {
            "id": f"{position:02d}", "task": entry["task_name"], "model": "deepseek-v4-flash",
            "started_at": started.isoformat(), "answer_generated": False,
            "submission_valid": False, "grader_status": "NOT_STARTED", "score": None,
            "reward": None, "error": None,
        }
        try:
            workspace.mkdir(parents=True)
            prompt = (task / "instruction.md").read_text(encoding="utf-8")
            payload, response = model_call(prompt)
            write_json(item_dir / "request.json", payload)
            write_json(item_dir / "raw_response.json", response)
            result["answer_generated"] = bool(response.get("choices", [{}])[0].get("message", {}).get("content"))
            submission = extract_submission(response)
            write_json(workspace / "final_answer.json", submission)
            result["submission_valid"] = True
            graded = verifier.grade(entry["task_name"], submission["answer"])
            verifier_dir = logs / "verifier"; verifier_dir.mkdir(parents=True)
            details = {"schema_version": "textqa-verifier-details/1.0", "status": graded["status"],
                       "task_id": entry["task_name"], "private_result": {"score": graded["score"],
                       "subscores": {}, "failure_codes": graded["failure_codes"],
                       "evidence": {"question_id": position, "predicted_answer": submission["answer"]}}}
            reward = {"reward": graded["reward"]}
            write_json(verifier_dir / "details.json", details)
            write_json(verifier_dir / "grader-result.json", {"schema_version": "chembench-grader-result/1.0", "reward": graded["reward"], "raw_score": graded["score"], "status": graded["status"]})
            write_json(verifier_dir / "reward.json", reward)
            result.update({
                "grader_status": details.get("status"),
                "score": details["private_result"]["score"],
                "reward": reward.get("reward"),
                "usage": response.get("usage"),
            })
        except Exception as exc:  # Preserve failures and continue to the next independent item.
            result["error"] = f"{type(exc).__name__}: {exc}"
            (item_dir / "error.txt").parent.mkdir(parents=True, exist_ok=True)
            (item_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        result["runtime_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
        write_json(item_dir / "summary.json", result)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    verifier.close()
    shutil.rmtree(private_dir)
    valid_scores = [float(row["score"]) for row in results if isinstance(row["score"], (int, float))]
    aggregate = {
        **run_meta,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "successfully_evaluated": len(valid_scores),
        "failed_evaluation": len(results) - len(valid_scores),
        "mean_score": statistics.mean(valid_scores) if valid_scores else None,
        "median_score": statistics.median(valid_scores) if valid_scores else None,
        "zero_score_count": sum(score == 0 for score in valid_scores),
        "nonzero_count": sum(score != 0 for score in valid_scores),
        "accuracy": (sum(score == 100 for score in valid_scores) / len(valid_scores)) if valid_scores else None,
    }
    write_json(output / "results.json", {"aggregate": aggregate, "results": results})
    print(json.dumps(aggregate, ensure_ascii=False))
    return 0 if len(valid_scores) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
