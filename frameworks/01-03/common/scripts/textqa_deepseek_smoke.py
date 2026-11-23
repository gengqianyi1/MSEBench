#!/usr/bin/env python3
"""Run one TextQA item through DeepSeek and its unmodified Docker grader."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_call(prompt: str) -> tuple[dict, dict]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        # DeepSeek-V4-Flash uses completion tokens for hidden reasoning before
        # TextQA requires a compact JSON answer.  In thinking mode the provider
        # may consume the whole output budget on invisible reasoning and leave
        # nothing for the parser, so this fixed benchmark profile disables it.
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return payload, body
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise
            last_error = exc
        time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def extract_submission(response: dict) -> dict:
    content = response["choices"][0]["message"].get("content") or ""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        # A provider may truncate an optional, long reasoning string after it
        # has already emitted the complete answer field.  Recover only that
        # unambiguous leading field; never infer an answer from reasoning.
        match = re.match(r'\s*\{\s*"answer"\s*:\s*("(?:\\.|[^"\\])*")', stripped)
        if not match:
            raise
        value = {"answer": json.loads(match.group(1)), "reasoning": "[truncated by provider]"}
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        raise ValueError("INVALID_SUBMISSION_SCHEMA")
    return {"answer": value["answer"], "reasoning": value.get("reasoning", "")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = ROOT / "textqa" / "generated" / "textqa_v001"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    item = next(entry for entry in manifest["tasks"] if entry["position"] == args.position)
    task = dataset / item["task_directory"]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    workspace, logs = output / "workspace", output / "verifier_logs"
    workspace.mkdir(parents=True)
    prompt = (task / "instruction.md").read_text(encoding="utf-8")
    payload, response = model_call(prompt)
    (output / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "raw_response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    submission = extract_submission(response)
    (workspace / "final_answer.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    image = "chembench/textqa-smoke-verifier:1.0"
    subprocess.run(["docker", "build", "-t", image, str(task / "tests")], check=True)
    subprocess.run([
        "docker", "run", "--rm", "-e", "CHEMBENCH_WORKSPACE=/workspace",
        "-e", "CHEMBENCH_LOGS=/logs/verifier", "-v", f"{workspace}:/workspace",
        "-v", f"{logs}:/logs", "-v", f"{task / 'tests'}:/task:ro", image,
        "sh", "/tests/test.sh",
    ], check=True)
    reward = json.loads((logs / "verifier" / "reward.json").read_text(encoding="utf-8"))
    details = json.loads((logs / "verifier" / "details.json").read_text(encoding="utf-8"))
    summary = {
        "task": item["task_name"], "position": args.position, "model": "deepseek-v4-flash",
        "thinking_mode": "disabled", "response_format": "json_object", "max_tokens": 2048,
        "submission_valid": True, "verifier_status": details.get("status"),
        "score": details["private_result"]["score"], "reward": reward.get("reward"),
        "usage": response.get("usage"), "retry_policy": "up to 2 only for 429/5xx/network timeout",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
