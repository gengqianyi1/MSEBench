#!/usr/bin/env python3
"""Generate an immutable, all-at-once TextQA Harbor dataset from UTF-8 JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = {"id", "category_1", "category_2", "material_category", "question_type", "question", "answer", "explanation"}
ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_public(row: dict) -> dict:
    return {key: row[key] for key in ("category_1", "category_2", "material_category", "question_type", "question")}


def choice_letters(value: str) -> list[str]:
    return [char for char in value.upper() if "A" <= char <= "Z"]


def ordering_tokens(value: str) -> list[str]:
    """The supplied corpus has one numeric ordering item (2 3 1 4 5)."""
    digits = __import__("re").findall(r"\d+", value)
    return digits or choice_letters(value)


def validate_row(row: dict, line: int) -> None:
    missing = REQUIRED - set(row)
    if missing:
        raise ValueError(f"line {line}: missing fields {sorted(missing)}")
    if not isinstance(row["id"], (str, int)):
        raise ValueError(f"line {line}: id must be string or integer")
    for key in REQUIRED - {"id"}:
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"line {line}: {key} must be non-empty text")
    kind = row["question_type"]
    tokens = ordering_tokens(row["answer"]) if "排序" in kind else choice_letters(row["answer"])
    if not tokens:
        raise ValueError(f"line {line}: answer has no scoreable option tokens")
    if "单选" in kind and len(set(tokens)) != 1:
        raise ValueError(f"line {line}: single-choice answer must contain exactly one option")


def task_toml(name: str, index: int, content_hash: str, version: str) -> str:
    return f'''schema_version = "1.4"
artifacts = ["/workspace"]

[task]
name = "chembench/{name}"
version = "1.0.0"
description = "No-tool material-science TextQA item {index:04d}"
authors = []
keywords = ["textqa", "multiple-choice", "chembench", "no-tools"]

[metadata]
source_task_id = "{version}_line_{index:04d}"
content_hash = "{content_hash}"
domain = "material-science-textqa"
execution_mode = "one-shot-llm-no-tools"
required_artifacts = ["final_answer.json"]
release_eligible = true

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[environment]
network_mode = "no-network"
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 512
storage_mb = 1024
gpus = 0

[environment.healthcheck]
command = "python --version"
interval_sec = 5.0
timeout_sec = 30.0
retries = 2

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 512
storage_mb = 1024
gpus = 0
'''


ENV_DOCKERFILE = '''FROM python:3.12-slim
RUN mkdir -p /workspace /logs/agent /logs/artifacts /logs/verifier
WORKDIR /workspace
COPY public/ /workspace/
CMD ["bash"]
'''

TEST_DOCKERFILE = '''FROM python:3.12-slim
RUN mkdir -p /workspace /logs/verifier /tests /task /run
COPY private_reference.json /task/private_reference.json
COPY grader.py /tests/grader.py
COPY test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
WORKDIR /workspace
CMD ["bash"]
'''

TEST_SH = "#!/bin/sh\nset -eu\npython /tests/grader.py\n"

GRADER = r'''#!/usr/bin/env python3
"""Private TextQA scorer: matches the supplied 100/50/0 policy."""
from __future__ import annotations
import json, os, re
from pathlib import Path

WORKSPACE = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
REFERENCE = Path("/task/private_reference.json")

def letters(value):
    # This intentionally scores only the declared answer field, never the
    # reasoning/raw response. It reproduces the supplied scorer's option-set
    # semantics without its unsafe whole-response fallback.
    seen = set(); result = []
    for char in re.findall(r"[A-Za-z]", str(value or "")):
        char = char.upper()
        if char not in seen:
            seen.add(char); result.append(char)
    return result

def ordering_tokens(value):
    digits = re.findall(r"\d+", str(value or ""))
    return digits or letters(value)

def score(kind, correct, predicted):
    expected, actual = letters(correct), letters(predicted)
    if "单选" in kind:
        return 100 if expected and actual and expected[0] == actual[0] else 0
    if "多选" in kind:
        want, got = set(expected), set(actual)
        if want == got: return 100
        if got and got.issubset(want): return 50
        return 0
    if "排序" in kind:
        expected, actual = ordering_tokens(correct), ordering_tokens(predicted)
        return 100 if expected and expected == actual else 0
    return 100 if expected and actual and expected[0] == actual[0] else 0

def write(name, value):
    LOGS.mkdir(parents=True, exist_ok=True)
    tmp = LOGS / (name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, LOGS / name)

def main():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    submission_path = WORKSPACE / "final_answer.json"
    predicted, error = "", None
    try:
        submitted = json.loads(submission_path.read_text(encoding="utf-8"))
        if not isinstance(submitted, dict) or not isinstance(submitted.get("answer"), str):
            error = "INVALID_SUBMISSION_SCHEMA"
        else:
            predicted = submitted["answer"]
    except FileNotFoundError:
        error = "MISSING_FINAL_ANSWER"
    except (json.JSONDecodeError, UnicodeDecodeError):
        error = "INVALID_SUBMISSION_JSON"
    raw_score = 0 if error else score(ref["question_type"], ref["answer"], predicted)
    reward = raw_score / 100.0
    private = {"score": raw_score, "subscores": {}, "failure_codes": [error] if error else [], "evidence": {"question_id": ref["source_id"], "question_type": ref["question_type"], "predicted_answer": predicted}}
    details = {"schema_version": "textqa-verifier-details/1.0", "status": "scored", "task_id": ref["task_id"], "private_result": private}
    write("details.json", details)
    write("grader-result.json", {"schema_version": "chembench-grader-result/1.0", "reward": reward, "raw_score": raw_score, "status": "scored"})
    write("reward.json", {"reward": reward})
    (LOGS / "reward.txt").write_text(str(reward), encoding="utf-8")
    print(json.dumps({"task_id": ref["task_id"], "score": raw_score, "reward": reward}))

if __name__ == "__main__": main()
'''


def write_task(destination: Path, row: dict, index: int, content_hash: str, version: str) -> str:
    name = f"{version.replace('_', '-')}-{index:04d}-{content_hash[:12]}"
    destination.mkdir(parents=True)
    (destination / "task.toml").write_text(task_toml(name, index, content_hash, version), encoding="utf-8")
    public = canonical_public(row)
    (destination / "environment" / "public").mkdir(parents=True)
    (destination / "environment" / "Dockerfile").write_text(ENV_DOCKERFILE, encoding="utf-8")
    (destination / "environment" / "public" / "question.json").write_text(json.dumps({"schema_version": "textqa-question/1.0", "source_id": row["id"], **public}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = f'''You are answering one closed-book material-science multiple-choice question. No tools are available.\n\n题目类型：{row["question_type"]}\n\n题目：\n{row["question"]}\n\nReturn exactly one JSON object:\n{{"answer":"<option letters only>","reasoning":"<brief reasoning>"}}\nFor multiple-choice use letters such as "A、B、D". For ordering preserve the requested order.\n'''
    (destination / "instruction.md").write_text(prompt, encoding="utf-8")
    tests = destination / "tests"; tests.mkdir()
    private = {"task_id": f"chembench/{name}", "source_id": row["id"], "question_type": row["question_type"], "answer": row["answer"], "explanation": row["explanation"]}
    (tests / "private_reference.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tests / "grader.py").write_text(GRADER, encoding="utf-8")
    (tests / "Dockerfile").write_text(TEST_DOCKERFILE, encoding="utf-8")
    # Harbor verifier images run Linux /bin/sh.  Write explicit LF bytes even
    # when generation is launched from Windows.
    (tests / "test.sh").write_bytes(TEST_SH.encode("utf-8"))
    os.chmod(tests / "test.sh", 0o755)
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--version", default="textqa_v001")
    parser.add_argument("--output-root", type=Path, default=ROOT / "textqa" / "generated")
    parser.add_argument("--force", action="store_true", help="replace only this unreferenced generation directory")
    args = parser.parse_args()
    if not __import__("re").fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.version):
        raise ValueError("version must contain only letters, digits, underscores, and hyphens")
    raw = args.input.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line); validate_row(row, number); rows.append((number, row))
    if not rows:
        raise ValueError("input contains no questions")
    destination = args.output_root / args.version
    if destination.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {destination}; use a new --version")
    stage_parent = destination.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=args.version + ".staging-", dir=stage_parent))
    try:
        tasks = stage / "tasks"; tasks.mkdir()
        manifest_rows = []
        names = set()
        for position, (line, row) in enumerate(rows, 1):
            digest = sha256_bytes(json.dumps(canonical_public(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            name = write_task(tasks / f"textqa-{position:04d}-{digest[:12]}", row, position, digest, args.version)
            if name in names: raise ValueError(f"duplicate public question at source line {line}")
            names.add(name)
            manifest_rows.append({"position": position, "source_line": line, "source_id": row["id"], "task_name": name, "task_directory": f"tasks/textqa-{position:04d}-{digest[:12]}", "content_sha256": digest, "question_type": row["question_type"], "category_1": row["category_1"], "category_2": row["category_2"], "material_category": row["material_category"]})
        # The source JSONL contains answer and explanation fields. Preserve its
        # name and digest in the manifest, but never copy it into the benchmark
        # package where a task author or model-facing build could disclose it.
        manifest = {"schema_version": "textqa-generation-manifest/1.0", "version": args.version, "created_at": datetime.now(timezone.utc).isoformat(), "source_file_name": args.input.name, "source_sha256": sha256_bytes(raw), "question_count": len(rows), "tasks": manifest_rows}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "dataset.yaml").write_text("schema_version: textqa-dataset/1.0\nversion: " + args.version + "\ntask_directories:\n" + "".join(f"  - {entry['task_directory']}\n" for entry in manifest_rows), encoding="utf-8")
        if destination.exists(): shutil.rmtree(destination)
        os.replace(stage, destination)
        latest = args.output_root.parent / "latest.json"
        latest_tmp = latest.with_suffix(".tmp")
        latest_tmp.write_text(json.dumps({"schema_version": "textqa-latest/1.0", "version": args.version, "dataset": str(destination.relative_to(args.output_root.parent)).replace("\\", "/")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(latest_tmp, latest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    print(json.dumps({"status": "generated", "version": args.version, "output": str(destination), "questions": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
