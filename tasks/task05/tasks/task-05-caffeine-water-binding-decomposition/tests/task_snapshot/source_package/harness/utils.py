"""Shared I/O, API, execution and workspace utilities."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def load_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="gbk", errors="replace")


def load_json(path: str | Path) -> dict | list:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(source.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def ensure_within(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and ensure it is located below *root*."""
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {path}") from exc
    return resolved


class LLMClient:
    """Small OpenAI-compatible chat-completions client with bounded retries."""

    def __init__(self, config: dict):
        self.url = self._build_url(config["url"])
        self.api_key = config["api_key"]
        self.model = config.get("model", "unknown")
        self.default_temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 16000)
        self.timeout = config.get("request_timeout_seconds", 180)
        self.max_retries = int(config.get("max_retries", 2))

    @staticmethod
    def _build_url(url: str) -> str:
        base = url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    def call(self, messages: list[dict], temperature: float | None = None, extra_body: dict | None = None) -> dict:
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM API error {response.status_code}: {response.text[:500]}")
                last_error = RuntimeError(f"LLM API error {response.status_code}: {response.text[:500]}")
            except requests.RequestException as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"LLM API failed after retries: {last_error}")


def validate_python_source(source: str, max_bytes: int, block_network: bool) -> None:
    if len(source.encode("utf-8")) > max_bytes:
        raise ValueError(f"Generated script exceeds {max_bytes} bytes")
    if block_network:
        forbidden = re.compile(
            r"(^|\n)\s*(?:from|import)\s+(?:requests|urllib|socket|httpx|aiohttp)\b|"
            r"\b(?:requests|urllib|socket|httpx|aiohttp)\s*\.",
            re.MULTILINE,
        )
        if forbidden.search(source):
            raise ValueError("Network-related Python modules are disabled for this task")


def run_command(cmd: list[str], workdir: str | Path, timeout: int = 600, env: dict | None = None) -> dict:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(workdir).resolve()),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
        )
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[-50000:],
            "stderr": result.stderr[-10000:],
            "wall_seconds": round(time.time() - start, 2),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": (exc.stdout or "")[-50000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"Command timed out after {timeout}s",
            "wall_seconds": timeout,
        }


def run_python_script(script_path: str | Path, workdir: str | Path, python_path: str = "python", timeout: int = 600, env: dict | None = None) -> dict:
    script = Path(script_path).resolve()
    workdir_path = Path(workdir).resolve()
    result = run_command([python_path, str(script)], workdir=workdir_path, timeout=timeout, env=env)
    result["result_json"] = None
    result_json_path = workdir_path / "result.json"
    if result_json_path.exists():
        try:
            result["result_json"] = load_json(result_json_path)
        except Exception:
            pass
    return result


def extract_json_block(text: str) -> dict | None:
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    candidate = match.group(1) if match else text.strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def create_workspace(run_dir: Path, task_dir: Path) -> Path:
    """Copy only Agent-visible task assets into a fresh workspace."""
    workspace = (run_dir / "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    private_names = {"rubric.json", "baseline", "private_grader", "private_data", "output", "__pycache__"}
    for item in task_dir.iterdir():
        if item.name in private_names or item.name.startswith("."):
            continue
        destination = workspace / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            shutil.copy2(item, destination)
    return workspace


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return records
