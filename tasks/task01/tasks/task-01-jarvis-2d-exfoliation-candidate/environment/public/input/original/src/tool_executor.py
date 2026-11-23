from __future__ import annotations

import ast
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Set

try:
    from .utils import tail
except ImportError:
    from utils import tail


FORBIDDEN_IMPORT_ROOTS: Set[str] = {
    "socket",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "http",
    "httpx",
    "aiohttp",
    "urllib3",
    "websocket",
    "paramiko",
    "chgnet",
    "importlib",
}

FORBIDDEN_CALLS: Set[str] = {
    "os.system",
    "os.remove",
    "os.rmdir",
    "os.removedirs",
    "os.unlink",
    "os.chmod",
    "os.chown",
    "os.kill",
    "shutil.rmtree",
    "__import__",
}

ALLOWED_SUBPROCESS_BINS: Set[str] = {"python", "python.exe", "python3"}


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def _literal_subprocess_binary(args_node: ast.AST) -> str | None:
    if isinstance(args_node, ast.List) and args_node.elts:
        first = args_node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return Path(first.value).name.lower()
    if isinstance(args_node, ast.Tuple) and args_node.elts:
        first = args_node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return Path(first.value).name.lower()
    return None


def safety_check(script: str, max_script_size_bytes: int) -> None:
    if len(script.encode("utf-8")) > max_script_size_bytes:
        raise ValueError(f"script exceeds max size of {max_script_size_bytes} bytes")

    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            name = _attr_name(node.func)
            if name in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {name}")
            if name in {"subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output"}:
                if node.args:
                    binary = _literal_subprocess_binary(node.args[0])
                    if binary and binary not in ALLOWED_SUBPROCESS_BINS:
                        raise ValueError(f"subprocess binary is not allowed: {binary}")


def _snapshot_files(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    }


def _changed_files(before: Set[str], after: Set[str]) -> list[str]:
    return sorted(after - before)


def run_python_tool(
    *,
    script: str,
    timeout_seconds: int,
    workspace: Path,
    project_root: Path,
    python_executable: str,
    max_script_runtime_seconds: int,
    max_script_size_bytes: int,
    stdout_stderr_tail_chars: int,
    tool_index: int,
) -> Dict[str, Any]:
    timeout = min(int(timeout_seconds or max_script_runtime_seconds), max_script_runtime_seconds)
    workspace = workspace.resolve()
    project_root = project_root.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    safety_check(script, max_script_size_bytes=max_script_size_bytes)

    script_path = workspace / f"tool_run_{tool_index:03d}.py"
    script_path.write_text(script, encoding="utf-8")

    before = _snapshot_files(workspace)
    # Do not pass provider credentials or arbitrary host environment variables
    # into agent-authored Python. Production still requires container-level
    # network and filesystem isolation.
    tmp_dir = workspace / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            # Required by Python extension modules and subprocess startup on Windows.
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
        )
        if key in os.environ
    }
    env.update({
        "HOME": str(workspace),
        "TMPDIR": str(tmp_dir),
        "MPLCONFIGDIR": str(workspace / ".matplotlib"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    start = time.time()
    try:
        proc = subprocess.run(
            [python_executable, str(script_path)],
            cwd=workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        timed_out = False
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    elapsed = time.time() - start
    after = _snapshot_files(workspace)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": tail(stdout, stdout_stderr_tail_chars),
        "stderr": tail(stderr, stdout_stderr_tail_chars),
        "script_file": str(script_path.relative_to(workspace)),
        "created_files": _changed_files(before, after),
        "result_json_exists": (workspace / "result.json").exists(),
    }
