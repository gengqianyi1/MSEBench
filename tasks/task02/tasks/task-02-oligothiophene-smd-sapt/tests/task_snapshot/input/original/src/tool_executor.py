from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Set

from utils import tail


FORBIDDEN_IMPORT_ROOTS: Set[str] = {
    "socket", "urllib", "requests", "ftplib", "smtplib", "telnetlib", "http", "ctypes", "cffi",
}
FORBIDDEN_DIRECT_CHEM_IMPORTS: Set[str] = {"pyscf", "psi4", "xtb"}
FORBIDDEN_CALLS: Set[str] = {
    "os.system", "os.popen", "os.remove", "os.rmdir", "os.removedirs", "os.unlink",
    "os.chmod", "os.chown", "os.kill", "shutil.rmtree",
}
FORBIDDEN_CALL_SUFFIXES: Set[str] = {"unlink", "rmdir", "chmod", "chown", "kill"}
FORBIDDEN_DYNAMIC_EXECUTION: Set[str] = {"exec", "eval", "compile"}
ALLOWED_SUBPROCESS_BINS: Set[str] = set()
ALLOWED_BENCHMARK_TOOL_FUNCTIONS: Set[str] = {
    "run_xtb_microtest", "run_xtb_optimization", "run_xtb_single_point",
    "run_pyscf_microtest", "run_pyscf_formal_single_point",
    "run_psi4_microtest", "run_psi4_sapt0",
}
PROTECTED_WORKSPACE_FILES: Set[str] = {
    "instance.json", "protocol.yaml", "benchmark_tools.py", "stage_A_schema.json", "stage_B_schema.json",
    "stage_C_schema.json", "result_schema.json", "stage_validation.py",
}
SUBPROCESS_CALLS: Set[str] = {
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_call", "subprocess.check_output",
}


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def _literal_subprocess_binary(args_node: ast.AST) -> str | None:
    if isinstance(args_node, (ast.List, ast.Tuple)) and args_node.elts:
        first = args_node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return Path(first.value).name.lower()
    return None


def _keyword_bool(call: ast.Call, keyword: str) -> bool:
    for item in call.keywords:
        if item.arg == keyword and isinstance(item.value, ast.Constant):
            return bool(item.value.value)
    return False


def safety_check(script: str, max_script_size_bytes: int) -> None:
    if len(script.encode("utf-8")) > max_script_size_bytes:
        raise ValueError(f"script exceeds max size of {max_script_size_bytes} bytes")
    tree = ast.parse(script)
    benchmark_module_aliases: set[str] = {"benchmark_tools"}
    benchmark_function_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "benchmark_tools":
                    benchmark_module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "benchmark_tools":
            for alias in node.names:
                if alias.name not in ALLOWED_BENCHMARK_TOOL_FUNCTIONS:
                    raise ValueError(f"benchmark_tools public API violation: {alias.name}")
                benchmark_function_aliases[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(f"forbidden import: {alias.name}")
                if root in FORBIDDEN_DIRECT_CHEM_IMPORTS:
                    raise ValueError(f"direct chemistry import is forbidden; use benchmark_tools.py: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {node.module}")
            if root in FORBIDDEN_DIRECT_CHEM_IMPORTS:
                raise ValueError(f"direct chemistry import is forbidden; use benchmark_tools.py: {node.module}")
        elif isinstance(node, ast.Call):
            name = _attr_name(node.func)
            if name:
                parts = name.split(".")
                if parts[0] in benchmark_module_aliases:
                    if len(parts) != 2 or parts[1] not in ALLOWED_BENCHMARK_TOOL_FUNCTIONS:
                        raise ValueError(f"benchmark_tools public API violation: {name}")
                elif len(parts) == 1 and parts[0] in benchmark_function_aliases:
                    pass
            if name in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {name}")
            # Reject both bare builtins (``exec``) and qualified aliases such
            # as ``builtins.exec``.  The latter previously bypassed the exact
            # name comparison while providing the same dynamic-code surface.
            if name and name.split(".")[-1] in FORBIDDEN_DYNAMIC_EXECUTION:
                raise ValueError(f"dynamic code execution is forbidden: {name}")
            if name in {"__import__", "importlib.import_module"} and node.args:
                module_node = node.args[0]
                if isinstance(module_node, ast.Constant) and isinstance(module_node.value, str):
                    root = module_node.value.split(".")[0]
                    if root in FORBIDDEN_IMPORT_ROOTS or root in FORBIDDEN_DIRECT_CHEM_IMPORTS:
                        raise ValueError(f"forbidden dynamic import: {module_node.value}")
            if name and name.split(".")[-1] in FORBIDDEN_CALL_SUFFIXES:
                raise ValueError(f"forbidden destructive/process-control call: {name}")
            if name in SUBPROCESS_CALLS:
                if _keyword_bool(node, "shell"):
                    raise ValueError("subprocess shell=True is forbidden")
                raise ValueError("candidate subprocess calls are forbidden; use the current run_python script and benchmark_tools.py")


def _script_markers(script: str) -> Dict[str, Any]:
    call_names: set[str] = set()
    try:
        tree = ast.parse(script)
    except SyntaxError:
        tree = None
    aliases: dict[str, str] = {}
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "benchmark_tools":
                for alias in node.names:
                    aliases[(alias.asname or alias.name).lower()] = alias.name.lower()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _attr_name(node.func)
                if name:
                    short = name.split(".")[-1].lower()
                    call_names.add(aliases.get(short, short))

    microtest_markers = {
        "pyscf": "run_pyscf_microtest",
        "psi4": "run_psi4_microtest",
        "xtb": "run_xtb_microtest",
    }
    production_markers = {
        "pyscf": "run_pyscf_formal_single_point",
        "psi4": "run_psi4_sapt0",
        "xtb": "run_xtb_optimization",
    }
    micro_tools = [tool for tool, marker in microtest_markers.items() if marker in call_names]
    production_tools = [tool for tool, marker in production_markers.items() if marker in call_names]
    if "run_xtb_single_point" in call_names and "xtb" not in production_tools:
        production_tools.append("xtb")
    tools = sorted(set(micro_tools) | set(production_tools))
    return {
        "external_tools": tools,
        "micro_tools": sorted(set(micro_tools)),
        "production_tools": sorted(set(production_tools)),
        "microtest_detected": bool(micro_tools) and not production_tools,
        "production_launch_detected": bool(production_tools),
    }


def timeout_floor_for_script(script: str, policy: Dict[str, Any]) -> tuple[str, int]:
    markers = _script_markers(script)
    production = markers["production_tools"]
    micro = markers["micro_tools"]
    if production:
        if len(production) > 1:
            timeout_class = "mixed_production"
        elif production[0] == "xtb":
            timeout_class = "xtb_production"
        elif production[0] == "pyscf":
            timeout_class = "pyscf_production"
        else:
            timeout_class = "psi4_production"
    elif micro:
        timeout_class = "microtest"
    else:
        timeout_class = "pure_python"
    key = f"{timeout_class}_floor_seconds"
    defaults = {
        "pure_python": 60,
        "microtest": 300,
        "xtb_production": 1800,
        "pyscf_production": 7200,
        "psi4_production": 7200,
        "mixed_production": 7200,
    }
    return timeout_class, int(policy.get(key, defaults[timeout_class]))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_files(root: Path) -> Dict[str, Dict[str, Any]]:
    if not root.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        stat = p.stat()
        out[rel] = {"sha256": _sha256(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return out


def _file_changes(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> tuple[list[str], list[str]]:
    created = sorted(set(after) - set(before))
    modified = sorted(
        rel for rel in set(after) & set(before)
        if after[rel]["sha256"] != before[rel]["sha256"]
    )
    return created, modified


def _ensure_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _classify_script(script: str, stderr: str, returncode: int) -> Dict[str, Any]:
    markers = _script_markers(script)
    error_text = stderr.lower()
    model_error_tokens = ["traceback", "syntaxerror", "typeerror", "attributeerror", "indexerror", "keyerror", "nameerror"]
    return {
        "external_tools": markers["external_tools"],
        "microtest_detected": markers["microtest_detected"],
        "production_launch_detected": markers["production_launch_detected"],
        "micro_tools": markers["micro_tools"],
        "production_tools": markers["production_tools"],
        "model_caused_error_suspected": returncode != 0 and any(token in error_text for token in model_error_tokens),
    }


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
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()

    before = _snapshot_files(workspace)
    protected_bytes = {
        rel: (workspace / rel).read_bytes()
        for rel in PROTECTED_WORKSPACE_FILES
        if (workspace / rel).exists() and (workspace / rel).is_file()
    }
    env = os.environ.copy()
    env["BENCHMARK_WORKSPACE"] = str(workspace)

    start = time.time()
    try:
        guard_code = f"""
import os, runpy, sys
from pathlib import Path
_workspace = Path({str(workspace)!r}).resolve()
_project = Path({str(project_root)!r}).resolve()
_script = Path({str(script_path)!r}).resolve()
sys.path.insert(0, str(_workspace))

def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def _audit(event, args):
    if event != 'open' or not args:
        return
    value = args[0]
    if not isinstance(value, (str, bytes, os.PathLike)):
        return
    try:
        path = Path(value).resolve()
    except Exception:
        return
    if _inside(path, _project) and not _inside(path, _workspace):
        raise PermissionError(f'candidate access outside workspace is forbidden: {{path}}')

sys.addaudithook(_audit)
runpy.run_path(str(_script), run_name='__main__')
"""
        proc = subprocess.run(
            [python_executable, "-I", "-c", guard_code],
            cwd=workspace,
            text=True,
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
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)

    elapsed = time.time() - start
    protected_violations: list[str] = []
    for rel, original in protected_bytes.items():
        target = workspace / rel
        if not target.exists() or target.read_bytes() != original:
            protected_violations.append(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original)
    if protected_violations:
        returncode = -1
        stderr = (stderr + "\n" if stderr else "") + "Protected benchmark inputs were modified and restored: " + ", ".join(sorted(protected_violations))
    after = _snapshot_files(workspace)
    created, modified = _file_changes(before, after)
    classification = _classify_script(script, stderr, returncode)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": tail(stdout, stdout_stderr_tail_chars),
        "stderr": tail(stderr, stdout_stderr_tail_chars),
        "script_file": str(script_path.relative_to(workspace)),
        "script_sha256": script_hash,
        "created_files": created,
        "modified_files": modified,
        "protected_file_violation": sorted(protected_violations),
        **classification,
        "result_json_exists": (workspace / "result.json").exists(),
    }
