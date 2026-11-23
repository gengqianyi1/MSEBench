from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, data: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, data: Any) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_run_name_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text.lower()


def make_run_dir(output_root: str | Path, prefix: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{prefix}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_file(src: str | Path, dst: str | Path) -> None:
    shutil.copy2(src, dst)


def redact_api_config(api_cfg: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(api_cfg)
    if "api_key" in safe:
        safe["api_key"] = "***REDACTED***"
    return safe


def tail(text: str, n_chars: int) -> str:
    if len(text) <= n_chars:
        return text
    return text[-n_chars:]
