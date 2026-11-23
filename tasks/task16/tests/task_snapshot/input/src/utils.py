\
from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_run_dir(root: str | Path, task_id: str, label: str | None = None) -> Path:
    """Create a fresh run directory named <label>_<task_id>_<timestamp>.

    `label` is normally the model name, so the three trajectories a delivery needs are told apart by
    who produced them rather than by a timestamp a reviewer has to cross-reference.

    The timestamp has one-second resolution, so two runs launched in the same second used to collide
    and the second one died with FileExistsError. A numeric suffix is appended instead.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") if label else ""
    stem = f"{slug}_{task_id}_{timestamp}" if slug else f"{task_id}_{timestamp}"
    root = Path(root)
    for attempt in range(100):
        candidate = root / (stem if attempt == 0 else f"{stem}_{attempt + 1}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {root} for {stem!r}")


def copy_file(src: str | Path, dst: str | Path) -> None:
    shutil.copy2(src, dst)


def set_seed(seed: int) -> None:
    random.seed(seed)
