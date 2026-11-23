#!/usr/bin/env python3
"""Validate every task contract (publishable-benchmark gate).

Unified core schema (see docs/TASK_CONTRACT.md):
  core (all families):            task, family, protocol, entry
  non-builtin (harbor/chembench/organizer): + instruction, task_dir, agent_ref
  harbor:    + verifier, harbor_task
  chembench: + verifier
  organizer: + organizer_task
  textqa (builtin, no agent/verifier/task_dir): core only

`result_glob` is intentionally NOT enforced: task14-17 legitimately omit it
(their results are pinned at frameworks/14-17/jobs/ in consolidate_results.task1417).

Exit 0 = all valid; nonzero = list of violations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"

CORE = ("task", "family", "protocol", "entry")
NON_BUILTIN = ("instruction", "task_dir", "agent_ref")
FAMILY_EXTRA = {
    "harbor": ("verifier", "harbor_task"),
    "chembench": ("verifier",),
    "organizer": ("organizer_task",),
    "textqa": (),
}
FAMILIES = set(FAMILY_EXTRA)


def _agent_ref_problem(task_name, ref):
    """agent_ref 可达性：路径形(.py)须存在于 ROOT；module:Class 形须可解析到 src/ 或 frameworks/。"""
    core = (ref or "").strip().split(" (")[0].strip()  # 去掉 "(frozen...)" 注释
    if not core:
        return None
    if core.endswith(".py"):
        return None if (ROOT / core).is_file() else f"{task_name}: agent_ref path missing: {core}"
    if ":" in core:
        module = core.split(":")[0]
        if module.startswith("msebench"):
            p = ROOT / "src" / (module.replace(".", "/") + ".py")
            return None if p.is_file() else f"{task_name}: agent_ref module missing: {module}"
        fname = module.split(".")[-1] + ".py"
        if any(ROOT.glob(f"frameworks/**/{fname}")):
            return None
        return f"{task_name}: agent_ref module missing: {module}"
    return None


def check_task(task_dir: Path):
    problems = []
    spec_path = task_dir / "task.json"
    if not spec_path.is_file():
        return [f"{task_dir.name}: missing task.json"]
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{task_dir.name}: task.json invalid: {exc}"]

    family = spec.get("family")
    if family not in FAMILIES:
        return [f"{task_dir.name}: unknown family {family!r}"]

    for field in CORE:
        if not spec.get(field):
            problems.append(f"{task_dir.name}: missing core field '{field}'")
    if family != "textqa":
        for field in NON_BUILTIN:
            if not spec.get(field):
                problems.append(f"{task_dir.name}: missing field '{field}'")
    for field in FAMILY_EXTRA[family]:
        if not spec.get(field):
            problems.append(f"{task_dir.name}: {family} missing field '{field}'")

    protocol = spec.get("protocol") or {}
    if protocol.get("disposition") not in {"plugin", "builtin"}:
        problems.append(f"{task_dir.name}: protocol.disposition must be plugin|builtin")
    if protocol.get("disposition") == "plugin" and not protocol.get("module"):
        problems.append(f"{task_dir.name}: plugin protocol needs 'module'")

    # 题面：文件型（本地或 task_dir 指向）；镜像型显式声明
    instr = spec.get("instruction", "file")
    if instr != "image":
        candidates = [task_dir]
        rel = spec.get("task_dir")
        if rel:
            candidates.append((task_dir / rel).resolve())
        if not any((c / "instruction.md").is_file() or list(c.rglob("instruction.md")) for c in candidates if c.exists()):
            problems.append(f"{task_dir.name}: missing instruction.md (题面)")
    ref_problem = _agent_ref_problem(task_dir.name, spec.get("agent_ref"))
    if ref_problem:
        problems.append(ref_problem)
    return problems


def main() -> int:
    if not TASKS.is_dir():
        print(f"no tasks dir: {TASKS}")
        return 2
    problems = []
    count = 0
    for task_dir in sorted(p for p in TASKS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        count += 1
        problems.extend(check_task(task_dir))
    if problems:
        print(f"INVALID ({len(problems)} problems across {count} tasks):")
        for p in problems:
            print(" -", p)
        return 1
    print(f"OK: {count} task contracts valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
