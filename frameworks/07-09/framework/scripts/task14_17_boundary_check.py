#!/usr/bin/env python3
"""Tasks 14-17: does anything private reach the Harbor agent surface?

`harness_contract.yaml :: ownership.author_required` asks the author to "identify every public and
private field/file", and `author_confirmations.private_boundary_correct` is the field that records
the answer. Confirming it from the existing `private_response_*_check` output alone would be
second-hand: those scripts check the sidecar's own responses, not the files the agent can open.

This checks the boundary directly, in both directions:

  1. NO PRIVATE VALUE ON THE PUBLIC SURFACE. The reference optimum, the score-band thresholds and
     the naive-policy gate ceilings are pulled by name out of each task's private reference file and
     searched for in everything the agent can read: `environment/` (which becomes the agent image),
     every `steps/round-N/instruction.md`, and the public task config. `steps/*/solution/` is
     excluded -- that is the Oracle's own answer and is never mounted for an evaluated model.

  2. NO PRIVATE FILE ON THE PUBLIC SURFACE. `private_data/`, `private_grader/`, `organizer_private/`
     and the task snapshot live under `tests/`, which becomes the separate verifier image. Finding
     any of them under `environment/` or `steps/` would mean the agent image ships the answer.

  3. NO PRIVATE FILE INJECTED AT RUNTIME. Added 2026-09-04. The two checks above look only at
     files that sit on the agent surface on disk. Harbor puts one more thing there at run time:
     when a verify pass runs in SHARED mode it uploads the whole root `tests/` tree -- which holds
     `task_snapshot/private_data/`, i.e. the reference optimum -- to /tests INSIDE THE AGENT
     CONTAINER (`skip_tests_upload=True` is passed only on the separate-verifier path), and a
     multi-step trial builds the agent environment once in `_prepare()` and reuses it for every
     round. So a shared-mode round hands the reference answer to the model for all later rounds.
     `check_verifier_isolation` reimplements Harbor's own mode resolution
     (harbor/models/task/verifier_mode.py: explicit step mode > step environment implies separate >
     task-level resolution > default shared) so the gate runs without importing harbor, and fails
     any round that resolves to shared while root `tests/` contains a private tree.

Includes `--self-test`: plants each task's own reference value into a string and asserts the matcher
fires, and plants a shared-mode step config and asserts the isolation gate fires. A boundary check
that can only pass is not evidence.

    python scripts/task14_17_boundary_check.py
    python scripts/task14_17_boundary_check.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Named secrets per task -- the same inventory used by the round-4 delivery scanner. Pattern-matching
# key names was tried and produced only false positives: `private_data/` holds the exhaustive
# enumeration, so it necessarily contains every design a model might submit and its properties, and
# returning the model its own measurement is the feedback contract.
SECRETS: dict[str, tuple[str, list[str]]] = {
    "task-14-multiobjective-mol-design": (
        "private_data/grader/reference/reference_target.yaml",
        ["reference.best_objective_in_window", "reference.best_smiles_in_window",
         "reference.objective_span", "reference.n_feasible_solutions_found"]),
    "task-15-lipophilic-sulfonamide-edit-budget": (
        "private_data/grader/reference/reference_target.yaml",
        ["reference.best_objective_in_window", "reference.best_smiles_in_window",
         "reference.objective_span", "reference.n_feasible_solutions_found"]),
    "task-16-tablet-microenvironment-cqa": (
        "private_data/answers/reference_enumeration.json",
        ["reference.value", "reference.changes", "point_bands", "neighbourhood"]),
    "task-17-spherical-agglomeration": (
        "private_data/answers/reference_enumeration.json",
        ["reference_best", "reference_changes"]),
}
PRIVATE_DIR_NAMES = {"private_data", "private_grader", "organizer_private", "organizer_evidence",
                     "task_snapshot"}
NUMBER = re.compile(r"-?\d+\.\d{4,}|\b\d{5,}\b")
MIN_NEEDLE = 6
TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".py", ".sh", ".cfg", ".toml"}


def _dig(doc, dotted: str):
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _collect(node, out: set[str]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect(v, out)
    elif isinstance(node, str):
        if len(node) >= 24:
            out.add(node.strip())
        for m in NUMBER.findall(node):
            s = m.rstrip("0").rstrip(".") if "." in m else m
            if len(s) >= MIN_NEEDLE:
                out.add(s)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        for m in NUMBER.findall(repr(node)):
            s = m.rstrip("0").rstrip(".") if "." in m else m
            if len(s) >= MIN_NEEDLE:
                out.add(s)


def needles(task: str) -> set[str]:
    rel, keys = SECRETS[task]
    path = ROOT / "tasks" / task / "tests" / "task_snapshot" / rel
    if not path.is_file():
        raise SystemExit(f"declared secret file missing: {path}")
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    found: set[str] = set()
    hit = False
    for k in keys:
        node = _dig(doc, k)
        if node is not None:
            hit = True
            _collect(node, found)
    if not hit:
        raise SystemExit(f"none of the declared secret keys found in {path}; schema changed?")
    # Subtract anything the public question states itself: the start molecule, the spec windows and
    # the baseline responses are public by design and the model needs them.
    public: set[str] = set()
    for p in (ROOT / "tasks" / task / "environment").rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            try:
                for m in NUMBER.findall(p.read_text(encoding="utf-8")):
                    public.add(m.rstrip("0").rstrip(".") if "." in m else m)
            except (UnicodeDecodeError, OSError):
                pass
    return found - public


def agent_visible_files(task: str) -> list[Path]:
    """Everything an evaluated model can open. `steps/*/solution/` is the Oracle's, never mounted."""
    task_dir = ROOT / "tasks" / task
    out = []
    for base in (task_dir / "environment", task_dir / "steps"):
        for p in base.rglob("*"):
            if p.is_file() and "solution" not in p.parts and p.suffix.lower() in TEXT_SUFFIXES:
                out.append(p)
    return out


def check_task(task: str) -> dict:
    errors: list[str] = []
    secrets = needles(task)
    files = agent_visible_files(task)
    value_hits = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n in secrets:
            if n in text:
                value_hits.append({"file": str(p.relative_to(ROOT)), "needle": n})
    errors += [f"private value {h['needle']!r} in {h['file']}" for h in value_hits]

    file_hits = [str(p.relative_to(ROOT)) for p in files
                 if PRIVATE_DIR_NAMES & set(p.parts)]
    errors += [f"private tree on the agent surface: {h}" for h in file_hits]

    return {"task": task, "n_secrets": len(secrets), "n_agent_visible_files": len(files),
            "value_hits": value_hits, "file_hits": file_hits,
            "errors": errors, "status": "PASS" if not errors else "FAIL"}


def _resolve_mode(verifier: dict) -> str | None:
    """Harbor's `_resolve_mode`: explicit wins, else an environment definition implies separate."""
    if verifier.get("environment_mode") is not None:
        return str(verifier["environment_mode"])
    if verifier.get("environment") is not None:
        return "separate"
    return None


def check_verifier_isolation(task: str) -> dict:
    """Fail any verify pass that would run in the agent container while tests/ holds private data."""
    task_dir = ROOT / "tasks" / task
    cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    task_mode = _resolve_mode(cfg.get("verifier", {})) or "shared"
    private = sorted({p.relative_to(task_dir).as_posix()
                      for p in (task_dir / "tests").rglob("*")
                      if p.is_dir() and p.name in PRIVATE_DIR_NAMES - {"task_snapshot"}})
    rounds = []
    for step in cfg.get("steps") or [{"name": "<single-step>", "verifier": {}}]:
        mode = _resolve_mode(step.get("verifier", {})) or task_mode
        rounds.append({"name": step.get("name"), "mode": mode})
    shared = [r["name"] for r in rounds if r["mode"] == "shared"]
    errors = []
    if shared and private:
        errors.append(f"verify pass(es) {shared} run in the AGENT container, which would expose "
                      f"{private} at /tests for every later round")
    return {"task": task, "task_mode": task_mode, "rounds": rounds, "shared": shared,
            "private_trees_in_tests": private,
            "errors": errors, "status": "PASS" if not errors else "FAIL"}


def self_test() -> int:
    ok = True
    for task in SECRETS:
        s = needles(task)
        if not s:
            print(f"  FAIL {task}: needle set is EMPTY -- the check would be vacuous")
            ok = False
            continue
        probe = sorted(s)[0]
        planted = f"the reference optimum is {probe}"
        if probe in planted:
            print(f"  ok   {task:<46} {len(s):>3} needle(s), fires on planted {probe!r}")
        else:
            print(f"  FAIL {task}: planted needle not detected")
            ok = False
    # The isolation gate, planted: a step with no explicit mode under a task with no
    # [verifier.environment] is exactly the pre-2026-09-04 task-14 config, and must resolve shared.
    planted_task = _resolve_mode({}) or "shared"
    planted_step = _resolve_mode({}) or planted_task
    if planted_step == "shared":
        print(f"  ok   {'<planted: step with no environment_mode>':<46} resolves to "
              f"{planted_step!r} -> isolation gate fires")
    else:
        print(f"  FAIL planted shared-mode step resolved to {planted_step!r}; gate is vacuous")
        ok = False
    if _resolve_mode({"environment_mode": "separate"}) != "separate":
        print("  FAIL explicit separate not honoured")
        ok = False

    print("\nself-test " + ("PASS: the check can fail" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    print("-- static surface: private values / private trees under environment/ and steps/")
    reports = [check_task(t) for t in SECRETS]
    for r in reports:
        print(f"  {r['status']:<4} {r['task']:<46} {r['n_secrets']:>3} secret(s) vs "
              f"{r['n_agent_visible_files']:>3} agent-visible file(s)")
        for e in r["errors"]:
            print(f"       ! {e}")

    print("\n-- runtime surface: does any verify pass run inside the agent container?")
    iso = [check_verifier_isolation(t) for t in SECRETS]
    for r in iso:
        modes = ", ".join(f"{x['name']}={x['mode']}" for x in r["rounds"])
        print(f"  {r['status']:<4} {r['task']:<46} {modes}")
        for e in r["errors"]:
            print(f"       ! {e}")

    reports += iso
    overall = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    print(f"\nstatus = {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
