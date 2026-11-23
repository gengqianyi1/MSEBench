#!/usr/bin/env python3
"""Replay compact, organizer-side scientific error fixtures.

The command takes the author Oracle workspace as input, copies it to a
temporary directory, applies one controlled mutation, and calls the private
grader in this verifier image.  No private reference data is copied into the
participant workspace and no network access is needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


EXPECTED = {
    "oracle": 98.75,
    "unit_mixup": 95.817,
    "solvent_order": 90.75,
    "sapt_dominance": 25.0,
}


def load_grader():
    path = Path("/task/private_grader/grader.py")
    spec = importlib.util.spec_from_file_location("task02_fixture_grader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load private grader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scale_xyz(path: Path, factor: float) -> None:
    pattern = re.compile(r"^(\s*[A-Za-z]+\s+)([-+0-9.eE]+)(\s+)([-+0-9.eE]+)(\s+)([-+0-9.eE]+)(.*)$")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values = [float(match.group(i)) * factor for i in (2, 4, 6)]
            line = (
                f"{match.group(1)}{values[0]:.16f}{match.group(3)}"
                f"{values[1]:.16f}{match.group(5)}{values[2]:.16f}{match.group(7)}"
            )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def swap_converged_energies(first: Path, second: Path) -> None:
    pattern = re.compile(r"(converged SCF energy\s*=\s*)([-+0-9.eE]+)")
    first_text, second_text = first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8")
    first_match, second_match = pattern.search(first_text), pattern.search(second_text)
    if first_match is None or second_match is None:
        raise RuntimeError("formal logs do not contain converged SCF energies")
    first_value, second_value = first_match.group(2), second_match.group(2)
    first.write_text(first_text[: first_match.start(2)] + second_value + first_text[first_match.end(2) :], encoding="utf-8")
    second.write_text(second_text[: second_match.start(2)] + first_value + second_text[second_match.end(2) :], encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mutate(workspace: Path, fixture: str) -> None:
    if fixture == "oracle":
        return
    if fixture == "unit_mixup":
        # Coordinates are submitted in bohr while the protocol interprets Å.
        for relative in (
            "structures/R_g.xyz",
            "structures/endpoint_n_hexane_selected.xyz",
            "structures/endpoint_water_selected.xyz",
        ):
            scale_xyz(workspace / relative, 1.889726)
        return
    if fixture == "solvent_order":
        swap_converged_energies(
            workspace / "logs/formal/n-hexane.log",
            workspace / "logs/formal/chloroform.log",
        )
        stage_b = load_json(workspace / "stage_B_bulk.json")
        order = stage_b["ranking"]["solvation_free_energy_order_ascending"]
        stage_b["ranking"]["solvation_free_energy_order_ascending"] = list(reversed(order))
        save_json(workspace / "stage_B_bulk.json", stage_b)
        return
    if fixture == "sapt_dominance":
        stage_c_path = workspace / "stage_C_local.json"
        stage_c = load_json(stage_c_path)
        endpoint = stage_c["endpoints"]["n-hexane"]
        endpoint["sapt"]["dominant_attractive_component"] = "electrostatics"
        for entry in endpoint["sapt_ensemble"]:
            entry["dominant_attractive_component"] = "electrostatics"
        save_json(stage_c_path, stage_c)
        result_path = workspace / "result.json"
        result = load_json(result_path)
        result["local_interactions"]["n-hexane"]["dominant_attractive_component"] = "electrostatics"
        save_json(result_path, result)
        return
    raise ValueError(f"unknown fixture: {fixture}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.oracle_workspace.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    grader = load_grader()
    rows = []
    with tempfile.TemporaryDirectory(prefix="task02-scientific-fixtures-") as temporary:
        root = Path(temporary)
        for fixture, expected in EXPECTED.items():
            workspace = root / fixture
            shutil.copytree(source, workspace)
            mutate(workspace, fixture)
            result = grader.grade(SimpleNamespace(
                task_dir=Path("/task"),
                submission_dir=workspace,
                private_data_dir=Path("/task/private_data"),
            ))
            observed = float(result["score"])
            rows.append({
                "fixture": fixture,
                "expected_score": expected,
                "observed_score": observed,
                "failure_codes": result.get("failure_codes", []),
                "pass": abs(observed - expected) <= 1e-6,
            })
    report = {"status": "PASS" if all(row["pass"] for row in rows) else "FAIL", "fixtures": rows}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
