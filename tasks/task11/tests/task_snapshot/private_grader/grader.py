from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("lithium_surrogate_v2", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def grade(context) -> dict:
    try:
        # 2026-09-27 fix: prefer canonical final_answer.json over candidate.json.
        # Old behavior (kept for audit): candidate.json was read first, so a
        # stale/draft candidate.json silently overrode the canonical
        # final_answer.json whenever the two files disagreed, contradicting
        # instruction.md ("the Harness saves the response's final_answer object
        # as canonical final_answer.json; candidate.json is an optional audit
        # artifact"). New behavior: final_answer.json is authoritative;
        # candidate.json is consulted only as a fallback when final_answer.json
        # is absent.
        path = context.submission_dir / "final_answer.json"
        if not path.is_file():
            path = context.submission_dir / "candidate.json"
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(candidate.get("smiles"), str):
            raise ValueError("smiles must be a string")
        smiles = candidate["smiles"].strip()
        if not smiles:
            raise ValueError("smiles must not be empty")
    except Exception:
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_CANDIDATE"], "evidence": {}}
    core = _load(Path(__file__).parent / "core" / "surrogate.py")
    reserved = {
        core.Chem.MolToSmiles(core.Chem.MolFromSmiles(line.strip()))
        for line in (context.private_data_dir / "reserved_smiles.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    core.BANNED_CANONICAL = reserved
    result = core.evaluate(smiles)
    checks = result.get("constraint_results", {})
    subscores = {key: float(value.get("points_awarded", 0.0)) for key, value in checks.items() if key in {"C1", "C2", "C3", "C4", "C5"}}
    failures = [key for key, value in checks.items() if not value.get("passed", False)]
    return {
        "score": float(result.get("score", 0.0)),
        "subscores": subscores,
        "failure_codes": failures,
        "evidence": {"canonical_smiles": checks.get("C6_leakage_control", {}).get("value", {}).get("canonical_smiles"), "properties": result.get("properties", {}), "measurement_contract": "rdkit_descriptor_surrogate_v2"},
    }
