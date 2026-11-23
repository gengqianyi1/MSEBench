"""Participant-safe smoke tests; no private case or oracle dependency."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_suite_exposes_exactly_one_public_case() -> None:
    suite = json.loads((ROOT / "configs" / "suite_public.json").read_text(encoding="utf-8"))
    assert suite["schema_version"] == "3.0"
    assert suite["task_id"] == "adaptive_xtb_torsion_profile_agent_v3"
    assert len(suite["cases"]) == 1
    assert suite["cases"][0]["case_id"] == "2_fluorobiphenyl_central_torsion_v1"
    assert suite["cases"][0]["visibility"] == "public"


def test_public_case_has_no_oracle_values() -> None:
    forbidden = {"oracle", "relative_energies_kcal_mol", "total_energies_hartree"}
    for path in (ROOT / "benchmark_case").rglob("*"):
        if not path.is_file():
            continue
        assert "oracle" not in path.name.casefold()
        if path.suffix.casefold() == ".json":
            text = path.read_text(encoding="utf-8").casefold()
            assert all(token not in text for token in forbidden)


def test_example_model_config_contains_no_credential_value() -> None:
    text = (ROOT / "configs" / "models.example.yaml").read_text(encoding="utf-8")
    assert "GJLD_API_KEY" in text
    assert "api_key:" not in text.casefold()
