#!/usr/bin/env python3
"""Build a transparent canonical view from immutable per-case campaigns.

The provider balance was exhausted part-way through the formal campaign.  This
script does not turn blocked cases into scores and does not replay a model.  It
selects the latest scientifically complete case when one exists, otherwise the
final permanent-provider-block receipt, copies the immutable evidence, and
runs the independent evaluator again over the copy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate_multicase_run_v3 import canonical_bytes, evaluate_case
from evaluator.evaluate_suite import aggregate_suite


TARGET = ROOT / "m" / "canonical_20260805"
MODELS = ("deepseek", "kimi", "glm")
SOURCE_CAMPAIGNS = {
    "deepseek": {
        "2_fluorobiphenyl_central_torsion_v1": "r5",
        "butane_central_torsion_v1": "r5",
        "difluoroethane_fccf_torsion_v1": "r5",
        "n_methylacetamide_cn_torsion_v1": "r5",
    },
    "kimi": {
        "2_fluorobiphenyl_central_torsion_v1": "r5",
        "butane_central_torsion_v1": "r5",
        "difluoroethane_fccf_torsion_v1": "r5",
        "n_methylacetamide_cn_torsion_v1": "r5",
    },
    "glm": {
        "2_fluorobiphenyl_central_torsion_v1": "r5",
        "butane_central_torsion_v1": "r1",
        "difluoroethane_fccf_torsion_v1": "r1",
        "n_methylacetamide_cn_torsion_v1": "r1",
    },
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).hexdigest().upper().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def main() -> int:
    if TARGET.exists() or TARGET.is_symlink():
        raise FileExistsError("canonical evidence target already exists")
    suite = _json(ROOT / "organizer_private" / "suite.json")
    cases = suite["cases"]
    by_case = {str(case["case_id"]): case for case in cases}
    expected = list(by_case)
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        records: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for case_id in expected:
            campaign = SOURCE_CAMPAIGNS[model][case_id]
            source = ROOT / "m" / campaign / model / case_id
            destination = TARGET / model / case_id
            if not source.is_dir():
                raise FileNotFoundError(source)
            source_tree = _tree_sha256(source)
            shutil.copytree(source, destination)
            copied_tree = _tree_sha256(destination)
            if copied_tree != source_tree:
                raise RuntimeError("copied canonical evidence tree differs from source")
            replay = evaluate_case(destination, by_case[case_id], project_root=ROOT)
            stored = _json(destination / "evaluation.json")
            if canonical_bytes(replay) != canonical_bytes(stored):
                raise RuntimeError("canonical independent replay differs from frozen evaluation")
            _write(destination / "canonical_replay_evaluation.json", replay)
            records.append(replay)
            row = {
                "logical_model": model,
                "case_id": case_id,
                "source_campaign": campaign,
                "source_relative_path": f"m/{campaign}/{model}/{case_id}",
                "source_tree_sha256": source_tree,
                "copied_tree_sha256_before_canonical_annotation": copied_tree,
                "canonical_replay_match": True,
                "status": replay["status"],
                "official_score": replay["official_score"],
                "provider_status": replay["provider_status"],
            }
            source_rows.append(row)
            summary_rows.append(row)
        _write(TARGET / model / "source_receipts.json", {"schema_version": "3.0", "cases": source_rows})
        _write(TARGET / model / "suite_evaluation.json", aggregate_suite(records, expected))
    _write(
        TARGET / "campaign_summary.json",
        {
            "schema_version": "3.0",
            "construction": "immutable_case_selection_plus_independent_replay",
            "selection_policy": (
                "latest scientifically complete exact-model case if available; "
                "otherwise final permanent-provider-block receipt"
            ),
            "blocked_cases_are_not_numeric_zero": True,
            "case_count_per_model": len(expected),
            "run_count": len(summary_rows),
            "all_canonical_replays_match": all(row["canonical_replay_match"] for row in summary_rows),
            "rows": summary_rows,
        },
    )
    print(json.dumps({"target": str(TARGET), "run_count": len(summary_rows), "all_replays_match": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
