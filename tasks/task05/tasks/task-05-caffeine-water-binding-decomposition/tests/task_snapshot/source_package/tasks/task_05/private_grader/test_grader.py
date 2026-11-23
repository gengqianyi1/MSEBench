from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from grader import _implementation, grade


class GraderFixtureTests(unittest.TestCase):
    def test_bridge_resolves_actual_scientific_grader(self):
        module = _implementation()
        self.assertTrue(callable(module.grade))
        self.assertIn("private_data/evaluator_bundle/grader.py", Path(module.__file__).as_posix())

    def test_empty_submission_is_rejected_by_actual_grader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = root / "submission"
            run_dir = root / "run"
            submission.mkdir()
            run_dir.mkdir()
            result = grade(SimpleNamespace(
                submission_dir=submission,
                private_data_dir=None,
                run_dir=run_dir,
                task_dir=Path(__file__).resolve().parents[1],
                run_manifest={},
            ))
            self.assertEqual(float(result["score"]), 0.0)
            self.assertTrue(result.get("failure_codes"))
            self.assertTrue((run_dir / "explain_report.json").is_file())

    def test_golden_fixture_scores_full_credit_when_available(self):
        task_dir = Path(__file__).resolve().parents[1]
        project_root = task_dir.parents[1]
        golden = project_root / "golden" / task_dir.name / "repetition_01" / "submission"
        harness_manifest = project_root / "golden" / task_dir.name / "repetition_01" / "harness_manifest_for_grader.json"
        if not golden.is_dir() or not harness_manifest.is_file():
            self.skipTest("full package Golden fixture is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.loads(harness_manifest.read_text(encoding="utf-8"))
            result = grade(SimpleNamespace(
                submission_dir=golden,
                private_data_dir=task_dir / "private_data",
                run_dir=root,
                task_dir=task_dir,
                run_manifest=manifest,
            ))
            self.assertEqual(float(result["score"]), 100.0)
            self.assertTrue((root / "canonical_replay.json").is_file())


if __name__ == "__main__":
    unittest.main()
