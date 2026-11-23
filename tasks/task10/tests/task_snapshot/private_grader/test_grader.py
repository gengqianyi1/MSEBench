from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from grader import grade


class GraderAttackTests(unittest.TestCase):
    def _case(self, rows, labels=None, pipeline=None):
        labels = labels or [("a", 1.0, "valid"), ("b", 2.0, "valid"), ("c", 3.0, "test")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = root / "submission"
            private = root / "private"
            submission.mkdir(); private.mkdir()
            with (submission / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["id", "prediction_eV"]); writer.writerows(rows)
            with (private / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["id", "gap_expt_eV", "split"]); writer.writerows(labels)
            (submission / "pipeline.json").write_text(json.dumps(pipeline or {"estimator": "fixture"}), encoding="utf-8")
            # A forged result file is deliberately ignored by the grader.
            (submission / "result.json").write_text('{"test_MAE_eV":0,"valid_R2":1}', encoding="utf-8")
            return grade(SimpleNamespace(submission_dir=submission, private_data_dir=private, run_dir=root, task_dir=root))

    def test_correct_predictions_and_forged_metrics(self):
        result = self._case([("a", 1.0), ("b", 2.0), ("c", 3.0)])
        self.assertEqual(result["score"], 100.0)

    def test_row_reordering_is_supported(self):
        result = self._case([("c", 3.0), ("a", 1.0), ("b", 2.0)])
        self.assertEqual(result["score"], 100.0)

    def test_missing_prediction_rejected(self):
        self.assertIn("MISSING_IDS", self._case([("a", 1.0), ("b", 2.0)])["failure_codes"])

    def test_duplicate_id_rejected(self):
        self.assertIn("DUPLICATE_ID", self._case([("a", 1.0), ("a", 1.0), ("c", 3.0)])["failure_codes"])

    def test_unknown_id_rejected(self):
        self.assertIn("UNKNOWN_IDS", self._case([("a", 1.0), ("b", 2.0), ("c", 3.0), ("x", 0.0)])["failure_codes"])


if __name__ == "__main__":
    unittest.main()
