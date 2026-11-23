#!/usr/bin/env python3
"""Regression tests for Task 01 Harbor-controlled efficiency accounting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


GRADER_PATH = Path(__file__).with_name("run_legacy_grader.py")


def load_grader():
    spec = importlib.util.spec_from_file_location("task01_legacy_grader", GRADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EfficiencyAccountingTests(unittest.TestCase):
    def run_case(self, accounting: dict | None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            task = root / "task"
            work = root / "workspace"
            logs = root / "logs"
            run = root / "run"
            task.mkdir()
            work.mkdir()
            (task / "grader.py").write_text(
                "def grade(context):\n    return {'score': 100.0}\n", encoding="utf-8"
            )
            (task / "task.yaml").write_text(
                "grader_path: grader.py\nprivate_data_path: private_data\n"
                "max_score: 100\npass_score: 1\n",
                encoding="utf-8",
            )
            if accounting is not None:
                (run / "llm_request_accounting.json").parent.mkdir()
                (run / "llm_request_accounting.json").write_text(
                    json.dumps(accounting), encoding="utf-8"
                )

            module = load_grader()
            module.TASK, module.WORK, module.LOG, module.RUN = task, work, logs, run
            # The boundary test supplies a synthetic task-local grader.  Public
            # replay-input copying is exercised by the full Oracle test and is
            # unrelated to control-plane request accounting.
            module._provision_public_replay_inputs = lambda: None
            return (module.main(),
                    json.loads((logs / "reward.json").read_text(encoding="utf-8")),
                    json.loads((logs / "details.json").read_text(encoding="utf-8")))

    def test_request_count_boundaries(self):
        for count, expected, gate in (
            (39, 1.0, "PASS"),
            (40, 1.0, "PASS"),
            (41, 0.0, "LLM_REQUEST_LIMIT_EXCEEDED"),
            (94, 0.0, "LLM_REQUEST_LIMIT_EXCEEDED"),
        ):
            with self.subTest(actual_llm_requests=count):
                status, reward, details = self.run_case(
                    {"actual_llm_requests": count, "counting_authority": "harbor_control_plane"}
                )
                self.assertEqual(status, 0)
                self.assertAlmostEqual(reward["efficiency"], expected)
                self.assertAlmostEqual(reward["final_score"], 100 * expected)
                self.assertEqual(details["llm_request_gate"], gate)
                self.assertEqual(reward["reward"], 1.0 if expected else 0.0)

    def test_missing_accounting_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing Harbor control-plane"):
            self.run_case(None)

    def test_untrusted_accounting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Harbor control plane"):
            self.run_case({"actual_llm_requests": 40, "counting_authority": "agent"})


if __name__ == "__main__":
    unittest.main()
