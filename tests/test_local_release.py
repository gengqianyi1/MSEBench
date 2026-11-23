"""Regression checks for local workspace assembly and verifier selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_local
import score_local


class LocalReleaseTests(unittest.TestCase):
    def fixture(self, root):
        code, data = root / "code", root / "data"
        code.mkdir()
        data.mkdir()
        payload = b'{"answer": 42}\n'
        resource = data / "tasks/task04/reference.json"
        resource.parent.mkdir(parents=True)
        resource.write_bytes(payload)
        document = {
            "version": "0.2.0", "release_mode": "local-evaluation",
            "files": [{"path": "tasks/task04/reference.json", "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest(), "task": "task04"}],
            "workspace_links": {"frameworks/family/task04": "../../tasks/task04"},
        }
        for directory in (code, data):
            (directory / "data_manifest.json").write_text(json.dumps(document))
        return code, data

    def test_prepare_preserves_paths_and_restores_internal_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            code, data = self.fixture(root)
            output = root / "workspace"
            prepare_local.prepare(code, output, source=data)
            self.assertEqual((output / "frameworks/family/task04/reference.json").read_bytes(),
                             (data / "tasks/task04/reference.json").read_bytes())
            self.assertEqual(json.loads((output / "workspace.json").read_text())["resource_files"], 1)

    def test_prepare_rejects_corruption_without_publishing_partial_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            code, data = self.fixture(root)
            (data / "tasks/task04/reference.json").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare_local.prepare(code, root / "workspace", source=data)
            self.assertFalse((root / "workspace").exists())
            self.assertFalse(list(root.glob(".msebench-prepare-*")))

    def test_prepare_rejects_link_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            code, data = self.fixture(root)
            doc = json.loads((code / "data_manifest.json").read_text())
            doc["workspace_links"] = {"escape": "../../external"}
            for directory in (code, data):
                (directory / "data_manifest.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "External workspace link"):
                prepare_local.prepare(code, root / "workspace", source=data)
            self.assertFalse((root / "workspace").exists())

    def test_score_rejects_wrong_verifier_before_writing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            submission = root / "submission"
            submission.mkdir()
            lock = root / "images/task10/images.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(json.dumps({"images": {"task10-verifier.tar": {
                "tag": "msebench/task10-verifier:0.2.0", "image_id": "sha256:expected",
            }}}))
            with patch.object(score_local, "ROOT", root), patch.object(
                score_local.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "sha256:wrong\n"),
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    score_local.score(10, submission, root / "score")
            self.assertFalse((root / "score").exists())

    def test_score_uses_task07_sidecar_and_preserves_failed_run_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            submission, evidence = root / "submission", root / "evidence"
            submission.mkdir()
            evidence.mkdir()
            lock = root / "images/task07/images.lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_text(json.dumps({"images": {"task07-sidecar.tar": {
                "tag": "msebench/task07-sidecar:0.2.0", "image_id": "sha256:expected",
            }}}))
            replies = [subprocess.CompletedProcess([], 0, "sha256:expected\n"),
                       subprocess.CompletedProcess([], 2, "", "missing evidence record")]
            with patch.object(score_local, "ROOT", root), patch.object(
                score_local.subprocess, "run", side_effect=replies,
            ) as docker:
                self.assertEqual(score_local.score(7, submission, root / "score", evidence), 2)
            command = docker.call_args.args[0]
            self.assertEqual(command[-2:], ["msebench/task07-sidecar:0.2.0", "/tests/test.sh"])
            self.assertEqual((root / "score/stderr.txt").read_text(), "missing evidence record")
            self.assertEqual(list(submission.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
