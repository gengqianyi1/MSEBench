"""Offline integration/package contract checks; not a Docker or Harbor run."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'frameworks/19-20/scripts'))
from task_paths import task_path
from provider_policy import matrix_complete

class IntegratedChecks(unittest.TestCase):
    def test_tasks_resolve_without_symlinks(self):
        for n in (19, 20):
            self.assertEqual(task_path(n), ROOT / 'tasks' / f'task{n}')
            self.assertTrue((task_path(n) / 'tests/fixtures/oracle.json').is_file())

    def test_six_build_contexts_and_copy_inputs_exist(self):
        for n in (19, 20):
            manifest = yaml.safe_load((ROOT / f'images/task{n}/images.yaml').read_text())
            self.assertEqual(len(manifest['images']), 3)
            for image in manifest['images']:
                recipe = image['build']
                self.assertNotIn('\\', recipe['context'])
                context = ROOT / recipe['context']
                dockerfile = context / recipe['dockerfile']
                lines = dockerfile.read_text().splitlines()
                self.assertIn('FROM python:3.11.9-slim', lines)
                for line in lines:
                    if line.startswith('COPY '):
                        for source in shlex.split(line)[1:-1]:
                            self.assertTrue((context / source).exists(), str(context / source))
                if image['role'] == 'environment':
                    self.assertTrue(all(not line.startswith('COPY tests') for line in lines))

    def test_extended_matrix_completeness(self):
        models = ('deepseek', 'glm', 'kimi', 'ling', 'qwen', 'deepseek41')
        rows = [dict(model=m, task=n, score=25, status='COMPLETED') for m in models for n in (19, 20)]
        self.assertTrue(matrix_complete(rows, models))
        self.assertFalse(matrix_complete(rows[:-1], models))

    def test_public_and_private_budget_agree(self):
        for n in (19, 20):
            task = task_path(n)
            suite = json.loads((task / 'tests/task_snapshot/private_data/suite.json').read_text())
            self.assertEqual(suite['budget']['max_tool_calls'], 4)
            self.assertEqual(suite['budget']['max_agent_turns'], 6 if n == 19 else 5)

if __name__ == '__main__':
    unittest.main()
