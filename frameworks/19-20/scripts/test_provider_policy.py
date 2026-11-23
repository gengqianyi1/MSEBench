import subprocess
import sys
import unittest
from pathlib import Path
from provider_policy import ProviderFailure, retryable, matrix_complete

ROOT = Path(__file__).resolve().parents[1]


class ProviderPolicyTests(unittest.TestCase):
    def test_balance_is_permanent_and_typed(self):
        self.assertFalse(retryable(402))
        error = ProviderFailure(402)
        self.assertEqual(error.code, 402)
        self.assertEqual(error.failure_code, 'PROVIDER_INSUFFICIENT_BALANCE')

    def test_only_transient_errors_retry(self):
        for status in (400, 401, 403, 404, 422, 200):
            self.assertFalse(retryable(status))
        for status in (None, 408, 409, 429, 500, 503):
            self.assertTrue(retryable(status))

    def test_partial_infra_or_duplicate_matrix_not_success(self):
        rows = [dict(model=m, task=t, status='COMPLETED', score=25)
                for m in ('deepseek', 'kimi', 'glm') for t in (19, 20)]
        self.assertTrue(matrix_complete(rows))
        self.assertFalse(matrix_complete(rows[:-1]))
        self.assertFalse(matrix_complete(rows[:-1] + [rows[0]]))
        rows[-1].update(status='INFRASTRUCTURE_FAILURE', score=None)
        self.assertFalse(matrix_complete(rows))

    def test_invalid_selection_rejected_before_docker_or_api(self):
        for script in ('run_llm.py', 'run_docker_model_matrix.py'):
            process = subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                '--campaign', 'unused', '--model', 'deepseek'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn('must be supplied together', process.stderr)


if __name__ == '__main__':
    unittest.main()
