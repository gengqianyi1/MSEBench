import asyncio
import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('integration', ROOT / 'scripts/run_task19_20_verifier_docker_integration.py')
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


class Captured(Exception):
    pass


class RecoveryTest(unittest.TestCase):
    def check_ids(self, ids):
        cls = integration._agent_class()
        agent = cls.__new__(cls)
        received = []
        tool_requests = []
        cfg = dict(sidecar_url_env='TEST_UNUSED_SIDECAR', sidecar_url='http://127.0.0.1:1',
                   candidate_required_fields=['answer'], use_native_tool_calls=True)

        async def read(*args, **kwargs):
            return ''

        def post(url, payload):
            self.assertTrue(url.endswith('/session'))
            tool_requests.append(url)
            return dict(session_id='s', case_count=1, case={'budget': {'max_agent_turns': 3, 'max_tool_calls': 4}})

        def action(messages, *args):
            received.append(copy.deepcopy(messages))
            if len(received) == 2:
                raise Captured()
            agent._last_finish_reason = 'length'
            return 'submit_candidate', None, '', dict(role='assistant', content=None,
                tool_calls=[dict(id=i, type='function', function=dict(name='submit_candidate', arguments='{"answer":')) for i in ids])

        agent._fixed_read = read
        agent._post_json = post
        agent._call_tool_action = action
        with self.assertRaises(Captured):
            asyncio.run(agent._run_hidden_suite_case('case 1', object(), SimpleNamespace(metadata={}), cfg))
        replies = [m for m in received[1] if m['role'] == 'tool']
        self.assertEqual([m['tool_call_id'] for m in replies], ids)
        self.assertTrue(all(json.loads(m['content'])['error'] == 'OUTPUT_LIMIT_TRUNCATED' for m in replies))
        self.assertEqual(len(tool_requests), 1)  # session only; no experiment executed

    def test_single_call(self):
        self.check_ids(['call_one'])

    def test_all_calls_in_truncated_batch(self):
        self.check_ids(['call_one', 'call_two'])


if __name__ == '__main__':
    unittest.main()
