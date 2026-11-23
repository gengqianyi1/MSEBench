import asyncio
import collections
import copy
import json
import unittest
from pathlib import Path
from task_paths import task_path
from types import SimpleNamespace
import test_truncated_call_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTest(unittest.TestCase):
    def test_balanced_complete_pair_space(self):
        suite = json.loads((task_path(19) / 'tests/task_snapshot/private_data/suite.json').read_text())
        pairs = suite['task']['pair_hypotheses']
        self.assertEqual(len({frozenset(p['candidate_ids']) for p in pairs}), 45)
        self.assertEqual(set(collections.Counter(c for p in pairs for c in p['candidate_ids']).values()), {9})

    def test_public_types(self):
        cls = recovery.integration._agent_class()
        for task in (task_path(19), task_path(20)):
            cfg = json.loads((task / 'environment/public/.chembench_agent.json').read_text())
            cfg['candidate_schema'] = json.loads((task / 'environment/public' / Path(cfg['candidate_schema_path']).name).read_text())
            spec = cls._hidden_suite_tool_spec(cfg)[-1]['function']['parameters']
            self.assertEqual(spec, cfg['candidate_schema'])
            self.assertEqual(spec['properties']['claims']['type'], 'array')
            self.assertEqual(spec['properties']['uncertainty']['type'], 'object')

    def test_invalid_is_not_provenance_failure(self):
        cls = recovery.integration._agent_class()
        agent = cls.__new__(cls)
        writes = {}
        captured = []
        cfg = dict(sidecar_url_env='TEST_PROTOCOL2_UNUSED', sidecar_url='http://127.0.0.1:1',
                   candidate_required_fields=['answer'], hidden_suite_tools=[dict(name='diagnostic')])
        async def read(*args, **kwargs):
            return ''
        async def write(env, path, value):
            writes[path] = copy.deepcopy(value)
        async def imports(*args):
            return {}
        def post(url, payload):
            if url.endswith('/session'):
                return dict(session_id='s', case_count=1, case=dict(case_id='test', budget=dict(max_agent_turns=1, max_tool_calls=4)))
            return dict(ok=False, status='invalid', feedback=dict(schema_valid=False))
        def action(messages, *args):
            captured.extend(copy.deepcopy(messages))
            agent._last_finish_reason = 'stop'
            return 'submit_candidate', {'answer': 'bad'}, '', dict(role='assistant', content=None,
                tool_calls=[dict(id='a', type='function', function=dict(name='submit_candidate', arguments='{"answer":"bad"}'))])
        agent._fixed_read, agent._write_workspace_file = read, write
        agent._import_sidecar_artifacts, agent._post_json, agent._call_tool_action = imports, post, action
        context = SimpleNamespace(metadata={})
        asyncio.run(agent._run_hidden_suite_case('case 1', object(), context, cfg))
        self.assertEqual(context.metadata['failure'], 'sidecar_submission_rejected')
        self.assertNotIn('/workspace/cases/case-1/case_result.json', writes)
        self.assertEqual(writes['/workspace/final_answer.json']['candidate'], {'answer': 'bad'})
        self.assertNotIn('read_file', captured[0]['content'])
        self.assertNotIn('run_python', captured[0]['content'])


if __name__ == '__main__':
    unittest.main()
