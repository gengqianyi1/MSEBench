"""Organizer scripted Oracle through the actual restricted Agent and Harbor lifecycle.

No model identity is claimed: actions come from packaged Oracle dialogues (07/08)
or the declared twelve-point adaptive baseline (09).
"""
import json
import os
from pathlib import Path
from agents.chembench_agent import MSEBenchAgent
from msebench_harness.protocol import resolve_case_number

ROOT = Path(__file__).resolve().parents[1]


class ReleaseOracleAgent(MSEBenchAgent):
    @staticmethod
    def name():
        return 'release-oracle'

    async def run(self, instruction, environment, context):
        task_number = int(os.environ['MSEBENCH_RELEASE_TASK'])
        count = {7: 5, 8: 6, 9: 4}[task_number]
        number = resolve_case_number(instruction, count)
        if task_number in (7, 8):
            task = next((ROOT / 'tasks').glob(f'task-{task_number:02d}-*'))
            replies = json.loads((task / 'steps' / f'case-{number}' / 'solution' / 'oracle_dialogue.json').read_text(encoding='utf-8'))
        else:
            replies = [
                {'action': 'run_scan_batch', 'angles_deg': angles, 'finalize': final,
                 'confidence': 'high' if final else None, 'rationale': rationale}
                for angles, final, rationale in [
                    ([0,60,120,180,240,300], False, 'Establish six evenly separated sectors.'),
                    ([30,150,270], False, 'Refine three unsampled sectors.'),
                    ([90,210,330], True, 'Complete the twelve-point adaptive baseline.'),
                ]
            ]
        if os.environ.get('MSEBENCH_RELEASE_WRONG') == '1':
            if task_number == 7:
                replies[-1]['candidate'].update(name='legal_wrong_methane', smiles='C',
                                               rationale='Valid SMILES deliberately violates the design constraints.')
            elif task_number == 8:
                replies[-1]['candidate']['diagnosis'].update(primary_category='UNKNOWN', primary_cause_code='UNKNOWN')
            else:
                for reply, angles in zip(replies, [[0,5,10,15,20,25],[30,35,40],[45,50,55]]):
                    reply['angles_deg'] = angles
                    reply['rationale'] = 'Schema-valid deliberately clustered sampling leaves most torsion sectors unsampled.'
        self._release_replies = iter(replies)
        self._release_actions = []
        os.environ['CHEMBENCH_USE_NATIVE_TOOL_CALLS'] = ''
        await super().run(instruction, environment, context)
        context.metadata['action_source'] = 'scripted_organizer_no_llm'
        context.metadata['fixture'] = 'legal_wrong' if os.environ.get('MSEBENCH_RELEASE_WRONG') == '1' else 'oracle'
        context.metadata['scripted_actions'] = self._release_actions

    def _call(self, *args, **kwargs):
        action = next(self._release_replies)
        self._release_actions.append(action)
        return json.dumps(action)
