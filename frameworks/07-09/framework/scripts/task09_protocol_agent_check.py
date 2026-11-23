#!/usr/bin/env python3
"""Run the four-case adaptive torsion protocol through the restricted Agent."""
import asyncio,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

TASK=ROOT/'tasks/task-09-adaptive-xtb-torsion-profile'
OUTPUT=ROOT/'parity/task09_protocol_workspace'

def action(angles,finalize,confidence,rationale):
    return json.dumps({
        'action':'run_scan_batch',
        'angles_deg':angles,
        'finalize':finalize,
        'confidence':confidence,
        'rationale':rationale,
    })

async def main():
    if OUTPUT.exists(): raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    shutil.copytree(TASK/'environment/public',OUTPUT)
    env=FakeEnvironment(OUTPUT)
    agent=object.__new__(MSEBenchAgent)
    replies=[]
    for n in range(1,5):
        replies.extend([
            action([0,60,120,180,240,300],False,None,'Establish six evenly separated sectors without using an oracle.'),
            action([30,150,270],False,None,'Refine the largest unsampled gaps returned by public feedback.'),
            action([90,210,330],True,'high','Complete the adaptive campaign with the remaining high-priority grid points.'),
        ])
    scripted=iter(replies)
    agent._call=lambda *a,**k: next(scripted)
    contexts=[]
    for n in range(1,5):
        context=SimpleNamespace(metadata={}); contexts.append(context)
        await agent.run(f'Solve protected adaptive torsion case {n}.',env,context)
    assert all((OUTPUT/'cases'/f'case-{n}'/'case_result.json').is_file() and contexts[n-1].metadata['tool_calls']==3 and contexts[n-1].metadata['failure'] is None for n in range(1,5))
    assert not list(OUTPUT.rglob('organizer_private'))
    print(json.dumps({'status':'PASS','task':9,'cases':4,'adaptive_sidecar_agent':True,'workspace':str(OUTPUT)}))
    return 0

if __name__=='__main__':
    asyncio.run(main())
