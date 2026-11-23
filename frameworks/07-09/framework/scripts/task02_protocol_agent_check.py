#!/usr/bin/env python3
"""Protocol-only Task 02 run: prove staged run_python sidecar ownership."""
import asyncio,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

TASK=ROOT/'tasks/task-02-oligothiophene-smd-sapt'
OUTPUT=ROOT/'parity/task02_protocol_workspace'

async def main():
    if OUTPUT.exists(): raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    shutil.copytree(TASK/'environment/public',OUTPUT)
    env=FakeEnvironment(OUTPUT)
    agent=object.__new__(MSEBenchAgent)
    script="""import json\nfrom pathlib import Path\nPath('result.json').write_text(json.dumps({'task_id':'task_02_oligothiophene_smd_sapt','status':'completed'}))\n"""
    replies=iter([
        json.dumps({'tool_calls':[{'tool':'run_python','arguments':{'script':script,'timeout_seconds':30}}]}),
        json.dumps({'candidate':{'final_status':'completed','rationale':'Staged protocol sidecar fixture.'}}),
    ])
    agent._call=lambda *a,**k: next(replies)
    agent._call_tool_action=lambda *a,**k: (None,None,next(replies),{})
    context=SimpleNamespace(metadata={})
    await agent.run('Solve protected SMD-SAPT case 1.',env,context)
    assert (OUTPUT/'cases/case-1/case_result.json').is_file()
    assert context.metadata['tool_calls']==1 and context.metadata['failure'] is None
    assert not list(OUTPUT.rglob('organizer_private'))
    print(json.dumps({'status':'PASS','task':2,'staged_sidecar_agent':True,'tool_calls':1}))
    return 0

if __name__=='__main__':
    asyncio.run(main())
