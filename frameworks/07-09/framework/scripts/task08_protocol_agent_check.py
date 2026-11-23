#!/usr/bin/env python3
"""Run all six scripted Oracle dialogues through the restricted Harbor Agent."""
import asyncio,json,os,shutil,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment
TASK=ROOT/'tasks/task-08-xtb-failure-diagnosis-restart'; OUTPUT=ROOT/'parity/task08_protocol_workspace'
async def main():
 if OUTPUT.exists(): raise FileExistsError(f'refusing to overwrite {OUTPUT}')
 shutil.copytree(TASK/'environment/public',OUTPUT); env=FakeEnvironment(OUTPUT); agent=object.__new__(MSEBenchAgent); replies=[]
 for n in range(1,7): replies.extend(json.dumps(x) for x in json.loads((TASK/'steps'/f'case-{n}'/'solution/oracle_dialogue.json').read_text()))
 scripted=iter(replies); agent._call=lambda *a,**k: next(scripted); contexts=[]
 for n in range(1,7):
  context=SimpleNamespace(metadata={}); contexts.append(context); await agent.run(f'Solve protected xTB recovery case {n}.',env,context)
 assert all((OUTPUT/'cases'/f'case-{n}'/'case_result.json').is_file() and contexts[n-1].metadata['tool_calls']>=6 and contexts[n-1].metadata['failure'] is None for n in range(1,7)); assert not list(OUTPUT.rglob('organizer_private'))
 print(json.dumps({'status':'PASS','task':8,'cases':6,'native_sidecar_agent':True,'workspace':str(OUTPUT)})); return 0
if __name__=='__main__': asyncio.run(main())
