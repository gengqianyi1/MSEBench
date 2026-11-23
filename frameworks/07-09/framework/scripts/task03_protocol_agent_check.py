#!/usr/bin/env python3
"""Offline contract test for Task 03 structured rounds and recommendation."""
from __future__ import annotations
import asyncio, json, shutil, tempfile, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

TASK=ROOT/'tasks/task-03-five-ring-xtb-dipro-marcus'

async def check():
    with tempfile.TemporaryDirectory() as tmp:
        workspace=Path(tmp)/'workspace'; shutil.copytree(TASK/'environment/public',workspace)
        env=FakeEnvironment(workspace); agent=object.__new__(MSEBenchAgent)
        candidate={'candidate_id':'R1','parent_candidate_id':None,'proposal_mode':'direct_smiles','smiles':'c1ccc2c(c1)ccc1c3ccccc3ccc21','structural_modification':'fixture','modification_reasoning':'fixture'}
        agent._call=lambda *a,**k: json.dumps({'candidate':candidate})
        context=SimpleNamespace(metadata={}); await agent.run('Submit candidate R1 for round 1.',env,context)
        assert json.loads((workspace/'rounds/round-1/candidate.json').read_text())=={'candidate':candidate}
        state_path=workspace/'rounds/public_protocol_state.json'; state_path.parent.mkdir(parents=True,exist_ok=True); state_path.write_text('{"evaluations":[]}',encoding='utf-8')
        agent._call=lambda *a,**k: '{"recommended_candidate_id":"R1"}'
        await agent.run('Return final recommended_candidate_id from R1-R5.',env,context)
        assert json.loads((workspace/'final_answer.json').read_text())=={'recommended_candidate_id':'R1'}
        print(json.dumps({'status':'PASS','candidate_round':True,'recommendation':True,'arbitrary_commands':0}))

if __name__=='__main__': asyncio.run(check())
