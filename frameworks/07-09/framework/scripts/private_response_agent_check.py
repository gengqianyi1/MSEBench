#!/usr/bin/env python3
"""Offline end-to-end checks: restricted Agent -> private sidecar -> round artifacts."""
from __future__ import annotations
import asyncio,json,os,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
from urllib.request import urlopen
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

SERVER=ROOT/'templates/sidecar/private_response_server.py'
TASKS={16:('task-16-tablet-microenvironment-cqa','formulation','reference'),17:('task-17-spherical-agglomeration','process','reference_design')}

async def one(number):
    name,unit,refkey=TASKS[number]; task=ROOT/'tasks'/name; snapshot=task/'tests/task_snapshot'; ref=json.loads((snapshot/'private_data/answers/reference_enumeration.json').read_text())
    design=ref[refkey]['design'] if number==16 else ref[refkey]
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    # Keep the sidecar in the Harbor virtual environment used by this check;
    # a system ``python`` need not have the project package dependencies.
    scientific_python=sys.executable
    proc=subprocess.Popen([scientific_python,str(SERVER),'--task-number',str(number),'--snapshot',str(snapshot),'--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    old=os.environ.get(f'CHEMBENCH_TASK{number}_SIDECAR_URL'); os.environ[f'CHEMBENCH_TASK{number}_SIDECAR_URL']=f'http://127.0.0.1:{port}'
    try:
        healthy=False
        for _ in range(100):
            try:
                with urlopen(f'http://127.0.0.1:{port}/health',timeout=.2): healthy=True; break
            except Exception: time.sleep(.05)
        if not healthy:
            raise RuntimeError('sidecar failed healthcheck: '+(proc.stderr.read() if proc.poll() is not None and proc.stderr else 'timeout'))
        with tempfile.TemporaryDirectory() as tmp:
            workspace=Path(tmp)/'workspace'; shutil.copytree(task/'environment/public',workspace); env=FakeEnvironment(workspace); agent=object.__new__(MSEBenchAgent)
            calls=0
            def reply(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==1:
                    tool='validate_formulation' if number==16 else 'check_process_legality'
                    return json.dumps({'tool_calls':[{'tool':tool,'arguments':{unit:design}}]})
                return json.dumps({'candidate':{'name':'oracle-fixture','rationale':'boundary test',unit:design}})
            agent._call=reply
            contexts=[]
            for n in range(1,6):
                context=SimpleNamespace(metadata={}); contexts.append(context)
                await agent.run(f'Submit exactly one {unit} for round {n}.',env,context)
            assert all((workspace/'rounds'/f'round-{n}'/'candidate.json').is_file() and (workspace/'rounds'/f'round-{n}'/'feedback.json').is_file() for n in range(1,6))
            state=json.loads((workspace/'protocol_state.json').read_text()); assert state['last_round']==5 and 'session_id' in state['private_response']
            assert contexts[0].metadata['tool_calls'] == 1 and contexts[0].metadata['failure'] is None
            return {'task':number,'rounds':5,'public_tool_calls':1,'sidecar_feedback':True,'agent_has_private_files':False}
    finally:
        if old is None: os.environ.pop(f'CHEMBENCH_TASK{number}_SIDECAR_URL',None)
        else: os.environ[f'CHEMBENCH_TASK{number}_SIDECAR_URL']=old
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()

async def main():
    results=[await one(16),await one(17)]; print(json.dumps({'status':'PASS','results':results},indent=2))
if __name__=='__main__': asyncio.run(main())
