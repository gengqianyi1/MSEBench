#!/usr/bin/env python3
"""Task 07 restricted Agent -> hidden-suite sidecar smoke check."""
from __future__ import annotations
import asyncio,json,os,shutil,socket,subprocess,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment
TASK=ROOT/'tasks/task-07-qed-molecular-design'
async def main():
 with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
 # The sidecar imports task-local adapter modules.  Launch it with the same
 # interpreter that runs this regression, rather than a host ``python`` that
 # may not have Harbor or the project root available.
 scientific=sys.executable
 proc=subprocess.Popen([scientific,str(ROOT/'templates/sidecar/hidden_suite_server.py'),'--adapter',str(TASK/'tests/private_suite_adapter.py'),'--snapshot',str(TASK/'tests/task_snapshot'),'--port',str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
 old=os.environ.get('CHEMBENCH_TASK07_SIDECAR_URL'); os.environ['CHEMBENCH_TASK07_SIDECAR_URL']=f'http://127.0.0.1:{port}'
 try:
  for _ in range(100):
   try:
    with urlopen(f'http://127.0.0.1:{port}/health',timeout=.2): break
   except Exception: time.sleep(.05)
  else: raise RuntimeError(proc.stderr.read() if proc.poll() is not None else 'sidecar timeout')
  with tempfile.TemporaryDirectory() as tmp:
   work=Path(tmp)/'workspace'; shutil.copytree(TASK/'environment/public',work); env=FakeEnvironment(work); agent=object.__new__(MSEBenchAgent)
   replies=[]
   for n in range(1,6): replies.extend(json.dumps(x) for x in json.loads((TASK/'steps'/f'case-{n}'/'solution/oracle_dialogue.json').read_text()))
   scripted=iter(replies); agent._call=lambda *a,**k: next(scripted); contexts=[]
   for n in range(1,6):
    context=SimpleNamespace(metadata={}); contexts.append(context); await agent.run(f'Solve protected QED suite case {n}.',env,context)
   assert all((work/'cases'/f'case-{n}'/'case_result.json').is_file() and contexts[n-1].metadata['tool_calls']>=3 for n in range(1,6))
   assert not list(work.rglob('qed_hidden_*.yaml')); print(json.dumps({'status':'PASS','task':7,'cases':5,'agent_tool_submit':True,'minimum_tool_calls_each':True,'private_case_files_in_workspace':False}))
 finally:
  if old is None: os.environ.pop('CHEMBENCH_TASK07_SIDECAR_URL',None)
  else: os.environ['CHEMBENCH_TASK07_SIDECAR_URL']=old
  proc.terminate()
  try: proc.wait(timeout=5)
  except subprocess.TimeoutExpired: proc.kill()
if __name__=='__main__': asyncio.run(main())
