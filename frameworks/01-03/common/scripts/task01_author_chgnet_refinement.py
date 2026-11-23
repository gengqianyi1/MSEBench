#!/usr/bin/env python3
"""Author-mode deterministic Task 01 refinement using the declared CHGNet tool.

It starts the unmodified organizer sidecar, seeds it with a previously accepted
public-artifact workspace, invokes the one allowed CHGNet operation exactly
once, submits the updated workspace, then invokes the unchanged private grader.
No private labels are mounted in the agent/seed workspace.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/task-01-jarvis-2d-exfoliation-candidate'
IMAGE='chembench/std-task01-verifier:2.1.0'; PORT=8715; NAME='chembench-task01-author-chgnet-refinement'

def posix(path:Path)->str: return path.resolve().as_posix()
def post(url:str, body:dict)->dict:
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=1800) as r:return json.loads(r.read())
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--seed-workspace',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 seed=a.seed_workspace.resolve(); out=a.output_dir.resolve(); evidence=out/'evidence'
 if out.exists():raise FileExistsError(out)
 selected=[x['candidate_id'] for x in json.loads((seed/'result.json').read_text(encoding='utf-8'))['selected_generated_candidates']]
 if len(selected)!=5 or len(set(selected))!=5:raise RuntimeError('result.json does not declare exactly five distinct selected candidates')
 out.mkdir(parents=True); subprocess.run(['docker','rm','-f',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 command=['docker','run','-d','--user','0:0','--name',NAME,'-p',f'127.0.0.1:{PORT}:{PORT}',
 '-v',f'{posix(ROOT/"templates/sidecar/hidden_suite_server.py")}:/sidecar.py:ro',
 '-v',f'{posix(TASK/"tests/private_suite_adapter.py")}:/tests/private_suite_adapter.py:ro',
 '-v',f'{posix(TASK/"tests/task_snapshot")}:/task:ro','-v',f'{posix(evidence)}:/organizer_evidence',
 '-v',f'{posix(seed)}:/seed:ro','-e','CHEMBENCH_ORACLE_SEED_WORKSPACE=/seed',IMAGE,
 'micromamba','run','-n','chembench','python','/sidecar.py','--adapter','/tests/private_suite_adapter.py','--snapshot','/task','--evidence-dir','/organizer_evidence','--host','0.0.0.0','--port',str(PORT)]
 try:
  subprocess.run(command,check=True)
  for _ in range(120):
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2):break
   except Exception:time.sleep(1)
  else:raise RuntimeError('sidecar did not become healthy')
  session=post(f'http://127.0.0.1:{PORT}/session',{})['session_id']
  tool=post(f'http://127.0.0.1:{PORT}/tool',{'session_id':session,'tool':'relax_selected_candidates_chgnet','arguments':{'candidate_ids':selected}})
  if not tool.get('ok'):raise RuntimeError(f'CHGNet tool failed: {tool}')
  submitted=post(f'http://127.0.0.1:{PORT}/submit',{'session_id':session,'candidate':{'final_status':'completed','rationale':'Author-mode deterministic CHGNet refinement of previously accepted candidate library.'}})
  if not submitted.get('ok'):raise RuntimeError(f'sidecar rejected refined submission: {submitted}')
  (out/'sidecar_tool_response.json').write_text(json.dumps(tool,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'submission.json').write_text(json.dumps(submitted,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  workspace=evidence/submitted['evidence_id']/'workspace';logs=out/'verifier_logs';run=out/'verifier_run'
  subprocess.run(['docker','run','--rm','-v',f'{posix(workspace)}:/workspace','-v',f'{posix(logs)}:/logs','-v',f'{posix(run)}:/run',IMAGE,'micromamba','run','-n','chembench','python','/tests/run_legacy_grader.py'],check=True)
  print(json.dumps({'status':'PASS','evidence_workspace':str(workspace),'reward':json.loads((logs/'verifier/reward.json').read_text(encoding='utf-8'))},ensure_ascii=False))
  return 0
 finally:subprocess.run(['docker','rm','-f',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if __name__=='__main__':raise SystemExit(main())
