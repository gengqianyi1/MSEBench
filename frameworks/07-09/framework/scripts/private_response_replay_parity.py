#!/usr/bin/env python3
"""Prove Harbor round reconstruction preserves all Task 16/17 archived grader scores."""
from __future__ import annotations
import importlib.util,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]; TASKS={16:'task-16-tablet-microenvironment-cqa',17:'task-17-spherical-agglomeration'}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
def one(number):
 task=ROOT/'tasks'/TASKS[number]; snap=task/'tests/task_snapshot'; corpus=snap/'private_grader/replay_corpus'; errors=[]; checked=0
 sys.path.insert(0,str(snap))
 try: grader=load(f'task{number}_parity_grader',snap/'private_grader/grader.py')
 finally: sys.path.remove(str(snap))
 for case in sorted(p for p in corpus.iterdir() if p.is_dir()):
  expected=float(json.loads((case/'expected.json').read_text())['total_score'])
  with tempfile.TemporaryDirectory() as tmp:
   work=Path(tmp); records=[json.loads(line) for line in (case/'trajectory.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
   for record in records:
    n=record.get('round'); candidate=record.get('submitted_candidate')
    if n and isinstance(candidate,dict):
     path=work/'rounds'/f'round-{n}'/'candidate.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({'candidate':candidate}),encoding='utf-8')
   os.environ['CHEMBENCH_WORKSPACE']=str(work); final=load(f'task{number}_final_{checked}',task/'tests/final_grader.py'); final.main()
   actual=float(grader.grade(SimpleNamespace(task_dir=snap,private_data_dir=snap/'private_data',submission_dir=work,run_dir=work/'run'))['score']); checked+=1
   if abs(actual-expected)>1e-9: errors.append({'case':case.name,'expected':expected,'actual':actual})
 return {'task':number,'corpora_checked':checked,'errors':errors}
def main():
 results=[one(16),one(17)]; ok=not any(x['errors'] for x in results); print(json.dumps({'status':'PASS' if ok else 'FAIL','results':results},indent=2)); return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
