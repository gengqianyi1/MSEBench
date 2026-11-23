#!/usr/bin/env python3
"""Independently grade the Task 16/17 five-step private reference Oracles."""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True
import importlib.util,json,os,shutil,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]; TASKS={16:'task-16-tablet-microenvironment-cqa',17:'task-17-spherical-agglomeration'}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
def one(number):
 task=ROOT/'tasks'/TASKS[number]; snap=task/'tests/task_snapshot'
 with tempfile.TemporaryDirectory() as tmp:
  work=Path(tmp)
  for n in range(1,6):
   target=work/'rounds'/f'round-{n}'/'candidate.json'; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(task/'steps'/f'round-{n}'/'solution/candidate.json',target)
  os.environ['CHEMBENCH_WORKSPACE']=str(work); final=load(f'oracle_final_{number}',task/'tests/final_grader.py'); final.main()
  sys.path.insert(0,str(snap))
  try: grader=load(f'oracle_grader_{number}',snap/'private_grader/grader.py')
  finally: sys.path.remove(str(snap))
  result=grader.grade(SimpleNamespace(task_dir=snap,private_data_dir=snap/'private_data',submission_dir=work,run_dir=work/'run'))
  return {'task':number,'score':result['score'],'failure_codes':result.get('failure_codes',[])}
def main():
 results=[one(16),one(17)]; ok=all(float(r['score'])==100 for r in results); print(json.dumps({'status':'PASS' if ok else 'FAIL','results':results},indent=2)); return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
