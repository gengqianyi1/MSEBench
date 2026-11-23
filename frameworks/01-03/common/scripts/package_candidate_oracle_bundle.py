#!/usr/bin/env python3
"""Package a scored Agent run as an organizer-only candidate Oracle bundle.

This never labels the output an Oracle.  Scientific acceptance remains an
author decision after reviewing the preserved model trajectory and grader logs.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS={
 "task01":"task-01-jarvis-2d-exfoliation-candidate",
 "task02":"task-02-oligothiophene-smd-sapt",
}

def digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def copytree(source:Path,target:Path)->None:
 for p in source.rglob('*'):
  rel=p.relative_to(source); dest=target/rel
  if p.is_dir(): dest.mkdir(parents=True,exist_ok=True)
  elif p.is_file(): dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--task',choices=TASKS,required=True); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--evidence-workspace',type=Path); a=p.parse_args()
 run=a.run_dir.resolve(); out=a.output_dir.resolve()
 if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
 task=ROOT/'tasks'/TASKS[a.task]; toml=(task/'task.toml').read_text(encoding='utf-8')
 import tomllib; meta=tomllib.loads(toml); required=meta['metadata']['required_artifacts']
 if a.evidence_workspace:
  source=a.evidence_workspace.resolve(); evidence_id=str(source)
  logs_source=run/'verifier_logs'
 else:
  case=json.loads(next((run/f'{a.task}_workspace/cases').rglob('case_result.json')).read_text(encoding='utf-8'))
  evidence_id=case['submission']['evidence_id']; source=run/f'evidence-{a.task}'/evidence_id/'workspace'
  logs_source=run/f'{a.task}_verifier_logs'
 if not source.is_dir(): raise RuntimeError(f'evidence workspace missing: {source}')
 missing=[x for x in required if not (source/x).exists()]
 if missing: raise RuntimeError(f'evidence lacks required artifacts: {missing}')
 copytree(source,out/'workspace'); copytree(logs_source,out/'verifier')
 for name in (f'{a.task}_llm_calls.jsonl',f'{a.task}_llm_call_summary.json',f'{a.task}.harbor.yaml'):
  src=run/name
  if src.is_file(): shutil.copyfile(src,out/name)
 artifacts={str(x.relative_to(source)):digest(x) for x in source.rglob('*') if x.is_file()}
 reward=json.loads((logs_source/'verifier/reward.json').read_text(encoding='utf-8'))
 details=json.loads((logs_source/'verifier/details.json').read_text(encoding='utf-8'))
 manifest={'schema_version':'chembench-candidate-oracle-bundle/1.0','status':'CANDIDATE_REQUIRES_AUTHOR_APPROVAL','task_name':meta['task']['name'],'task_version':meta['task']['version'],'source_run':str(run),'model':'deepseek-v4-flash','submission_evidence_id':evidence_id,'score':details.get('scientific_score'),'reward':reward.get('reward'),'artifacts':artifacts,'required_artifacts':required,'grader_status':details.get('status'),'failure_codes':details.get('legacy_result',{}).get('failure_codes',[])}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PACKAGED','output':str(out),'score':manifest['score'],'reward':manifest['reward'],'artifact_files':len(artifacts)},ensure_ascii=False))
 return 0
if __name__=='__main__': raise SystemExit(main())
