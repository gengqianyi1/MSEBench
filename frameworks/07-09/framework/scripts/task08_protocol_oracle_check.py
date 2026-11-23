#!/usr/bin/env python3
"""Replay the six native Docker Oracle bundles through the new protocol grader."""
import importlib.util,json,os,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-08-xtb-failure-diagnosis-restart'; SOURCE=ROOT/'parity/task08_docker_oracle/deterministic_evidence'
def load(path): spec=importlib.util.spec_from_file_location('task08_protocol_grader_check',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def main():
 summary=json.loads((ROOT/'parity/task08_docker_oracle/summary.json').read_text()); assert len(summary['results'])==6
 token=summary['source_session_id']
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp); work=root/'workspace'; evidence=root/'evidence'; logs=root/'logs'; work.mkdir()
  for index,row in enumerate(summary['results'],1):
   case_id=row['case_id']; source=SOURCE/case_id; target=evidence/token/case_id; target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(source,target)
   artifact={'case_id':case_id,'candidate':json.loads((source/'final_answer.json').read_text()).get('model_submission',{}),'tool_events':[],'submission':{'ok':True,'evidence_id':f'{token}/{case_id}'}}
   path=work/'cases'/f'case-{index}'/'case_result.json'; path.parent.mkdir(parents=True); path.write_text(json.dumps(artifact))
  old={k:os.environ.get(k) for k in ('CHEMBENCH_WORKSPACE','CHEMBENCH_LOGS','CHEMBENCH_ORGANIZER_EVIDENCE','CHEMBENCH_ORACLE_VALIDATION')}; os.environ.update({'CHEMBENCH_WORKSPACE':str(work),'CHEMBENCH_LOGS':str(logs),'CHEMBENCH_ORGANIZER_EVIDENCE':str(evidence),'CHEMBENCH_ORACLE_VALIDATION':'1'})
  try: load(TASK/'tests/protocol_grader.py').main()
  finally:
   for k,v in old.items(): os.environ.pop(k,None) if v is None else os.environ.__setitem__(k,v)
  result=json.loads((logs/'grader-result.json').read_text()); assert result['score']==100 and not result['failure_codes'] and all(x['eligible'] for x in result['case_results']); print(json.dumps({'status':'PASS','task':8,'docker_oracle_cases':6,'protocol_score':100},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
