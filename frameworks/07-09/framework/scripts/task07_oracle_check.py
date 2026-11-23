#!/usr/bin/env python3
"""Generate deterministic five-case baseline evidence and replay the Harbor grader."""
from __future__ import annotations
import importlib.util,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-07-qed-molecular-design'; RUNTIME=TASK/'tests/task_snapshot/private_data/runtime_package'
sys.path.insert(0,str(RUNTIME))
import yaml
from baselines.baseline_agent import run_baseline
from evaluator.evaluate import evaluate_submission
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def build(work, evidence):
 suite=yaml.safe_load((RUNTIME/'organizer_private/suite.yaml').read_text())
 expected={}
 token='oracle-fixture'
 for index,relative in enumerate(suite['cases'],1):
   cfg=RUNTIME/relative; config=yaml.safe_load(cfg.read_text()); case_id=config['case']['id']; events=[]; answer=run_baseline(cfg,process_events=events); trace=[]
   expected[case_id]=evaluate_submission(answer,cfg,process_events=events)
   for event in events:
    if event.get('event')!='tool_call': continue
    response={'ok':event.get('status')=='SUCCESS','tool':event.get('tool')}
    if event.get('status')=='SUCCESS': response['result']=event.get('result')
    else: response['error']='tool_error'
    trace.append({'tool':event.get('tool'),'arguments':event.get('arguments'),'response':response})
   root=evidence/token/case_id; root.mkdir(parents=True)
   (root/'evidence.json').write_text(json.dumps({'schema_version':'task07-organizer-evidence/1.0','session_id':token,'case_index':index,'case_id':case_id,'candidate':answer['final_candidate'],'tool_events':trace,'submission_feedback':{},'provider_status':'HARBOR_PROTOCOL','requested_model_id':'Pro/moonshotai/Kimi-K2.6','actual_model_id':'Pro/moonshotai/Kimi-K2.6','exact_model_identity':True}))
   artifact={'case_id':case_id,'candidate':answer['final_candidate'],'tool_events':trace,'submission':{'ok':True,'evidence_id':f'{token}/{case_id}'}}
   path=work/'cases'/f'case-{index}'/'case_result.json'; path.parent.mkdir(parents=True); path.write_text(json.dumps(artifact))
 return expected
def grade(work,logs,evidence,name='task07_protocol_grader'):
  oldw=os.environ.get('CHEMBENCH_WORKSPACE'); oldl=os.environ.get('CHEMBENCH_LOGS'); olde=os.environ.get('CHEMBENCH_ORGANIZER_EVIDENCE'); os.environ['CHEMBENCH_WORKSPACE']=str(work); os.environ['CHEMBENCH_LOGS']=str(logs); os.environ['CHEMBENCH_ORGANIZER_EVIDENCE']=str(evidence)
  try: grader=load(name,TASK/'tests/protocol_grader.py'); grader.main()
  finally:
   if oldw is None: os.environ.pop('CHEMBENCH_WORKSPACE',None)
   else: os.environ['CHEMBENCH_WORKSPACE']=oldw
   if oldl is None: os.environ.pop('CHEMBENCH_LOGS',None)
   else: os.environ['CHEMBENCH_LOGS']=oldl
   if olde is None: os.environ.pop('CHEMBENCH_ORGANIZER_EVIDENCE',None)
   else: os.environ['CHEMBENCH_ORGANIZER_EVIDENCE']=olde
  return json.loads((logs/'grader-result.json').read_text())
def main():
 with tempfile.TemporaryDirectory() as tmp:
  work=Path(tmp)/'workspace'; logs=Path(tmp)/'logs'; evidence=Path(tmp)/'evidence'; work.mkdir(); expected=build(work,evidence); result=grade(work,logs,evidence)
  assert not result['failure_codes'] and result['score']>0
  for case_id,direct in expected.items():
   replay=json.loads((work/'benchmark_run/protocol'/case_id/'evaluation.json').read_text())
   assert replay.get('score')==direct.get('score') and replay.get('eligible')==direct.get('eligible')
  print(json.dumps({'status':'PASS','task':7,'oracle_score':result['score'],'cases':5},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
