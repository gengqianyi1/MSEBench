#!/usr/bin/env python3
"""Security probes for Task 08's organizer evidence replay."""
from __future__ import annotations
import importlib.util,json,os,shutil,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/task-08-xtb-failure-diagnosis-restart'
SOURCE=ROOT/'parity/task08_docker_oracle/deterministic_evidence'
SUMMARY=ROOT/'parity/task08_docker_oracle/summary.json'

def load_grader():
    path=TASK/'tests/protocol_grader.py'
    spec=importlib.util.spec_from_file_location('task08_security_grader',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def case_id(index):
    return json.loads(SUMMARY.read_text())['results'][index-1]['case_id']

def setup(root, evidence_prefix=None):
    if evidence_prefix is None:
        evidence_prefix=json.loads(SUMMARY.read_text())['source_session_id']
    work=root/'workspace'; evidence=root/'evidence'; logs=root/'logs'; work.mkdir(parents=True); evidence.mkdir(parents=True); logs.mkdir(parents=True)
    for index in range(1,7):
        cid=case_id(index)
        target=evidence/evidence_prefix/cid
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(SOURCE/cid,target)
        artifact={'case_id':cid,'candidate':json.loads((target/'final_answer.json').read_text()).get('model_submission',{}),'tool_events':[],'submission':{'ok':True,'evidence_id':f'{evidence_prefix}/{cid}'}}
        path=work/'cases'/f'case-{index}'/'case_result.json'; path.parent.mkdir(parents=True); path.write_text(json.dumps(artifact))
    return work,evidence,logs

def grade(work,evidence,logs):
    old={k:os.environ.get(k) for k in ('CHEMBENCH_WORKSPACE','CHEMBENCH_LOGS','CHEMBENCH_ORGANIZER_EVIDENCE','CHEMBENCH_ORACLE_VALIDATION')}
    os.environ.update({'CHEMBENCH_WORKSPACE':str(work),'CHEMBENCH_LOGS':str(logs),'CHEMBENCH_ORGANIZER_EVIDENCE':str(evidence),'CHEMBENCH_ORACLE_VALIDATION':'1'})
    try:
        load_grader().main()
    finally:
        for k,v in old.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
    return json.loads((logs/'grader-result.json').read_text())

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        honest_token=json.loads(SUMMARY.read_text())['source_session_id']
        # Honest oracle baseline.
        work,evidence,logs=setup(root)
        honest=grade(work,evidence,logs)
        assert honest['score']==100 and not honest['failure_codes'], honest
        # Forged evidence ID and traversal.
        for evil in ('../oracle/aspirin_gfn2_opt_hess_restart_v1','oracle/../aspirin_gfn2_opt_hess_restart_v1','oracle/../../etc'):
            work2,evidence2,logs2=setup(root/f'scenario_{abs(hash(evil))}')
            artifact=work2/'cases/case-1/case_result.json'; value=json.loads(artifact.read_text()); value['submission']['evidence_id']=evil; artifact.write_text(json.dumps(value))
            result=grade(work2,evidence2,logs2)
            assert result['score']<100 or result['failure_codes'], (evil,result)
        # Tampered organizer evidence must not retain the honest score.
        work3,evidence3,logs3=setup(root/'tampered')
        cid=case_id(1); tampered=evidence3/honest_token/cid/'final_answer.json'
        tampered.write_text(json.dumps({'model_submission':{},'trusted_runtime':{}}))
        result=grade(work3,evidence3,logs3)
        assert result['score']<100 or result['failure_codes'], result
        # Workspace self-reported changes are ignored: modify candidate but not organizer evidence.
        work4,evidence4,logs4=setup(root/'self_report')
        artifact=work4/'cases/case-1/case_result.json'; value=json.loads(artifact.read_text()); value['candidate']['final_status']='SUCCESS_FAKE'; value['submission']['ok']=True; artifact.write_text(json.dumps(value))
        result=grade(work4,evidence4,logs4)
        assert result['score']==100 and not result['failure_codes'], result
    print(json.dumps({'status':'PASS','task':8,'evidence_forgery_blocked':True,'path_traversal_blocked':True,'evidence_tamper_blocked':True,'workspace_self_report_ignored':True}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
