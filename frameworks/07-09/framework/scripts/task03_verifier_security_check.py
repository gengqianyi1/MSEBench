#!/usr/bin/env python3
"""Prove Task 03 verifier overwrites a forged native result by recomputation."""
from __future__ import annotations
import importlib.util, json, os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-03-five-ring-xtb-dipro-marcus'; SNAP=TASK/'tests/task_snapshot'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def main():
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        for n in range(1,6):
            path=work/'rounds'/f'round-{n}'/'candidate.json'; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'candidate':{'candidate_id':f'R{n}','smiles':'not-smiles'}}),encoding='utf-8')
        (work/'final_answer.json').write_text('{"recommended_candidate_id":"R1"}',encoding='utf-8')
        (work/'result.json').write_text('{"score":{"total_score":100,"eligible":true,"recommendation_valid":true},"final_recommendation":{"recommended_candidate_id":"R1"}}',encoding='utf-8')
        os.environ['CHEMBENCH_WORKSPACE']=str(work); os.environ['CHEMBENCH_TASK_ROOT']=str(SNAP)
        final=load('task03_final_security',TASK/'tests/final_grader.py'); final.main()
        result=json.loads((work/'result.json').read_text(encoding='utf-8'))
        assert result['score']['total_score']==0 and result['checks']['independently_recomputed'] is True
        grader=load('task03_legacy_security',SNAP/'private_grader/grader.py'); graded=grader.grade(SimpleNamespace(submission_dir=work,private_data_dir=SNAP/'private_data',task_dir=SNAP,run_dir=work/'run'))
        assert graded['score']==0
        print(json.dumps({'status':'PASS','forged_score':100,'recomputed_score':0,'independent':True}))
    return 0

if __name__=='__main__': raise SystemExit(main())
