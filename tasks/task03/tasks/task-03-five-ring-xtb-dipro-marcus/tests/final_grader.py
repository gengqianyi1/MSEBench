#!/usr/bin/env python3
"""Independently recompute all Task 03 candidates; never trust Agent-produced result.json."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import yaml

WORK=Path(os.environ.get('CHEMBENCH_WORKSPACE','/workspace')); TASK=Path(os.environ.get('CHEMBENCH_TASK_ROOT','/task'))
SRC=TASK/'input/original/src'; sys.path.insert(0,str(SRC))
from evaluator import evaluate_candidate  # noqa: E402
from scoring import final_score  # noqa: E402

def main():
    cfg=yaml.safe_load((TASK/'input/original/tasks/five_ring_xtb_dipro_marcus/task_config.yaml').read_text(encoding='utf-8'))
    evaluations=[]; seen=set(); protocol_errors=[]
    for n in range(1,6):
        path=WORK/'rounds'/f'round-{n}'/'candidate.json'
        if not path.is_file():
            protocol_errors.append(f'MISSING_ROUND_{n}_CANDIDATE')
            continue
        try:
            payload=json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            protocol_errors.append(f'INVALID_ROUND_{n}_CANDIDATE_JSON')
            continue
        if not isinstance(payload, dict):
            protocol_errors.append(f'INVALID_ROUND_{n}_CANDIDATE_SCHEMA')
            continue
        candidate=payload.get('candidate',payload) if isinstance(payload,dict) else {}
        ev=evaluate_candidate(candidate,cfg,WORK/'verifier_compute'/f'round-{n}',seen,run_xtb=True)
        if ev.canonical_smiles: seen.add(ev.canonical_smiles)
        evaluations.append(ev)
    try:
        answer=json.loads((WORK/'final_answer.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        answer={}
    recommendation=answer.get('recommended_candidate_id')
    if protocol_errors:
        score={'recommended_candidate_id':recommendation,'recommendation_valid':False,'eligible':False,'constraint_score':0,'rate_performance_score':0.0,'total_score':0.0,'recommended_candidate':None,'recommended_rate_s_inv':None,'n_target_hits':0,'n_successful_computed_candidates':len([ev for ev in evaluations if getattr(ev,'status',None)=='success']),'ranked_candidate_ids_by_rate':[]}
    else:
        score=final_score(evaluations,cfg,recommendation)
    result={'score':score,'final_recommendation':{'recommended_candidate_id':recommendation},'checks':{'independently_recomputed':True,'protocol_complete':not protocol_errors,'protocol_errors':protocol_errors,'n_evaluated':len(evaluations)},'evaluations':[ev.to_dict() for ev in evaluations]}
    (WORK/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'n_evaluated':len(evaluations),'score':score['total_score']}))
    return 0

if __name__=='__main__': raise SystemExit(main())
