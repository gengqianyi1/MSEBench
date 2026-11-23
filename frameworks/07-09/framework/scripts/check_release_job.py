"""Check actual Harbor trial and submission evidence, not just CLI exit status."""
import argparse
import json
import math
from pathlib import Path


def check_job(job: Path, task: int, wrong: bool = False) -> dict:
    errors=[]
    expected={7:5,8:6,9:4}[task]
    trials=list(job.glob('*/result.json'))
    if len(trials)!=1:
        return {'status':'FAIL','errors':['Expected exactly one completed trial result']}
    path=trials[0]
    result=json.loads(path.read_text(encoding='utf-8'))
    steps=result.get('step_results') or []
    if not result.get('finished_at') or result.get('exception_info'):
        errors.append('Trial incomplete or has exception')
    if [s['step_name'] for s in steps]!=[f'case-{n}' for n in range(1,expected+1)]:
        errors.append('Missing or out-of-order cases')
    if any(s.get('exception_info') for s in steps):
        errors.append('Agent or verifier step has exception')
    final=path.parent/'steps'/f'case-{expected}'
    grader_path=final/'verifier/grader-result.json'
    if not grader_path.is_file():
        return {'status':'FAIL','errors':errors+['Missing final grader result']}
    grader=json.loads(grader_path.read_text(encoding='utf-8'))
    if grader.get('failure_codes'):
        errors.append('Verifier rejected evidence: '+str(grader['failure_codes']))
    if not wrong and grader.get('status')!='PASS':
        errors.append('Oracle did not pass scientific eligibility')
    score=grader.get('score')
    reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward')
    if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=100:
        errors.append('Invalid score')
    elif type(reward) not in (int,float) or not math.isfinite(reward) or not math.isclose(score/100,reward,abs_tol=1e-12):
        errors.append('Score/reward mismatch')
    for n in range(1,expected+1):
        submission=final/'artifacts/workspace/cases'/f'case-{n}'/'case_result.json'
        if not submission.is_file():
            errors.append(f'Case {n} submission missing')
            continue
        value=json.loads(submission.read_text(encoding='utf-8'))
        if (value.get('submission') or {}).get('ok') is not True:
            errors.append(f'Case {n} submission was not accepted')
    return {'status':'FAIL' if errors else 'PASS','task':task,'fixture':'legal_wrong' if wrong else 'oracle',
            'score':score,'reward':reward,'trial_result':str(path),'errors':errors}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--task',type=int,choices=[7,8,9],required=True)
    parser.add_argument('--wrong',action='store_true')
    args=parser.parse_args()
    result=check_job(args.job,args.task,args.wrong)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
