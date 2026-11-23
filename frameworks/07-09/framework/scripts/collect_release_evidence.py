"""Collect only completed, exception-free Harbor results; do not infer readiness."""
import json
import math
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SELECTIONS={
    7:{'oracle':'release_v4_oracle','legal_wrong':'release_v4_wrong'},
    8:{'oracle':'release_v3_oracle','legal_wrong':'release_v4_wrong'},
    9:{'oracle':'release_v4_oracle','legal_wrong':'release_v4_wrong'},
}

def main():
    records=[]
    for task, fixtures in SELECTIONS.items():
        root=ROOT/'release_evidence'/f'task{task:02d}'
        for fixture,job in fixtures.items():
            trials=list((root/'jobs'/job).glob('*/result.json'))
            assert len(trials)==1
            result=json.loads(trials[0].read_text(encoding='utf-8'))
            steps=result['step_results']
            assert result['finished_at'] and not result['exception_info']
            assert all(not row['exception_info'] for row in steps)
            final=trials[0].parent/'steps'/steps[-1]['step_name']
            grader=json.loads((final/'verifier/grader-result.json').read_text(encoding='utf-8'))
            reward=result['verifier_result']['rewards']['reward']
            assert not grader['failure_codes']
            assert math.isclose(reward,grader['score']/100,rel_tol=0,abs_tol=1e-12)
            destination=root/f'{fixture}_submission'
            destination.mkdir(exist_ok=True)
            workspace=final/'artifacts/workspace'
            shutil.copytree(workspace/'cases',destination/'cases',dirs_exist_ok=True)
            shutil.copy2(workspace/'protocol_state.json',destination/'protocol_state.json')
            shutil.copytree(final/'verifier',root/f'{fixture}_verification',dirs_exist_ok=True)
            refs=[]
            for path in sorted((destination/'cases').glob('*/case_result.json')):
                value=json.loads(path.read_text(encoding='utf-8'))
                assert value['submission']['ok'] is True
                refs.append(value['submission']['evidence_id'])
            record={'task':task,'fixture':fixture,'score':grader['score'],'reward':reward,
                    'status':grader['status'],'harbor_trial':str(trials[0].relative_to(ROOT)),
                    'submission':str(destination.relative_to(ROOT)),
                    'case_count':len(steps),'all_steps_exception_free':True,
                    'all_submissions_accepted':True,'score_to_reward_parity':True,
                    'evidence_ids':refs,'agent_kind':'scripted_organizer_no_llm'}
            records.append(record)
    (ROOT/'release_evidence/verified_results.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k in ('task','fixture','score','reward','status')} for r in records]))

if __name__=='__main__': main()
