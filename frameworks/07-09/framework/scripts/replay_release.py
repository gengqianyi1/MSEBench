"""Replay the exact Harbor submissions in a fresh immutable verifier container."""
import json
import math
from pathlib import Path
import shutil
import subprocess
from check_release_job import check_job

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT.parent/'delivery/docker/image_manifest.json').read_text())
    rows=[]
    for task in (7,8,9):
        base=ROOT/'release_evidence'/f'task{task:02d}'
        image=next(r['immutable_reference'] for r in manifest['images'] if r['task']==task and r['role']=='sidecar')
        jobs={7:('release_v4_oracle','release_v4_wrong'),
              8:('release_v3_oracle','release_v4_wrong'),
              9:('release_v4_oracle','release_v4_wrong')}
        for fixture,wrong in [(jobs[task][0],False),(jobs[task][1],True)]:
            job=base/'jobs'/fixture
            audit=check_job(job,task,wrong)
            if audit['status']!='PASS': raise RuntimeError(audit)
            trial=Path(audit['trial_result']).parent
            # The original participant submission is never modified by replay.
            final=trial/'steps'/('case-'+str({7:5,8:6,9:4}[task]))
            output=base/('release_replay_wrong' if wrong else 'release_replay_oracle')
            if output.exists(): raise FileExistsError(output)
            work=output/'workspace';logs=output/'verifier'
            work.mkdir(parents=True);logs.mkdir()
            source=final/'artifacts/workspace'
            shutil.copytree(source/'cases',work/'cases')
            shutil.copy2(source/'protocol_state.json',work/'protocol_state.json')
            command=['docker','run','--rm','--network','none','-e','CHEMBENCH_ORACLE_VALIDATION=1',
                     '--mount',f'type=bind,source={work},target=/workspace',
                     '--mount',f'type=bind,source={logs},target=/logs/verifier',
                     '--mount',f'type=bind,source={base/"organizer_evidence"},target=/organizer_evidence,readonly',
                     image,'python','/tests/protocol_grader.py']
            with (output/'docker.log').open('w',encoding='utf-8') as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            result=json.loads((logs/'grader-result.json').read_text())
            original=json.loads((final/'verifier/grader-result.json').read_text())
            reward=float((logs/'reward.txt').read_text())
            assert result==original, 'Independent verifier result differs from Harbor verifier'
            assert math.isclose(reward,audit['reward'],abs_tol=1e-12)
            rows.append({'task':task,'fixture':'legal_wrong' if wrong else 'oracle','score':result['score'],
                         'reward':reward,'immutable_image':image,'verifier_exact_match':True,
                         'harbor_audit':audit,'replay':str(output.relative_to(ROOT))})
        assert rows[-1]['score']<rows[-2]['score'], 'Wrong submission did not score lower than Oracle'
    (ROOT/'release_evidence/release_parity.json').write_text(json.dumps({'status':'PASS','runs':rows},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','runs':len(rows)}))

if __name__=='__main__': main()
