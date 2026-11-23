#!/usr/bin/env python3
"""Generate Task 03's five design rounds and final recommendation step."""
from pathlib import Path
import json
import sys

CANDIDATE_TEST = '''#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-{n}/candidate.json")
v=json.loads(p.read_text()) if p.is_file() else {{}}
c=v.get("candidate") if isinstance(v,dict) else None
required={required!r}
if not isinstance(c,dict) or any(k not in c for k in required) or c.get("candidate_id")!="R{n}":
    raise SystemExit("invalid round candidate")
PY
echo 1 > /logs/verifier/reward.txt
'''


def main() -> int:
    root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
    task=root/'tasks/task-03-five-ring-xtb-dipro-marcus'
    references=json.loads((task/'tests/task_snapshot/private_data/reference_examples/satisfied_molecules.json').read_text(encoding='utf-8'))['molecules']
    references=sorted(references,key=lambda value:float(value['marcus_hole_rate_s_inv']),reverse=True)[:5]
    required=['candidate_id','parent_candidate_id','proposal_mode','smiles','structural_modification','modification_reasoning']
    for n in range(1,6):
        step=task/'steps'/f'round-{n}'; (step/'tests').mkdir(parents=True,exist_ok=True)
        if n==1:
            intro='Read `/workspace/input/original/tasks/five_ring_xtb_dipro_marcus/question.md`. Submit candidate R1.\n'
        else:
            intro=f'Read `/workspace/rounds/round-{n-1}/feedback.json` and submit candidate R{n}.\n'
            work=step/'workdir'; work.mkdir(exist_ok=True)
            (work/'setup.sh').write_text(f'''#!/bin/bash
set -euo pipefail
python /workspace/chembench_protocol/evaluate_round.py --round {n-1} --candidate /workspace/rounds/round-{n-1}/candidate.json --feedback /workspace/rounds/round-{n-1}/feedback.json --state /workspace/rounds/public_protocol_state.json
''',encoding='utf-8',newline='\n')
        schema='''{"candidate":{"candidate_id":"R%d","parent_candidate_id":null,"proposal_mode":"direct_smiles","smiles":"...","structural_modification":"...","modification_reasoning":"..."}}''' % n
        (step/'instruction.md').write_text(intro+'Return exactly this JSON schema and write it to the declared round path:\n```json\n'+schema+'\n```\n',encoding='utf-8',newline='\n')
        (step/'tests/test.sh').write_text(CANDIDATE_TEST.format(n=n,required=required),encoding='utf-8',newline='\n')
        solution=step/'solution'; solution.mkdir(exist_ok=True)
        candidate={'candidate_id':f'R{n}','parent_candidate_id':None if n==1 else 'R1','proposal_mode':'direct_smiles','smiles':references[n-1]['smiles'],'structural_modification':'calibrated private lifecycle fixture','modification_reasoning':'Exercises the complete independently recomputed xTB-DIPRO path.'}
        (solution/'candidate.json').write_text(json.dumps({'candidate':candidate},indent=2)+'\n',encoding='utf-8',newline='\n')
        (solution/'solve.sh').write_text(f'#!/bin/bash\nset -euo pipefail\nmkdir -p /workspace/rounds/round-{n}\ncp /solution/candidate.json /workspace/rounds/round-{n}/candidate.json\n',encoding='utf-8',newline='\n')
    step=task/'steps'/'recommendation'; (step/'tests').mkdir(parents=True,exist_ok=True); (step/'workdir').mkdir(exist_ok=True)
    (step/'workdir/setup.sh').write_text('''#!/bin/bash
set -euo pipefail
python /workspace/chembench_protocol/evaluate_round.py --round 5 --candidate /workspace/rounds/round-5/candidate.json --feedback /workspace/rounds/round-5/feedback.json --state /workspace/rounds/public_protocol_state.json
''',encoding='utf-8',newline='\n')
    (step/'instruction.md').write_text('Read `/workspace/rounds/public_protocol_state.json`. Return exactly `{"recommended_candidate_id":"R1"}` with one evaluated ID R1-R5 and write `/workspace/final_answer.json`.\n',encoding='utf-8',newline='\n')
    (step/'tests/test.sh').write_text('#!/bin/bash\nset -euo pipefail\npython /tests/final_grader.py\npython /tests/run_legacy_grader.py\n',encoding='utf-8',newline='\n')
    best='R1'
    (step/'solution').mkdir(exist_ok=True)
    (step/'solution/final_answer.json').write_text(json.dumps({'recommended_candidate_id':best},indent=2)+'\n',encoding='utf-8',newline='\n')
    (step/'solution/solve.sh').write_text('#!/bin/bash\nset -euo pipefail\ncp /solution/final_answer.json /workspace/final_answer.json\n',encoding='utf-8',newline='\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
