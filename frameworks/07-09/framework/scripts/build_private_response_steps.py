#!/usr/bin/env python3
"""Generate five Harbor steps and reconstruction graders for Tasks 16 and 17."""
from pathlib import Path
import json
import sys

TASKS={
16:('task-16-tablet-microenvironment-cqa','formulation'),
17:('task-17-spherical-agglomeration','process'),
}
FINAL='''#!/usr/bin/env python3
import json,os
from pathlib import Path
WORK=Path(os.environ.get("CHEMBENCH_WORKSPACE","/workspace"))
def main():
 records=[]; rounds=[]
 for n in range(1,6):
  path=WORK/"rounds"/f"round-{{n}}"/"candidate.json"
  if not path.is_file(): continue
  candidate=json.loads(path.read_text(encoding="utf-8")).get("candidate")
  if isinstance(candidate,dict) and isinstance(candidate.get("{unit}"),dict):
   records.append({{"round":n,"submitted_candidate":candidate}}); rounds.append(candidate)
 (WORK/"trajectory.jsonl").write_text("".join(json.dumps(r)+"\\n" for r in records),encoding="utf-8")
 summary={{"reconstructed_from_harbor_steps":True,"n_rounds_recorded":len(records),"max_rounds":5}}
 (WORK/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 (WORK/"final_report.json").write_text(json.dumps({{"checks":summary}},indent=2),encoding="utf-8")
 (WORK/"final_answer.json").write_text(json.dumps({{"rounds":rounds}},indent=2),encoding="utf-8")
 print(json.dumps(summary)); return 0
if __name__=="__main__": raise SystemExit(main())
'''
TEST='''#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-{n}/candidate.json"); v=json.loads(p.read_text()) if p.is_file() else {{}}; c=v.get("candidate") if isinstance(v,dict) else None
if not isinstance(c,dict) or not isinstance(c.get("{unit}"),dict): raise SystemExit("invalid candidate")
PY
echo 1 > /logs/verifier/reward.txt
'''
def main():
 root=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
 for number,(name,unit) in TASKS.items():
  task=root/'tasks'/name
  reference=json.loads((task/'tests/task_snapshot/private_data/answers/reference_enumeration.json').read_text(encoding='utf-8'))
  design=reference['reference']['design'] if number==16 else reference['reference_design']
  for n in range(1,6):
   step=task/'steps'/f'round-{n}'; (step/'tests').mkdir(parents=True,exist_ok=True)
   schema='{"candidate":{"name":"...","rationale":"...","%s":{"...":"..."}}}'%unit
   (step/'instruction.md').write_text(f'Submit exactly one {unit} for round {n}. The restricted harness may return public diagnostic-tool results before commitment. Write `/workspace/rounds/round-{n}/candidate.json`.\n```json\n{schema}\n```\n',encoding='utf-8',newline='\n')
   (step/'tests/test.sh').write_text(TEST.format(n=n,unit=unit),encoding='utf-8',newline='\n')
   solution=step/'solution'; solution.mkdir(exist_ok=True)
   (solution/'candidate.json').write_text(json.dumps({'candidate':{'name':'private calibrated lifecycle oracle','rationale':'Exercises the complete protected measurement and independent grader path.',unit:design}},indent=2)+'\n',encoding='utf-8',newline='\n')
   (solution/'solve.sh').write_text(f'#!/bin/bash\nset -euo pipefail\nmkdir -p /workspace/rounds/round-{n}\ncp /solution/candidate.json /workspace/rounds/round-{n}/candidate.json\n',encoding='utf-8',newline='\n')
  last=task/'steps/round-5/tests/test.sh'; last.write_text('#!/bin/bash\nset -euo pipefail\npython /tests/final_grader.py\npython /tests/run_legacy_grader.py\n',encoding='utf-8',newline='\n')
  (task/'tests/final_grader.py').write_text(FINAL.format(unit=unit),encoding='utf-8',newline='\n')
 return 0
if __name__=='__main__': raise SystemExit(main())
