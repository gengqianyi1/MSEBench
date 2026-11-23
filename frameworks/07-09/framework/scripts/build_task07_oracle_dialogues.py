#!/usr/bin/env python3
"""Generate organizer-only scripted dialogues from the deterministic QED baseline."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-07-qed-molecular-design'; RUNTIME=TASK/'tests/task_snapshot/private_data/runtime_package'; sys.path.insert(0,str(RUNTIME))
import yaml
from baselines.baseline_agent import run_baseline
def main():
 suite=yaml.safe_load((RUNTIME/'organizer_private/suite.yaml').read_text())
 for index,relative in enumerate(suite['cases'],1):
  events=[]; answer=run_baseline(RUNTIME/relative,process_events=events)
  calls=[{'tool':e['tool'],'arguments':e['arguments']} for e in events if e.get('event')=='tool_call']
  dialogue=[{'tool_calls':calls},{'candidate':answer['final_candidate']}]
  target=TASK/'steps'/f'case-{index}'/'solution'; target.mkdir(parents=True,exist_ok=True); (target/'oracle_dialogue.json').write_text(json.dumps(dialogue,indent=2)+'\n')
  with (target/'solve.sh').open('w',encoding='utf-8',newline='\n') as handle:
   handle.write('#!/bin/bash\nset -euo pipefail\n# The Oracle Agent consumes oracle_dialogue.json through the protected sidecar.\n')
 print(json.dumps({'status':'PASS','cases':len(suite['cases'])})); return 0
if __name__=='__main__': raise SystemExit(main())
