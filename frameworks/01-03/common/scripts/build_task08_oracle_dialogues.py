#!/usr/bin/env python3
"""Extract organizer-only Harbor dialogues from the six native Docker Oracle runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-08-xtb-failure-diagnosis-restart'; SOURCE=ROOT/'parity/task08_docker_oracle/deterministic_evidence'
def main():
 summary=json.loads((ROOT/'parity/task08_docker_oracle/summary.json').read_text())
 for index,row in enumerate(summary['results'],1):
  source=SOURCE/row['case_id']; calls=[]; id_map={}
  for line in (source/'trajectory.jsonl').read_text().splitlines():
   event=json.loads(line)
   if event.get('event')=='tool_call':
    payload=event.get('payload',{}); name=payload.get('name')
    if name!='submit_final':
     calls.append({'tool':name,'arguments':payload.get('arguments',{})}); id_map[payload.get('call_id')]=f'harbor-{len(calls):03d}'
  candidate=json.loads((source/'final_answer.json').read_text())['model_submission']; dialogue=[{'tool_calls':calls},{'candidate':candidate}]
  for evidence in candidate.get('diagnosis',{}).get('evidence',[]):
   if evidence.get('tool_call_id') in id_map: evidence['tool_call_id']=id_map[evidence['tool_call_id']]
  target=TASK/'steps'/f'case-{index}'/'solution'; target.mkdir(parents=True,exist_ok=True); (target/'oracle_dialogue.json').write_text(json.dumps(dialogue,indent=2)+'\n')
  with (target/'solve.sh').open('w',encoding='utf-8',newline='\n') as handle: handle.write('#!/bin/bash\nset -euo pipefail\n# Oracle dialogue is executed through the protected native-xTB sidecar.\n')
 print(json.dumps({'status':'PASS','cases':len(summary['results'])})); return 0
if __name__=='__main__': raise SystemExit(main())
