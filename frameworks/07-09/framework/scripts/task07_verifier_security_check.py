#!/usr/bin/env python3
"""Prove forged public tool results cannot inflate Task 07 score."""
import json,tempfile
from pathlib import Path
from scripts.task07_oracle_check import build,grade
def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp); work=root/'workspace'; work.mkdir(); build(work); honest=grade(work,root/'honest','task07_honest')['score']
  artifact=work/'cases/case-1/case_result.json'; value=json.loads(artifact.read_text())
  for event in value['tool_events']:
   result=event.get('response',{}).get('result')
   if isinstance(result,dict): result['qed']=1.0; result['feasible']=True
  artifact.write_text(json.dumps(value)); forged=grade(work,root/'forged','task07_forged')['score']
  assert forged < honest; print(json.dumps({'status':'PASS','honest_score':honest,'forged_score':forged,'inflation_blocked':True})); return 0
if __name__=='__main__': raise SystemExit(main())
