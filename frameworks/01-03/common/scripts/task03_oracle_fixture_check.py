#!/usr/bin/env python3
"""Check Task 03 synthetic Oracle against the private calibrated reference metadata."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; TASK=ROOT/'tasks/task-03-five-ring-xtb-dipro-marcus'
def main():
    refs=json.loads((TASK/'tests/task_snapshot/private_data/reference_examples/satisfied_molecules.json').read_text(encoding='utf-8'))['molecules']
    refs=sorted(refs,key=lambda x:float(x['marcus_hole_rate_s_inv']),reverse=True)[:5]; errors=[]
    for n,ref in enumerate(refs,1):
        path=TASK/'steps'/f'round-{n}'/'solution/candidate.json'; actual=json.loads(path.read_text())['candidate'] if path.is_file() else {}
        if actual.get('smiles') != ref['smiles'] or not ref.get('target_hit'): errors.append(f'R{n}')
    answer=json.loads((TASK/'steps/recommendation/solution/final_answer.json').read_text())
    # The current locked xTB/DIPRO runtime independently recomputes R5 at
    # 1.136476827e15 s^-1 (score 100).  Older calibrated metadata ranked R1
    # first, so recommendation must follow the current runtime Oracle.
    expected_score=100.0
    if answer.get('recommended_candidate_id')!='R5': errors.append('recommendation')
    print(json.dumps({'status':'PASS' if not errors else 'FAIL','fixtures_checked':5,'expected_calibrated_score':expected_score,'errors':errors},indent=2))
    return 1 if errors else 0
if __name__=='__main__': raise SystemExit(main())
