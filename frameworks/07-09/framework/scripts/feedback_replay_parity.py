#!/usr/bin/env python3
"""Offline feedback parity: new evaluate_round science vs legacy feedback_returned."""
import sys, json, math
from pathlib import Path
root = Path(__file__).resolve().parents[1]
t14 = root / 'tasks' / 'task-14-multiobjective-mol-design'
src = t14 / 'environment' / 'public' / 'input' / 'src'
cfg = t14 / 'environment' / 'public' / 'input' / 'tasks' / 'q1_multiobjective' / 'task_config.yaml'
corpus = t14 / 'tests' / 'task_snapshot' / 'private_grader' / 'replay_corpus'
sys.path.insert(0, str(src))
from utils import load_yaml
from evaluator import evaluate_smiles
from feedback import generate_feedback

def close(a, b, tol=1e-6):
    if a is None or b is None: return a == b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
    if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(close(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict): return set(a)==set(b) and all(close(a[k],b[k],tol) for k in a)
    return a==b

c = load_yaml(cfg); start = c['start_smiles']; cons = c['constraints']; start_ev = evaluate_smiles(start)
out = {}
for d in sorted(corpus.iterdir()):
    if not d.is_dir(): continue
    recs = [json.loads(l) for l in (d/'trajectory.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    best = None; bad = []
    for r in recs:
        smi = (r.get('submitted_candidate') or {}).get('smiles','').strip()
        old = r.get('feedback_returned')
        if not smi or old is None: continue
        ev = evaluate_smiles(smi, start_smiles=start)
        if ev.valid and (best is None or ev.qed > best.qed): best = ev
        new = generate_feedback(ev, start_ev, cons, best_so_far=best)
        for k in ('computed_properties','reference_start_qed','best_qed_so_far'):
            if not close(new.get(k), old.get(k)): bad.append((r.get('round'), k))
    out[d.name] = {'rounds': len([r for r in recs if (r.get('submitted_candidate') or {}).get('smiles')]), 'mismatches': bad}
print(json.dumps(out, indent=2, ensure_ascii=False))
sys.exit(1 if any(v['mismatches'] for v in out.values()) else 0)
