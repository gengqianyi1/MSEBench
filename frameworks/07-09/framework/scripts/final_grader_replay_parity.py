#!/usr/bin/env python3
"""Offline final-grader parity: run the ACTUAL tests/final_grader.py, then legacy scorer."""
import os, sys, json, subprocess, tempfile, math
from pathlib import Path
from types import SimpleNamespace
import importlib.util

root = Path(__file__).resolve().parents[1]
t14 = root / 'tasks' / 'task-14-multiobjective-mol-design'
corpus = t14 / 'tests' / 'task_snapshot' / 'private_grader' / 'replay_corpus'
snap = t14 / 'tests' / 'task_snapshot'
priv = snap / 'private_data'

sys.path.insert(0, str(snap))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
grader = load('chembench_grader', snap / 'private_grader' / 'grader.py')

out = {}
for d in sorted(corpus.iterdir()):
    if not d.is_dir(): continue
    recs = [json.loads(l) for l in (d/'trajectory.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        for r in recs:
            cand = r.get('submitted_candidate')
            if cand and cand.get('smiles'):
                rd = ws / 'rounds' / ('round-%s' % r.get('round'))
                rd.mkdir(parents=True, exist_ok=True)
                (rd/'candidate.json').write_text(json.dumps({'candidate': cand}, ensure_ascii=False), encoding='utf-8')
        env = dict(os.environ); env['CHEMBENCH_WORKSPACE'] = str(ws)
        p = subprocess.run([sys.executable, str(t14/'tests'/'final_grader.py')],
                           capture_output=True, text=True, encoding='utf-8', env=env)
        if p.returncode != 0:
            out[d.name] = {'expected': None, 'got': None, 'match': False, 'error': p.stderr[-800:]}
            continue
        ctx = SimpleNamespace(task_dir=str(snap), submission_dir=str(ws), run_dir=str(ws), private_data_dir=str(priv))
        got = float(grader.grade(ctx).get('score', 0.0))
    exp = json.loads((d/'expected.json').read_text(encoding='utf-8'))['total_score']
    out[d.name] = {'expected': exp, 'got': got, 'match': math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9)}
print(json.dumps(out, indent=2, ensure_ascii=False))
sys.exit(0 if all(v.get('match') for v in out.values()) else 1)
