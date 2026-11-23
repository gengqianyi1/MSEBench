#!/usr/bin/env python3
"""Validate an organizer-only candidate Oracle bundle without changing score."""
from __future__ import annotations
import argparse, hashlib, json, tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def hash_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
 return h.hexdigest()

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('bundle',type=Path); a=p.parse_args(); b=a.bundle.resolve()
 m=json.loads((b/'manifest.json').read_text(encoding='utf-8')); task_name=m['task_name'].split('/')[-1]
 task=ROOT/'tasks'/task_name; spec=tomllib.loads((task/'task.toml').read_text(encoding='utf-8'))
 errors=[]; workspace=b/'workspace'
 for name in spec['metadata']['required_artifacts']:
  if not (workspace/name).exists(): errors.append(f'missing required artifact: {name}')
 for rel,expected in m.get('artifacts',{}).items():
  f=workspace/rel
  if not f.is_file(): errors.append(f'hashed artifact missing: {rel}')
  elif hash_file(f)!=expected: errors.append(f'hash mismatch: {rel}')
 reward_path=b/'verifier/verifier/reward.json'; details_path=b/'verifier/verifier/details.json'
 if not reward_path.is_file() or not details_path.is_file(): errors.append('missing private grader logs')
 else:
  reward=json.loads(reward_path.read_text(encoding='utf-8')); details=json.loads(details_path.read_text(encoding='utf-8'))
  if reward.get('reward')!=m.get('reward'): errors.append('manifest reward differs from private grader')
  if details.get('scientific_score')!=m.get('score'): errors.append('manifest score differs from private grader')
 private_markers=('test_labels_private','private_reference','oracle_bundle','satisfied_molecules')
 leaks=[]
 for f in workspace.rglob('*'):
  if f.is_file() and any(x in str(f.relative_to(workspace)).lower() for x in private_markers): leaks.append(str(f.relative_to(workspace)))
 if leaks: errors.append('private marker paths found: '+', '.join(leaks))
 print(json.dumps({'status':'PASS' if not errors else 'FAIL','bundle':str(b),'score':m.get('score'),'reward':m.get('reward'),'errors':errors},ensure_ascii=False,indent=2))
 return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
