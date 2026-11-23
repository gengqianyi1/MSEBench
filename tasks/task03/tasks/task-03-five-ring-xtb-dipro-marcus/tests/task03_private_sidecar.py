#!/usr/bin/env python3
"""Organizer-only five-round xTB/DIPRO evaluator for Task 03."""
from __future__ import annotations
import argparse, json, os, secrets, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import yaml

class Engine:
    def __init__(self, snapshot: Path):
        self.snapshot=snapshot.resolve(); self.lock=threading.Lock(); self.sessions={}
        sys.path.insert(0,str(self.snapshot/'input'/'original'/'src'))
        from evaluator import evaluate_candidate
        from feedback import build_feedback
        from scoring import select_best_rate
        self.evaluate_candidate=evaluate_candidate; self.build_feedback=build_feedback; self.select_best_rate=select_best_rate
        self.cfg=yaml.safe_load((self.snapshot/'input'/'original'/'tasks'/'five_ring_xtb_dipro_marcus'/'task_config.yaml').read_text(encoding='utf-8'))
    def start(self):
        token=secrets.token_urlsafe(24); self.sessions[token]=[]; return {'session_id':token,'max_rounds':5}
    def evaluate(self, token, candidate):
        with self.lock:
            values=self.sessions.get(token)
            if values is None: return {'ok':False,'error':{'code':'invalid_session'}}
            if len(values)>=5: return {'ok':False,'error':{'code':'quota_exceeded'}}
            number=len(values)+1; cid=f'R{number}'
            if not isinstance(candidate,dict) or candidate.get('candidate_id')!=cid: return {'ok':False,'error':{'code':'invalid_candidate_id'}}
            seen={x.canonical_smiles for x in values if x.canonical_smiles}
            compute_root=Path(os.environ.get('CHEMBENCH_SIDECAR_COMPUTE_ROOT','/organizer_evidence/compute'))
            try:
                ev=self.evaluate_candidate(candidate,self.cfg,compute_root/token/cid,seen,run_xtb=True)
            except Exception as exc:
                return {'ok':False,'error':{'code':'evaluator_failure','type':type(exc).__name__}}
            values.append(ev); best=self.select_best_rate(values); feedback=self.build_feedback(ev,self.cfg,best)
            summary={'evaluated_candidate_ids':[x.candidate_id for x in values],'best_candidate_id_so_far':None if best is None else best.candidate_id,'best_rate_so_far_s_inv':None if best is None else best.marcus_hole_rate_s_inv,'rounds_completed':number,'rounds_total':5}
            return {'ok':True,'feedback':feedback,'summary':summary}

def handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,status,value):
            body=json.dumps(value).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self): self.reply(200,{'ok':True}) if self.path=='/health' else self.reply(404,{'ok':False})
        def do_POST(self):
            try: payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            except Exception: return self.reply(400,{'ok':False,'error':{'code':'invalid_json'}})
            if self.path=='/session': return self.reply(200,engine.start())
            if self.path=='/evaluate': return self.reply(200,engine.evaluate(str(payload.get('session_id','')),payload.get('candidate')))
            self.reply(404,{'ok':False})
    return H

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--snapshot',type=Path,required=True); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8712); a=p.parse_args()
    ThreadingHTTPServer((a.host,a.port),handler(Engine(a.snapshot))).serve_forever()
