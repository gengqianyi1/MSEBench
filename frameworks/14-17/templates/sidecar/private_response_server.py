#!/usr/bin/env python3
"""Organizer-side HTTP boundary for Tasks 16/17 private synthetic response models.

This file contains no model coefficients. Point it at an organizer task snapshot at runtime.
"""
from __future__ import annotations
import argparse, json, os, secrets, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import yaml

PUBLIC_FIELDS = {
    16: {'status','detail','measured_cqa','desirability_D','budget_bookkeeping','feasibility_constraints_violated','best_desirability_D_so_far'},
    17: {'status','detail','measured_responses','desirability_D','window_report','budget_bookkeeping','feasibility_constraints_violated','best_desirability_D_so_far','history','round','rounds_total','measurements_remaining_after_this_round','hint','reminder'},
}


class ResponseSession:
    def __init__(self, task_number:int, snapshot:Path):
        self.task_number=task_number; self.snapshot=snapshot.resolve(); self.lock=threading.Lock(); self.sessions={}
        private=self.snapshot/'private_data'; public=self.snapshot/'input'; os.chdir(private)
        sys.path.insert(0,str(private/'src')); sys.path.insert(1,str(public/'src'))
        if task_number==16:
            config=private/'tasks/q1_microenvironment/task_config.yaml'
        elif task_number==17:
            config=private/'tasks/q1_agglomeration/task_config.yaml'
        else: raise ValueError('task_number must be 16 or 17')
        self.cfg=yaml.safe_load(config.read_text(encoding='utf-8')); self.max_rounds=int(self.cfg['iteration']['max_rounds'])
        import evaluator, feedback
        self.evaluator=evaluator; self.feedback=feedback

    def start(self)->dict[str,Any]:
        token=secrets.token_urlsafe(24); baseline=self.evaluator.evaluate_reference_start(self.cfg['objective'])
        self.sessions[token]={'evaluations':[],'history':[]}
        measured=baseline.cqa if self.task_number==16 else baseline.responses
        return {'session_id':token,'max_rounds':self.max_rounds,'baseline_measurement':measured}

    def evaluate(self,token:str,candidate:dict[str,Any])->dict[str,Any]:
        with self.lock:
            state=self.sessions.get(token)
            if state is None: return {'ok':False,'error':{'code':'invalid_session'}}
            round_idx=len(state['evaluations'])+1
            if round_idx>self.max_rounds: return {'ok':False,'error':{'code':'quota_exceeded'}}
            key='formulation' if self.task_number==16 else 'process'; raw=candidate.get(key) if isinstance(candidate,dict) else None
            ev=self.evaluator.evaluate_design(raw,self.cfg['objective']); state['evaluations'].append(ev)
            valid=[item for item in state['evaluations'] if item.valid]
            best=max(valid,key=lambda item:item.desirability) if valid else None
            if self.task_number==16:
                feedback=self.feedback.generate_feedback(ev,self.cfg['constraints'],best)
            else:
                feedback=self.feedback.generate_feedback(ev,self.cfg['constraints'],best,round_idx,self.max_rounds,list(state['history']))
                history_item={'round':round_idx,'process':ev.design,'responses':ev.responses,'desirability_D':ev.desirability}
                state['history'].append(history_item)
            if set(feedback)-PUBLIC_FIELDS[self.task_number]:
                return {'ok':False,'error':{'code':'response_contract_violation'}}
            return {'ok':True,'round':round_idx,'feedback':feedback,'measurements_remaining':self.max_rounds-round_idx}


def handler(engine:ResponseSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,status,value):
            body=json.dumps(value).encode(); self.send_response(status); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self): self.send(200,{'ok':True}) if self.path=='/health' else self.send(404,{'ok':False})
        def do_POST(self):
            try: payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            except Exception: return self.send(400,{'ok':False,'error':{'code':'invalid_json'}})
            if self.path=='/session': return self.send(200,engine.start())
            if self.path=='/evaluate': return self.send(200,engine.evaluate(str(payload.get('session_id','')),payload.get('candidate')))
            self.send(404,{'ok':False})
    return Handler


def main():
    p=argparse.ArgumentParser(); p.add_argument('--task-number',type=int,required=True); p.add_argument('--snapshot',type=Path,required=True); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8787); a=p.parse_args()
    server=ThreadingHTTPServer((a.host,a.port),handler(ResponseSession(a.task_number,a.snapshot))); server.serve_forever()
if __name__=='__main__': main()
