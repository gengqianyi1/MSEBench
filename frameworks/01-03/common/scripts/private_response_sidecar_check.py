#!/usr/bin/env python3
"""End-to-end quota and leakage checks for Task 16/17 private response sidecars."""
from __future__ import annotations
import json, socket, subprocess, sys, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]; SERVER=ROOT/'templates/sidecar/private_response_server.py'
TASKS={16:'task-16-tablet-microenvironment-cqa',17:'task-17-spherical-agglomeration'}

def post(url,value):
    req=Request(url,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=30) as response: return json.loads(response.read())

def forbidden_keys(value):
    hits=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if any(word in str(key).lower() for word in ('reference','achievement','score','oracle','coefficient','gradient','sensitivity')): hits.append(str(key))
            hits.extend(forbidden_keys(item))
    elif isinstance(value,list):
        for item in value: hits.extend(forbidden_keys(item))
    return hits

def one(number):
    task=ROOT/'tasks'/TASKS[number]; snapshot=task/'tests/task_snapshot'; ref=json.loads((snapshot/'private_data/answers/reference_enumeration.json').read_text(encoding='utf-8'))
    design=ref['reference']['design'] if number==16 else ref['reference_design']; key='formulation' if number==16 else 'process'
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    proc=subprocess.Popen([sys.executable,str(SERVER),'--task-number',str(number),'--snapshot',str(snapshot),'--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        base=f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                with urlopen(base+'/health',timeout=.2): break
            except Exception: time.sleep(.05)
        else: raise RuntimeError('sidecar did not become healthy')
        session=post(base+'/session',{}); token=session['session_id']; responses=[]
        for _ in range(5): responses.append(post(base+'/evaluate',{'session_id':token,'candidate':{key:design}}))
        quota=post(base+'/evaluate',{'session_id':token,'candidate':{key:design}})
        assert all(item.get('ok') for item in responses)
        assert quota.get('error',{}).get('code')=='quota_exceeded'
        hits=forbidden_keys({'session':session,'responses':responses}); assert not hits,hits
        return {'task':number,'rounds':5,'quota_enforced':True,'forbidden_key_hits':hits}
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
        if proc.returncode not in (0,-15,1):
            stderr=proc.stderr.read() if proc.stderr else ''; raise RuntimeError(stderr)

def main():
    results=[one(16),one(17)]; print(json.dumps({'status':'PASS','results':results},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
