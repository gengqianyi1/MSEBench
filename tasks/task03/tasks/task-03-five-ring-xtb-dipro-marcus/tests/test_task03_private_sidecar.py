#!/usr/bin/env python3
"""Regression: Task 03 feedback comes from the private HTTP evaluator only."""
from __future__ import annotations
import json, os, subprocess, sys, time
from urllib.request import Request, urlopen

PORT=18712
environment=dict(os.environ, CHEMBENCH_SIDECAR_COMPUTE_ROOT='/workspace/sidecar_compute')
proc=subprocess.Popen([sys.executable,'/tests/task03_private_sidecar.py','--snapshot','/task','--port',str(PORT)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,env=environment)
try:
    for _ in range(50):
        try:
            with urlopen(f'http://127.0.0.1:{PORT}/health',timeout=1) as response:
                if response.status==200: break
        except OSError: time.sleep(.1)
    else: raise RuntimeError(proc.stderr.read() if proc.poll() is not None else 'sidecar startup timeout')
    def post(path,payload):
        req=Request(f'http://127.0.0.1:{PORT}{path}',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        return json.loads(urlopen(req,timeout=10).read())
    session=post('/session',{})['session_id']
    response=post('/evaluate',{'session_id':session,'candidate':{'candidate_id':'R1','parent_candidate_id':None,'proposal_mode':'direct_smiles','smiles':'not-a-smiles','structural_modification':'test','modification_reasoning':'test'}})
    assert response['ok'] is True and response['feedback']['status']=='invalid_smiles',response
    assert 'workdir' not in json.dumps(response) and 'details' not in json.dumps(response),response
    print('TASK03_PRIVATE_SIDECAR_OK')
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
