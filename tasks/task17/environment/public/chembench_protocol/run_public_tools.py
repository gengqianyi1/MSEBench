#!/usr/bin/env python3
import argparse,base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'input/src'))
from tools import MAX_TOOL_CALLS_PER_TURN,run_tool_calls  # noqa:E402
p=argparse.ArgumentParser(); p.add_argument('--payload',required=True); a=p.parse_args(); value=json.loads(base64.b64decode(a.payload)); used=int(value.get('used',0))
batch=run_tool_calls(value.get('calls'),quota_per_turn=MAX_TOOL_CALLS_PER_TURN); now=used+int(batch.get('n_executed',0))+int(batch.get('n_error',0)); print(json.dumps({'results':batch.get('results',[]),'calls_used':now}))
