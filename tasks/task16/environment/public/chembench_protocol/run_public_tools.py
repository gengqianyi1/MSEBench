#!/usr/bin/env python3
import argparse,base64,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'input/src'))
from tools import run_tool_calls  # noqa:E402
p=argparse.ArgumentParser(); p.add_argument('--payload',required=True); a=p.parse_args(); value=json.loads(base64.b64decode(a.payload)); used=int(value.get('used',0))
cfg=yaml.safe_load((ROOT/'input/tasks/q1_microenvironment/task_config.yaml').read_text(encoding='utf-8')); maximum=int(cfg.get('tools',{}).get('call_budget_per_round',4)); left=max(0,maximum-used)
batch=run_tool_calls(value.get('calls'),cfg['constraints'],left); now=maximum-int(batch.get('budget_left',left)); print(json.dumps({'results':batch.get('results',[]),'calls_used':now}))
