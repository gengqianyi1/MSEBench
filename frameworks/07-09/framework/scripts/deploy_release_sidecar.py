"""Render/start a digest-pinned sidecar from the organizer's offline image bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[7,8,9], required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT.parent/'delivery/docker/image_manifest.json')
    parser.add_argument('--port', type=int)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    row = next(r for r in manifest['images'] if r['task']==args.task and r['role']=='sidecar')
    output = ROOT/'release_evidence'/f'task{args.task:02d}'
    evidence = output/'organizer_evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    config = {'services':{'sidecar':{
        'image':row['immutable_reference'], 'pull_policy':'never',
        'ports':[f'127.0.0.1:{args.port or 18700+args.task}:8717'],
        'environment':{'CHEMBENCH_MODEL_ID':'organizer/scripted-oracle-v1',
                       'CHEMBENCH_ACTUAL_MODEL_ID':'organizer/scripted-oracle-v1'},
        'volumes':[{'type':'bind','source':evidence.resolve().as_posix(),'target':'/organizer_evidence'}],
        'healthcheck':{'test':['CMD','python','-c',"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8717/health',timeout=3)"],
                       'interval':'5s','timeout':'5s','retries':12},
    }}}
    path=output/'sidecar.compose.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    if args.start:
        subprocess.run(['docker','image','inspect',row['immutable_reference']],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['docker','compose','-p',f'matbench-release-{args.task:02d}',
                        '-f',str(path),'up','-d','--wait'],check=True)
    print(path)

if __name__=='__main__': main()
