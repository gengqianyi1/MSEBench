#!/usr/bin/env python3
"""Launch every task from one immutable TextQA generation as a Harbor job.

Harbor 0.21 accepts one local task path per ``harbor run`` invocation; repeated
``--path`` flags silently retain only the final value.  This launcher therefore
creates one isolated Harbor job per manifest entry and bounds process-level
parallelism itself.
"""
from __future__ import annotations
import argparse, json, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--agent',required=True)
    parser.add_argument('--jobs-dir',required=True,type=Path)
    parser.add_argument('--n-concurrent',type=int,default=4)
    parser.add_argument('--install-only',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--harbor',default='harbor')
    args=parser.parse_args()
    root=args.dataset.resolve(); manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    commands=[]
    for item in manifest['tasks']:
        command=[args.harbor,'run','--agent',args.agent,'--jobs-dir',str(args.jobs_dir/item['task_name']),'--n-concurrent','1','--yes','--path',str(root/item['task_directory'])]
        if args.install_only: command.append('--install-only')
        commands.append((item['task_name'],command))
    if args.dry_run:
        for _, command in commands: print(' '.join(command))
        return 0
    args.jobs_dir.mkdir(parents=True,exist_ok=True)
    def execute(row):
        name, command=row
        return name, subprocess.run(command,check=False).returncode
    failures=[]
    with ThreadPoolExecutor(max_workers=args.n_concurrent) as pool:
        futures=[pool.submit(execute,row) for row in commands]
        for future in as_completed(futures):
            name, code=future.result()
            print(json.dumps({'task_name':name,'exit_code':code}))
            if code: failures.append({'task_name':name,'exit_code':code})
    if failures:
        print(json.dumps({'status':'FAIL','failures':failures})); return 1
    print(json.dumps({'status':'PASS','tasks':len(commands)})); return 0
if __name__=='__main__': raise SystemExit(main())
