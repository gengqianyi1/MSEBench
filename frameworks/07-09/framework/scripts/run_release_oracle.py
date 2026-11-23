"""Run a real Harbor multistep Oracle job against the released local sidecar."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from check_release_job import check_job

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True, choices=[7,8,9])
    parser.add_argument('--name', default='oracle')
    parser.add_argument('--wrong', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--sidecar-host', default='127.0.0.1')
    args = parser.parse_args()
    task = next((ROOT/'tasks').glob(f'task-{args.task:02d}-*'))
    output = ROOT/'release_evidence'/f'task{args.task:02d}'
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT), 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8',
                'MSEBENCH_RELEASE_TASK': str(args.task),
                'MSEBENCH_RELEASE_WRONG': '1' if args.wrong else '0',
                f'CHEMBENCH_TASK{args.task:02d}_SIDECAR_URL':f'http://{args.sidecar_host}:{args.port or 18700+args.task}',
                'CHEMBENCH_USE_NATIVE_TOOL_CALLS':'', 'NO_COLOR':'1'})
    config = {'job_name':args.name, 'jobs_dir':(output/'jobs').as_posix(),
              'n_concurrent_trials':1, 'quiet':True,
              'tasks':[{'path':task.as_posix()}],
              'agents':[{'import_path':'agents.release_oracle:ReleaseOracleAgent'}],
              'verifier':{'env':{'CHEMBENCH_ORACLE_VALIDATION':'1'}},
              'environment':{'import_path':'agents.release_environment:OrganizerDockerEnvironment',
                             'delete':True, 'kwargs':{'evidence_dir':(output/'organizer_evidence').as_posix()}}}
    path = output/f'{args.name}.harbor.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    executable = Path(sys.executable).parent / ('harbor.exe' if os.name == 'nt' else 'harbor')
    with (output/f'{args.name}.log').open('w',encoding='utf-8') as log:
        result = subprocess.run([str(executable),'run','--config',str(path),'--yes'], cwd=ROOT, env=env, stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print(json.dumps({'task':args.task,'exit_code':result.returncode,'output':str(output)}))
        return result.returncode
    audit=check_job(output/'jobs'/args.name,args.task,args.wrong)
    (output/f'{args.name}.audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))
    return 0 if audit['status']=='PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
