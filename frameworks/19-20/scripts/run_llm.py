"""Portable entrypoint. Host credentials only; no local deployment dependencies."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from run_docker_model_matrix import MODELS

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model', help='registered key or any model name (generic route)')
    parser.add_argument('--task', type=int, choices=(19, 20))
    args = parser.parse_args()
    if bool(args.model) != bool(args.task):
        parser.error('--model and --task must be supplied together')
    if args.prepare_only and args.model:
        parser.error('Preparation does not select a single cell')
    env = dict(os.environ)
    missing = []
    names = () if args.prepare_only else ((args.model.upper(),) if args.model else tuple(m.upper() for m in MODELS))
    for name in names:
        source = 'TASKS_' + name + '_API_KEY'
        target = 'CHEMBENCH_PROFILE_' + name + '_SECRET'
        if env.get(source):
            env[target] = env[source]
        elif env.get('MSEBENCH_API_KEY'):
            env[target] = env['MSEBENCH_API_KEY']
        if not env.get(target):
            missing.append(source)
    if missing:
        parser.error('Set these host environment variables: ' + ', '.join(missing))
    # Deliberately discard any inherited path to another local project.
    env['PYTHONPATH'] = str(ROOT)
    env['PYTHONNOUSERSITE'] = '1'
    subprocess.run(['docker', 'info'], check=True, capture_output=True, timeout=30)
    extra = ['--prepare-only'] if args.prepare_only else (
        ['--model', args.model, '--task', str(args.task)] if args.model else [])
    return subprocess.call([sys.executable, str(ROOT/'scripts/run_docker_model_matrix.py'),
                            '--campaign', str(args.campaign.resolve()), *extra], cwd=ROOT, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
