"""Smoke-test the declared Task 01/02 sidecar release deployments."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / 'templates/sidecar/hidden_suite_server.py'
PACKAGE_ROOT = ROOT.parent
TASKS = {
    'task01': PACKAGE_ROOT / 'tasks/task01/tasks/task-01-jarvis-2d-exfoliation-candidate',
    'task02': PACKAGE_ROOT / 'tasks/task02/tasks/task-02-oligothiophene-smd-sapt',
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=TASKS)
    args = parser.parse_args()
    task = TASKS[args.task]
    spec = yaml.safe_load((task / 'environment/public/sidecar_deployment.yaml').read_text(encoding='utf-8'))
    deployment, release = spec['deployment'], spec['release']
    image = deployment['image_tag']
    inspect = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
    observed = inspect['RepoDigests'][0].split('@')[-1] if inspect.get('RepoDigests') else inspect['Id']
    if observed != release['private_image_digest']:
        raise RuntimeError(f'image digest mismatch: {observed} != {release["private_image_digest"]}')
    port = int(deployment['host_bind'].rsplit(':', 1)[1])
    name = f'chembench-{args.task}-release-smoke'
    with tempfile.TemporaryDirectory(prefix=f'{args.task}-sidecar-smoke-') as evidence:
        command = ['docker', 'run', '-d', '--user', deployment['container_user'], '--name', name,
            '-p', f'127.0.0.1:{port}:{port}',
            '-v', f'{SERVER.resolve().as_posix()}:/sidecar.py:ro',
            '-v', f'{(task / "tests/private_suite_adapter.py").resolve().as_posix()}:/tests/private_suite_adapter.py:ro',
            '-v', f'{(task / "tests/task_snapshot").resolve().as_posix()}:/task:ro',
            '-v', f'{Path(evidence).resolve().as_posix()}:/organizer_evidence', image, *deployment['entrypoint']]
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            health = None
            for _ in range(120):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        health = json.loads(response.read())
                    break
                except Exception:
                    time.sleep(0.5)
            if health is None:
                raise RuntimeError('sidecar did not become healthy')
        finally:
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True).stdout
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    result = {
        'status': 'PASS', 'task': args.task, 'image_tag': image,
        'declared_digest': release['private_image_digest'], 'observed_digest': observed,
        'health': health, 'deployment_fields_verified': sorted(deployment),
        'container_log_tail': logs[-2000:],
    }
    output = task / 'tests/private_data/oracle_bundle/verifier/sidecar_release_smoke.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
