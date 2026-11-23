import argparse
import asyncio
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from provider_policy import ProviderFailure, retryable, matrix_complete
from task_paths import task_path, PACKAGE

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "default": ("", "", "MSEBENCH_API_KEY"),
}

TASKS = {19: 'task-19-active-impurity-elucidation', 20: 'task-20-catalyst-deactivation-diagnosis'}


def resolve_route(model):
    """Resolve a model key to (provider_model_id, endpoint, key_env_name).

    Registered models use the frozen MODELS registry. Any other name falls back
    to the ambient CHEMBENCH_* route (profile's model/api_base/api_key), so an
    arbitrary LLM can be evaluated without editing MODELS.
    """
    endpoint = os.environ.get('CHEMBENCH_API_URL') or (
        os.environ.get('CHEMBENCH_API_BASE', '').rstrip('/') + '/chat/completions')
    return (os.environ.get('CHEMBENCH_MODEL_ID') or model, endpoint, 'CHEMBENCH_API_KEY')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def docker(*args, timeout=60):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True, timeout=timeout).stdout


class DockerWorkspace:
    def __init__(self, container):
        self.container = container

    async def exec(self, command, cwd='/workspace', timeout_sec=30, **kwargs):
        # Only trusted unchanged Agent code constructs these shell commands.
        # Model tools never receive an exec/shell capability for these two tasks.
        done = subprocess.run(['docker', 'exec', '-w', cwd, self.container, 'sh', '-c', command],
                              capture_output=True, text=True, timeout=timeout_sec)
        return SimpleNamespace(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode)

    async def upload_file(self, source, destination):
        docker('cp', str(source), self.container + ':' + destination)


def cell(campaign, model, number):
    root = campaign / f'{model}-task{number}'
    root.mkdir(exist_ok=False)
    plan = json.loads((campaign / 'frozen_plan.json').read_text())
    stale = [rel for rel, expected in plan['files'].items()
             if not (ROOT / rel).is_file() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected]
    if stale:
        # 2026-09-22: fail loudly and leave a readable result.json. The old
        # bare assert produced an empty cell directory with no verdict
        # (2026-09-21: two runs silently lost this way).
        message = ('frozen_plan stale, runner sources changed: ' + ', '.join(sorted(stale)) +
                   '. Rerun: python scripts/run_llm.py --campaign ' + str(campaign) +
                   ' --prepare-only, then relaunch the task.')
        (root / 'result.json').write_text(json.dumps(
            dict(model=model, task=number, status='INFRASTRUCTURE_FAILURE', score=None,
                 failure_code='FROZEN_PLAN_STALE', error_message=message,
                 mode='DOCKER_COMPONENTS_NOT_HARBOR', started_at=time.time()), indent=2))
        raise RuntimeError(message)
    provider_id, endpoint, key_name = resolve_route(model)
    for env_name, field in [('CHEMBENCH_MAX_TOKENS', 'max_tokens'),
                            ('CHEMBENCH_TEMPERATURE', 'temperature'),
                            ('CHEMBENCH_REQUEST_RETRIES', 'request_attempts'),
                            ('CHEMBENCH_REQUEST_TIMEOUT', 'request_timeout_seconds')]:
        configured = os.environ.get(env_name)
        if configured is not None and float(configured) != float(plan[field]):
            raise RuntimeError(f'{env_name} differs from frozen plan; prepare a new campaign')
        os.environ[env_name] = str(plan[field])
    key = os.environ.get(key_name, '')
    if not key:
        raise RuntimeError('configured model key missing')
    os.environ.update(CHEMBENCH_API_KEY=key,
                      CHEMBENCH_API_URL=os.environ.get('CHEMBENCH_API_URL', endpoint),
                      CHEMBENCH_MODEL_ID=os.environ.get('CHEMBENCH_MODEL_ID', provider_id),
                      CHEMBENCH_MAX_TOKENS=os.environ.get('CHEMBENCH_MAX_TOKENS', '32768'),
                      CHEMBENCH_TEMPERATURE=os.environ.get('CHEMBENCH_TEMPERATURE', '0'),
                      CHEMBENCH_REQUEST_RETRIES=os.environ.get('CHEMBENCH_REQUEST_RETRIES', '5'),
                      CHEMBENCH_REQUEST_TIMEOUT=os.environ.get('CHEMBENCH_REQUEST_TIMEOUT', '1800'),
                      CHEMBENCH_USE_NATIVE_TOOL_CALLS='true', CHEMBENCH_LLM_CALL_LOG=str(root / 'provider_calls.jsonl'),
                      CHEMBENCH_ENABLE_THINKING=os.environ.get('CHEMBENCH_ENABLE_THINKING', 'false'),
                      CHEMBENCH_REASONING_EFFORT=os.environ.get('CHEMBENCH_REASONING_EFFORT', 'low'))
    task_name = TASKS[number]
    task = task_path(task_name)
    relay = load('relay', ROOT / 'scripts/isolated_sidecar.py')
    integration = load('integration', ROOT / 'scripts/run_task19_20_verifier_docker_integration.py')
    agent_class = integration._agent_class()
    agent = agent_class.__new__(agent_class)
    def bounded_request(body, timeout):
        clean = {k: v for k, v in body.items() if k != '__api_key'}
        _attempts = max(1, int(os.environ.get('CHEMBENCH_REQUEST_RETRIES', '5')))
        _hard_timeout = float(os.environ.get('CHEMBENCH_REQUEST_TIMEOUT', '1800')) + 10.0
        for attempt in range(1, _attempts + 1):
            began = time.time()
            record = dict(request_id=uuid.uuid4().hex, attempt=attempt, started_unix=began,
                          endpoint=endpoint, request=clean)
            try:
                response = subprocess.run([sys.executable, str(ROOT / 'scripts/provider_request_once.py')],
                    input=json.dumps(clean), capture_output=True, text=True, timeout=_hard_timeout, check=True)
                outcome = json.loads(response.stdout)
            except subprocess.TimeoutExpired:
                outcome = {'http_status': None, 'error_type': 'HardRequestTimeout'}
            except Exception as error:
                outcome = {'http_status': None, 'error_type': type(error).__name__}
            record.update(outcome, elapsed_seconds=round(time.time()-began, 3))
            agent._record_provider_call(record)
            if outcome.get('http_status') == 200 and isinstance(outcome.get('response'), dict):
                return outcome['response']
            if not retryable(outcome.get('http_status')):
                break
        raise ProviderFailure(outcome.get('http_status'))
    agent._request_json = bounded_request
    context = SimpleNamespace(metadata={})
    sidecar = 'matrix-sidecar-' + uuid.uuid4().hex[:12]
    public = 'matrix-public-' + uuid.uuid4().hex[:12]
    owned = []
    server = None
    thread = None
    result = dict(model=model, provider_model_id=provider_id, task=number, status='RUNNING', score=None,
                  mode='DOCKER_COMPONENTS_NOT_HARBOR', started_at=time.time())
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    try:
        evidence = root / 'organizer_evidence'
        evidence.mkdir()
        os.chmod(evidence, 0o777)
        workspace = root / 'workspace'
        shutil.copytree(task / 'environment/public', workspace)
        for path in [workspace, *[p for p in workspace.rglob('*') if p.is_dir()]]:
            os.chmod(path, 0o777)
        shutil.copyfile(task / 'instruction.md', workspace / 'instruction.md')
        sidecar_image = plan['images'][f'sidecar{number}']
        public_image = plan['images'][f'public{number}']
        docker('run', '-d', '--pull=never', '--name', sidecar, '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '2', '--memory', '1g',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m', '--mount', f'type=bind,src={evidence},dst=/organizer_evidence', sidecar_image)
        owned.append(sidecar)
        docker('run', '-d', '--pull=never', '--name', public, '--network', 'none', '--cpus', '2', '--memory', '4g',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--mount', f'type=bind,src={workspace},dst=/workspace', public_image,
               'python', '-c', 'import time;time.sleep(4200)')
        owned.append(public)
        for name in owned:
            info = json.loads(docker('inspect', name))[0]
            assert info['HostConfig']['NetworkMode'] == 'none'
        server = HTTPServer(('127.0.0.1', 0), relay.handler(sidecar, 8800 + number))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for _ in range(10):
            status, body = relay.forward(sidecar, 8800 + number, 'GET', '/health', {})
            if status == 200 and body.get('ok'):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('sidecar health failed')
        cfg = json.loads((workspace / '.chembench_agent.json').read_text())
        os.environ[cfg['sidecar_url_env']] = f'http://127.0.0.1:{server.server_port}'
        asyncio.run(agent._run_hidden_suite_case('case 1\n' + (task / 'instruction.md').read_text(),
                                               DockerWorkspace(public), context, cfg))
        failure = context.metadata.get('failure')
        result['agent_metadata'] = context.metadata
        artifact = workspace / 'cases/case-1/case_result.json'
        if failure and ('infrastructure' in failure or 'provenance' in failure):
            result.update(status='INFRASTRUCTURE_FAILURE', failure=failure)
        elif failure == 'sidecar_submission_rejected' or not artifact.is_file():
            result.update(status='INVALID_SUBMISSION', score=0, failure=failure or 'missing_submission')
        else:
            code, stdout, stderr = integration._run_entrypoint(plan['images'][f'verifier{number}'], workspace, root / 'verifier')
            graded = integration._read_result(root / 'verifier', code)
            result['grader'] = graded
            if code or 'ORGANIZER_EVIDENCE_ERROR' in graded.get('failure_codes', []):
                result.update(status='EVIDENCE_OR_INFRASTRUCTURE_FAILURE', score=None)
            else:
                result.update(status='COMPLETED', score=graded['score'])
    except Exception as error:
        # Do not echo provider response bodies, headers, or credentials.
        result.update(status='INFRASTRUCTURE_FAILURE', score=None, error_type=type(error).__name__,
                      http_status=getattr(error, 'code', None),
                      failure_code=getattr(error, 'failure_code', 'RUNTIME_INFRASTRUCTURE_FAILURE'),
                      error_message=str(error)[:500])
    finally:
        if server is not None:
            if thread and thread.is_alive():
                server.shutdown()
                thread.join(timeout=5)
            server.server_close()
        for name in reversed(owned):
            try:
                docker('rm', '-f', name)
            except Exception:
                pass
        result['elapsed_seconds'] = round(time.time() - result['started_at'], 2)
        (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ('model', 'task', 'status', 'score', 'http_status')}) , flush=True)
    return 0 if result['status'] == 'COMPLETED' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--model', help='registered key or any model name (generic route)')
    parser.add_argument('--task', type=int, choices=TASKS)
    parser.add_argument('--prepare-only', action='store_true', help='Build and freeze; make no paid API requests')
    args = parser.parse_args()
    if bool(args.model) != bool(args.task):
        parser.error('--model and --task must be supplied together')
    if args.prepare_only and args.model:
        parser.error('--prepare-only creates a new full plan; omit --model/--task')
    if args.model:
        return cell(args.campaign.resolve(), args.model, args.task)
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    images = {}
    for number, name in TASKS.items():
        for kind, tag in [('public', f'matrix-task{number}-public:20260910')]:
            docker('build', '-t', tag, str(task_path(name) / 'environment'), timeout=300)
            images[f'{kind}{number}'] = json.loads(docker('image', 'inspect', tag))[0]['Id']
        task = task_path(name)
        sidecar_tag = f'matbench/task{number}-sidecar:20260910-protocol2'
        verifier_tag = f'matbench/task{number}-verifier:20260910-protocol2'
        docker('build', '-f', str(task / 'tests/sidecar/Dockerfile'), '-t', sidecar_tag, str(task), timeout=300)
        docker('build', '-t', verifier_tag, str(task / 'tests'), timeout=300)
        images[f'sidecar{number}'] = json.loads(docker('image', 'inspect', sidecar_tag))[0]['Id']
        images[f'verifier{number}'] = json.loads(docker('image', 'inspect', verifier_tag))[0]['Id']
    paths = [ROOT / 'agents/chembench_agent.py', ROOT / 'scripts/run_docker_model_matrix.py', ROOT / 'scripts/isolated_sidecar.py', ROOT / 'scripts/provider_request_once.py', ROOT / 'scripts/provider_policy.py', ROOT / 'scripts/run_task19_20_verifier_docker_integration.py', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    paths.append(ROOT / 'scripts/task_paths.py')
    paths += list((PACKAGE / 'src/msebench/context').glob('*.py'))
    paths += [p for n in TASKS for p in task_path(n).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    plan = dict(mode='DOCKER_COMPONENTS_NOT_HARBOR', models={k: dict(model=v[0], endpoint=v[1]) for k, v in MODELS.items()},
                images=images, tasks=TASKS,
                temperature=float(os.environ.get('CHEMBENCH_TEMPERATURE', '0')),
                max_tokens=int(os.environ.get('CHEMBENCH_MAX_TOKENS', '32768')),
                request_timeout_seconds=int(os.environ.get('CHEMBENCH_REQUEST_TIMEOUT', '1800')),
                request_attempts=int(os.environ.get('CHEMBENCH_REQUEST_RETRIES', '5')),
                native_tool_calls=True, max_cell_wall_time_seconds=4100, repetitions_per_cell=1,
                files={Path(os.path.relpath(p, ROOT)).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (campaign / 'frozen_plan.json').write_text(json.dumps(plan, indent=2))
    if args.prepare_only:
        print('Plan prepared. No model API requests made.', flush=True)
        return 0
    print(f'Frozen {len(MODELS) * len(TASKS)}-cell plan. Starting real API calls.', flush=True)

    def worker(model, number):
        try:
            done = subprocess.run([sys.executable, __file__, '--campaign', str(campaign), '--model', model, '--task', str(number)],
                                  capture_output=True, text=True, timeout=4100)
        except subprocess.TimeoutExpired:
            cell_root = campaign / f'{model}-task{number}'
            cell_root.mkdir(exist_ok=True)
            (cell_root / 'result.json').write_text(json.dumps(dict(model=model, task=number,
                status='INFRASTRUCTURE_FAILURE', score=None, failure_code='WORKER_TIMEOUT',
                mode='DOCKER_COMPONENTS_NOT_HARBOR', cleanup_required=True), indent=2))
            print(f'{model}-task{number}: worker timeout; organizer must inspect owned containers.', flush=True)
            return
        print(done.stdout.strip() or json.dumps(dict(model=model, task=number, exit_code=done.returncode)), flush=True)
        if done.returncode:
            (campaign / f'{model}-task{number}-worker-error.txt').write_text('Worker failed; inspect cell result and Docker state.\n')
    # Each cell is a separate process: credentials and session state are never shared.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, model, number) for number in TASKS for model in MODELS]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    result_paths = [campaign / f'{model}-task{number}/result.json' for model in MODELS for number in TASKS]
    results = [json.loads(p.read_text()) for p in result_paths if p.is_file()]
    (campaign / 'matrix_results.json').write_text(json.dumps(results, indent=2))
    print('Matrix finished. Partial or infra outcomes must not be used for ranking.', flush=True)
    return 0 if matrix_complete(results, models=MODELS) else 2


if __name__ == '__main__':
    raise SystemExit(main())
