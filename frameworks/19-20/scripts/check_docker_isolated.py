"""Real network-none sidecar/relay/verifier checks; not Harbor or live LLM."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import HTTPServer
from pathlib import Path
from task_paths import task_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'release_evidence/docker_isolated_20260910'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def docker(*args):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True, timeout=60).stdout


def run(number):
    task_name = 'task-19-active-impurity-elucidation' if number == 19 else 'task-20-catalyst-deactivation-diagnosis'
    task = task_path(task_name)
    tests = task / 'tests'
    relay = load('relay', ROOT / 'scripts/isolated_sidecar.py')
    integration = load('integration', ROOT / 'scripts/run_task19_20_verifier_docker_integration.py')
    image_info = json.loads(docker('image', 'inspect', f'matbench/task{number}-sidecar:20260910-protocol2'))[0]
    image = image_info['Id']  # local immutable identity, not a published registry digest
    container = f'matbench-verify-{number}-{uuid.uuid4().hex[:10]}'
    root = OUT / f'task{number}'
    evidence_dir = root / 'organizer_evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    port = 8800 + number
    server = HTTPServer(('127.0.0.1', 0), relay.handler(container, port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    started = False
    cases = []
    try:
        docker('run', '-d', '--pull=never', '--name', container, '--platform', 'linux/amd64',
               '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m', '--mount',
               f'type=bind,src={evidence_dir},dst=/organizer_evidence', image)
        started = True
        thread.start()
        for _ in range(10):
            status, response = relay.forward(container, port, 'GET', '/health', {})
            if status == 200 and response.get('ok'):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('sidecar health failed')
        inspection = json.loads(docker('inspect', container))[0]
        assert inspection['HostConfig']['NetworkMode'] == 'none'
        assert not inspection['HostConfig'].get('PortBindings')
        probe = '''import json,socket
blocked=False
s=socket.socket();s.settimeout(2)
try:s.connect(('1.1.1.1',443))
except OSError:blocked=True
finally:s.close()
print(json.dumps({'interfaces':[name for _,name in socket.if_nameindex()],'egress_blocked':blocked}))'''
        network = json.loads(docker('exec', container, 'python', '-c', probe))
        assert network['interfaces'] == ['lo'] and network['egress_blocked']
        # Verify historical cached image contains the exact current private source.
        source_pairs = [('private_suite_adapter.py', '/private/private_suite_adapter.py'),
                        ('sidecar/hidden_suite_server.py', '/private/hidden_suite_server.py')]
        source_pairs += [(p.relative_to(tests).as_posix(), '/private/' + p.relative_to(tests).as_posix())
                         for p in (tests / 'task_snapshot').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        for local, remote in source_pairs:
            digest = docker('exec', container, 'python', '-c',
                            'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())', remote).strip()
            assert digest == hashlib.sha256((tests / local).read_bytes()).hexdigest(), local
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(route, value):
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}' + route,
                                         data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(req, timeout=50) as response:
                return json.load(response)

        fixtures = [tests / 'fixtures/oracle.json', *sorted((tests / 'fixtures/valid_wrong').glob('*.json')),
                    *sorted((tests / 'fixtures/invalid').glob('*.json'))]
        for fixture in fixtures:
            candidate = json.loads(fixture.read_text())
            session = post('/session', {})
            token = session['session_id']
            for tool, arguments in integration.TASKS[task_name]['calls']:
                assert post('/tool', dict(session_id=token, tool=tool, arguments=arguments))['ok']
            submitted = post('/submit', dict(session_id=token, candidate=candidate))
            if fixture.parent.name == 'invalid':
                assert not submitted['ok']
                cases.append(dict(fixture=fixture.stem, schema_valid=False, status='INVALID_SUBMISSION', reward=None))
                continue
            assert submitted['ok']
            evidence = json.loads((evidence_dir / token / 'case-1-evidence.json').read_text())
            evidence.update(public_case=session['case'], submission=submitted)
            workspace = root / fixture.stem / 'workspace'
            target = workspace / 'cases/case-1'
            target.mkdir(parents=True, exist_ok=True)
            (target / 'case_result.json').write_text(json.dumps(evidence, indent=2))
            (workspace / 'final_answer.json').write_text(json.dumps({'candidate': candidate}, indent=2))
            verifier = integration._image_tag(task_name)
            logs = root / fixture.stem / 'verifier'
            code, stdout, stderr = integration._run_entrypoint(verifier, workspace, logs)
            result = integration._read_result(logs, code)
            assert code == 0
            assert result['score'] == 100 if fixture.stem == 'oracle' else 0 <= result['score'] < 100
            cases.append(dict(fixture=fixture.stem, schema_valid=True, **result))
        docker('stop', '--time', '2', container)
        transport_status, transport_body = relay.forward(container, port, 'POST', '/tool', {})
        assert transport_status == 502 and transport_body['status'] == 'infrastructure_error'
        report = dict(task=task_name, status='PASS', sidecar_image=image, platform='linux/amd64',
                      source_files_identical=len(source_pairs), network=network, published_ports=[],
                      transport_failure=dict(http_status=transport_status, body=transport_body),
                      scope='real Docker network-none sidecar -> host HTTP relay -> attested workspace -> separate network-none Docker verifier; deterministic fixture driver, NOT Harbor/live LLM', cases=cases)
        (root / 'result.json').write_text(json.dumps(report, indent=2))
        return report
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
        if started:
            docker('rm', '-f', container)


if __name__ == '__main__':
    reports = []
    for number in (19, 20):
        reports.append(run(number))
        print(f'Task {number} isolated Docker checks PASS', flush=True)
    (OUT / 'summary.json').write_text(json.dumps(reports, indent=2))
