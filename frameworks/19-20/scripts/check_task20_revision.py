"""Local HTTP sidecar -> signed evidence -> private grader CLI regression.

This exercises real HTTP and a separate process, but is NOT Docker/Harbor/LLM E2E.
"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from task_paths import task_path
TESTS = task_path(20) / 'tests'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    adapter = load('t20adapter', TESTS / 'private_suite_adapter.py').create_adapter(TESTS / 'task_snapshot')
    service = load('t20server', TESTS / 'sidecar/hidden_suite_server.py')
    oracle = json.loads((TESTS / 'fixtures/oracle.json').read_text())
    cases = {'oracle': copy.deepcopy(oracle)}
    for name in ('wrong_primary', 'wrong_claims', 'wide_intervals', 'unsafe_oxidation', 'no_claims'):
        c = copy.deepcopy(oracle)
        if name == 'wrong_primary':
            c['primary_mechanisms'] = ['sulfur_poisoning', 'external_mass_transfer']
            c['contributions'] = dict(coking=0.05, sintering=0.05, sulfur_poisoning=0.5, pore_blockage=0.05, external_mass_transfer=0.35)
        elif name == 'wrong_claims':
            for claim in c['claims']:
                claim['relation'] = 'contradicts'
        elif name == 'wide_intervals':
            c['predicted_performance'] = dict(conversion=[0, 1], selectivity=[0, 1])
        elif name == 'unsafe_oxidation':
            c['regeneration_plan'][0]['temperature_C'] = 900
            c['regeneration_plan'][0]['oxygen_percent'] = 21
        else:
            c['claims'] = []
        cases[name] = c
    calls = [('tpo', {'sample': 'spent'}), ('bet', {'sample': 'spent'}),
             ('xps', {'region': 'S2p'}), ('particle_size_test', {'mode': 'tem'})]
    records = []
    out = ROOT / 'release_evidence/review_20260910'
    out.mkdir(parents=True, exist_ok=True)
    fixture_dir = TESTS / 'fixtures/valid_wrong'
    fixture_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='task20-regression-') as temp:
        state = service.State(adapter, Path(temp))
        server = ThreadingHTTPServer(('127.0.0.1', 0), service.handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(route, data):
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}' + route,
                                             data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=10) as response:
                return json.load(response)

        try:
            for name, candidate in cases.items():
                session = post('/session', {})
                token = session['session_id']
                for tool, arguments in calls:
                    assert post('/tool', dict(session_id=token, tool=tool, arguments=arguments))['ok']
                submitted = post('/submit', dict(session_id=token, candidate=candidate))
                assert submitted['ok'], (name, 'schema-invalid')
                evidence = Path(temp) / token / 'case-1-evidence.json'
                logs = Path(temp) / ('logs-' + name)
                env = dict(os.environ, CHEMBENCH_ORGANIZER_EVIDENCE=str(evidence), CHEMBENCH_LOGS=str(logs))
                subprocess.run([sys.executable, str(TESTS / 'protocol_grader.py')], env=env,
                               check=True, capture_output=True, text=True, timeout=30)
                result = json.loads((logs / 'grader-result.json').read_text())
                reward = float((logs / 'reward.txt').read_text())
                assert abs(reward - result['score'] / 100) < 1e-9
                assert result['score'] == 100 if name == 'oracle' else 0 <= result['score'] < 100
                # Repeat the exact attested evidence; no science or fixture rewriting.
                subprocess.run([sys.executable, str(TESTS / 'protocol_grader.py')], env=env,
                               check=True, capture_output=True, text=True, timeout=30)
                assert json.loads((logs / 'grader-result.json').read_text()) == result
                saved = out / 'task20_cases' / name
                saved.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(evidence, saved / 'case-1-evidence.json')
                shutil.copyfile(logs / 'grader-result.json', saved / 'grader-result.json')
                shutil.copyfile(logs / 'reward.txt', saved / 'reward.txt')
                (saved / 'submission.json').write_text(json.dumps({'candidate': candidate}, indent=2))
                if name != 'oracle':
                    (fixture_dir / (name + '.json')).write_text(json.dumps(candidate, indent=2) + '\n')
                records.append(dict(fixture=name, schema_valid=True, score=result['score'], reward=reward,
                                    subscores=result['subscores'], failure_codes=result.get('failure_codes'),
                                    deterministic_replay=True, evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest()))
                try:
                    post('/submit', dict(session_id=token, candidate=candidate))
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                else:
                    raise AssertionError('second terminal submit accepted')
            # Fifth diagnostic call must fail even if the first four were rejected.
            for invalid_calls in (False, True):
                token = post('/session', {})['session_id']
                for _ in range(4):
                    post('/tool', dict(session_id=token, tool='unknown' if invalid_calls else 'tpo', arguments={'sample': 'spent'}))
                try:
                    post('/tool', dict(session_id=token, tool='tpo', arguments={'sample': 'spent'}))
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                else:
                    raise AssertionError('fifth diagnostic accepted')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    report = dict(status='PASS', scope='native localhost HTTP sidecar and private grader subprocess; NOT Harbor/Docker/live LLM',
                  terminal_rejection=True, successful_and_rejected_calls_consume_budget=True, cases=records)
    (out / 'task20_http_private_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
