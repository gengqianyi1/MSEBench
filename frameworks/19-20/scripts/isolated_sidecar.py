"""Organizer-only relay. No participant Docker socket, shell, or arbitrary URL.

Sidecar uses --network none. The host Agent retains its HTTP protocol; this
relay forwards only fixed protocol routes over Docker exec/stdin to loopback.
Docker/runtime acceptance is mandatory before declaring this release ready.
"""
import argparse
import json
import re
import subprocess
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROUTES = {('GET', '/health'), ('POST', '/session'), ('POST', '/tool'), ('POST', '/submit')}
CLIENT = '''import json,sys,urllib.request,urllib.error
p=json.load(sys.stdin)
allowed={('GET','/health'),('POST','/session'),('POST','/tool'),('POST','/submit')}
assert (p['method'],p['path']) in allowed
assert p['port'] in (8819,8820)
data=None if p['method']=='GET' else json.dumps(p['body']).encode()
r=urllib.request.Request('http://127.0.0.1:'+str(p['port'])+p['path'],data=data,method=p['method'],headers={'Content-Type':'application/json'})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
 with opener.open(r,timeout=30) as response:
  print(json.dumps({'status':response.status,'body':json.loads(response.read())}))
except urllib.error.HTTPError as error:
 print(json.dumps({'status':error.code,'body':json.loads(error.read())}))
'''


def forward(container, port, method, path, body, runner=subprocess.run):
    if (method, path) not in ROUTES:
        return 404, {'ok': False, 'error': 'route_not_allowed'}
    if not isinstance(body, dict):
        return 400, {'ok': False, 'error': 'body_must_be_object'}
    try:
        result = runner(['docker', 'exec', '-i', container, 'python', '-c', CLIENT],
                        input=json.dumps(dict(port=port, method=method, path=path, body=body)),
                        text=True, capture_output=True, timeout=40, check=True)
        response = json.loads(result.stdout)
        status = response['status']
        if not isinstance(status, int) or not 100 <= status <= 599 or not isinstance(response['body'], dict):
            raise ValueError('invalid relay response')
        return status, response['body']
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return 502, {'ok': False, 'error': 'sidecar_transport_failure', 'status': 'infrastructure_error'}


def handler(container, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def dispatch(self):
            self.connection.settimeout(10)
            if (self.command, self.path) not in ROUTES:
                status, value = 404, {'ok': False, 'error': 'route_not_allowed'}
            else:
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 <= size <= 1048576 or self.headers.get('Transfer-Encoding'):
                        raise ValueError('body_size')
                    body = json.loads(self.rfile.read(size) or b'{}')
                    status, value = forward(container, port, self.command, self.path, body)
                except (ValueError, OSError):
                    status, value = 400, {'ok': False, 'error': 'invalid_request'}
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = dispatch
        do_POST = dispatch
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[19, 20], required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--platform', choices=['linux/amd64', 'linux/arm64'], required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[^\s@]+@sha256:[a-f0-9]{64}', args.image):
        parser.error('--image requires repository@sha256:manifest, not a tag or bare image ID')
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    port = 8800 + args.task
    container = f'matbench-{args.task}-{uuid.uuid4().hex[:12]}'
    # Bind first: an occupied organizer port must fail before creating a container.
    server = HTTPServer(('127.0.0.1', port), handler(container, port))
    started = False
    try:
        command = ['docker', 'run', '-d', '--pull=never', '--name', container,
                   '--platform', args.platform, '--network', 'none', '--read-only',
                   '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
                   '--mount', f'type=bind,src={evidence},dst=/organizer_evidence', args.image]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        started = True
        inspection = json.loads(subprocess.run(['docker', 'inspect', container], check=True,
                                              capture_output=True, text=True, timeout=15).stdout)[0]
        assert inspection['HostConfig']['NetworkMode'] == 'none'
        assert not inspection['HostConfig'].get('PortBindings')
        record = dict(task=args.task, container=container, image=args.image,
                      platform=args.platform, network_mode='none', published_ports=[],
                      transport='fixed-docker-exec', harbor_e2e_verified=False)
        (evidence / 'deployment_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'Organizer relay: http://127.0.0.1:{port}; Ctrl-C stops owned container', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        if started:
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=30)


if __name__ == '__main__':
    main()
