import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('relay', Path(__file__).with_name('isolated_sidecar.py'))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


class RelayTests(unittest.TestCase):
    def test_fixed_command_and_stdin(self):
        candidate = {'candidate': {'text': '"; touch /tmp/forbidden; #'}}
        def fake(command, **kwargs):
            self.assertEqual(command[:4], ['docker', 'exec', '-i', 'owned-container'])
            self.assertNotIn('shell', kwargs)
            self.assertEqual(json.loads(kwargs['input'])['body'], candidate)
            return SimpleNamespace(stdout='{"status":200,"body":{"ok":true}}')
        self.assertEqual(relay.forward('owned-container', 8820, 'POST', '/submit', candidate, fake)[0], 200)

    def test_arbitrary_route_rejected_without_exec(self):
        def forbidden(*args, **kwargs):
            self.fail('must not invoke Docker for an unapproved route')
        for path in ['/exec', 'http://example.com', '/tool?url=http://example.com', '/../private']:
            self.assertEqual(relay.forward('owned-container', 8820, 'POST', path, {}, forbidden)[0], 404)

    def test_transport_errors_never_return_scientific_score(self):
        for error in [OSError(), subprocess.TimeoutExpired('docker', 40), subprocess.CalledProcessError(1, 'docker')]:
            def fake(*args, **kwargs):
                raise error
            status, body = relay.forward('owned-container', 8820, 'POST', '/tool', {}, fake)
            self.assertEqual(status, 502)
            self.assertEqual(body['status'], 'infrastructure_error')
            self.assertNotIn('score', body)
            self.assertNotIn('reward', body)

    def test_protocol_status_preserved(self):
        for status in (400, 500):
            fake = lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps({'status': status, 'body': {'ok': False}}))
            self.assertEqual(relay.forward('owned-container', 8820, 'POST', '/tool', {}, fake)[0], status)


if __name__ == '__main__':
    unittest.main()
