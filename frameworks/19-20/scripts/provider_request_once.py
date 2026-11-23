"""One provider exchange; parent enforces a hard subprocess wall-clock timeout."""
import json
import os
import sys
import urllib.error
import urllib.request


def _sse_read(response, requested_model):
    """Reassemble an SSE stream into a REST-shaped response dict.

    The antchat gateway drops idle connections; long reasoning generations
    must stream to keep bytes flowing (2026-09-21: this subprocess was the
    last non-streaming request path -- 19-20's matrix replaces the agent's
    _request_json with this wrapper, so the agent-side SSE fix never ran).
    Providers that ignore stream=true and return plain JSON are handled by
    the caller's non-SSE branch below.
    """
    role = None
    content_parts = []
    reasoning_parts = []
    tc_map = {}
    finish_reason = None
    model_name = None
    for raw_line in response:
        line = raw_line.decode('utf-8', 'replace').strip()
        if not line or line.startswith(':') or not line.startswith('data:'):
            continue
        payload = line[len('data:'):].strip()
        if not payload:
            continue
        if payload == '[DONE]':
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get('model'), str) and chunk['model']:
            model_name = chunk['model']
        choices = chunk.get('choices')
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if choice.get('finish_reason'):
            finish_reason = choice['finish_reason']
        delta = choice.get('delta') or choice.get('message') or {}
        if not isinstance(delta, dict):
            continue
        if delta.get('role'):
            role = delta['role']
        if isinstance(delta.get('content'), str):
            content_parts.append(delta['content'])
        if isinstance(delta.get('reasoning_content'), str):
            reasoning_parts.append(delta['reasoning_content'])
        for call in delta.get('tool_calls') or []:
            if not isinstance(call, dict):
                continue
            index = call.get('index', 0)
            if not isinstance(index, int):
                index = 0
            entry = tc_map.setdefault(index, {'id': None, 'type': 'function',
                                              'function': {'name': '', 'arguments': ''}})
            if isinstance(call.get('id'), str) and call['id']:
                entry['id'] = call['id']
            function = call.get('function') or {}
            if isinstance(function, dict):
                if isinstance(function.get('name'), str) and function['name']:
                    entry['function']['name'] = function['name']
                if isinstance(function.get('arguments'), str):
                    entry['function']['arguments'] += function['arguments']
    message = {'role': role or 'assistant', 'content': ''.join(content_parts)}
    if reasoning_parts:
        message['reasoning_content'] = ''.join(reasoning_parts)
    tool_calls = [tc_map[k] for k in sorted(tc_map)]
    if tool_calls:
        message['tool_calls'] = tool_calls
    return {'model': model_name or requested_model,
            'choices': [{'message': message, 'finish_reason': finish_reason or 'stop'}]}


body = json.load(sys.stdin)
stream_pref = os.environ.get('CHEMBENCH_STREAM', '')
want_stream = True if not stream_pref else stream_pref.strip().lower() in ('1', 'true', 'yes', 'on')
if want_stream:
    body = dict(body)
    body['stream'] = True
headers = {'Content-Type': 'application/json',
           'Authorization': 'Bearer ' + os.environ['CHEMBENCH_API_KEY']}
if want_stream:
    headers['Accept'] = 'text/event-stream'
request = urllib.request.Request(os.environ['CHEMBENCH_API_URL'],
                                 data=json.dumps(body).encode(), headers=headers)
try:
    _timeout = float(os.environ.get('CHEMBENCH_REQUEST_TIMEOUT', '300')) - 10.0
    with urllib.request.urlopen(request, timeout=_timeout) as response:
        ctype = response.headers.get('Content-Type') or ''
        if 'text/event-stream' in ctype:
            data = _sse_read(response, body.get('model'))
        else:
            data = json.loads(response.read())
        print(json.dumps({'http_status': response.status, 'response': data}))
except urllib.error.HTTPError as error:
    print(json.dumps({'http_status': error.code, 'error': error.read().decode('utf-8', 'replace')}))
except Exception as error:
    print(json.dumps({'http_status': None, 'error_type': type(error).__name__}))
