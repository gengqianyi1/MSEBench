"""Unified Harbor external agent for ChemBench migration.

This is a reference scaffold, not a Harbor built-in. It ports the task-agnostic
JSON/Python loop used by the legacy generic-agent tasks (10-13) into one Harbor
Agent. It intentionally does not implement task-native hidden protocols; those
must move into task environments/sidecars before parity.
"""
from __future__ import annotations
import socket as _socket
import urllib.parse as _urlparse
import base64, json, os, re, tempfile, time, urllib.error, urllib.request, uuid
import http.client as _http_client
from pathlib import Path

from harbor.agents.base import BaseAgent
from chembench_harness.protocol import resolve_case_number
try:
    from msebench.context.compactor import ContextCompactor
except ImportError:  # harness may run without src/ on PYTHONPATH; load by explicit path
    import importlib.util as _ilu
    _spec=_ilu.spec_from_file_location('matbench_context_compactor',str(next(parent / 'src/msebench/context/compactor.py' for parent in Path(__file__).resolve().parents if (parent / 'src/msebench/context/compactor.py').is_file())))
    _mod=_ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod); ContextCompactor=_mod.ContextCompactor


def _read_chat_completion(response):
    """Read an OpenAI-compatible SSE stream and rebuild a JSON response.

    The gateway resets long *non-streaming* (buffered) completions after about
    90 s; sending stream=True keeps the connection active.  Providers that
    ignore stream=true and return ordinary JSON are also accepted.
    """
    content_parts={}; reasoning_parts={}; typo_reasoning_parts={}
    choices={}; tool_calls={}; base={}
    first_nonempty=True

    def consume(chunk):
        nonlocal base
        if not isinstance(chunk, dict):
            return
        if not base:
            base={k:v for k,v in chunk.items() if k!='choices'}
        else:
            for k,v in chunk.items():
                if k!='choices':
                    base[k]=v
        for choice in chunk.get('choices') or []:
            if not isinstance(choice, dict):
                continue
            index=int(choice.get('index',0))
            state=choices.setdefault(index,{'finish_reason':None,'role':'assistant'})
            if choice.get('finish_reason') is not None:
                state['finish_reason']=choice.get('finish_reason')
            delta=choice.get('delta') or choice.get('message') or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get('role'), str):
                state['role']=delta['role']
            for key, store in (('content',content_parts),
                               ('reasoning_content',reasoning_parts),
                               ('reasoning_conetent',typo_reasoning_parts)):
                value=delta.get(key)
                if isinstance(value, str):
                    store.setdefault(index,[]).append(value)
            deltas=delta.get('tool_calls')
            if isinstance(deltas, list):
                for item in deltas:
                    if not isinstance(item, dict):
                        continue
                    call_index=int(item.get('index',0))
                    call=tool_calls.setdefault(index,{}).setdefault(call_index,{'id':'','type':'function','function':{'name':'','arguments':''}})
                    for key in ('id','type'):
                        if isinstance(item.get(key), str) and item[key]:
                            call[key]=item[key]
                    function=item.get('function') or {}
                    if isinstance(function, dict):
                        if isinstance(function.get('name'), str):
                            call['function']['name']+=function['name']
                        if isinstance(function.get('arguments'), str):
                            call['function']['arguments']+=function['arguments']

    while True:
        line=response.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line=line.decode('utf-8','replace')
        line=line.strip()
        if not line or line.startswith(':') or line.startswith('event:'):
            continue
        if line.startswith('data:'):
            payload=line[5:].strip()
            if payload=='[DONE]':
                continue
            try:
                consume(json.loads(payload))
            except json.JSONDecodeError:
                continue
        elif first_nonempty:
            first_nonempty=False
            remainder=line+'\n'+response.read().decode('utf-8','replace')
            return json.loads(remainder)

    result=dict(base)
    assembled=[]
    for index in sorted(choices):
        state=choices[index]
        message={'role':state.get('role','assistant'),'content':''.join(content_parts.get(index,[]))}
        reasoning=''.join(reasoning_parts.get(index,[]))
        typo_reasoning=''.join(typo_reasoning_parts.get(index,[]))
        if reasoning:
            message['reasoning_content']=reasoning
        if typo_reasoning:
            message['reasoning_conetent']=typo_reasoning
        if index in tool_calls:
            message['tool_calls']=[tool_calls[index][key] for key in sorted(tool_calls[index])]
        assembled.append({'index':index,'message':message,'finish_reason':state.get('finish_reason')})
    result['choices']=assembled
    result['stream']=True
    return result


def _response_aborted(data):
    choices=(data or {}).get('choices') or []
    if not choices:
        return True
    first=choices[0] if isinstance(choices[0], dict) else {}
    return first.get('finish_reason')=='abort'


def _enable_tcp_keepalive(sock):
    """Keep the socket active so a proxy idle timeout (matrixllm ~90s) never
    fires while the upstream model is thinking."""
    try:
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
        for opt, val in (("TCP_KEEPIDLE", 20), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 9)):
            if hasattr(_socket, opt):
                sock.setsockopt(_socket.IPPROTO_TCP, getattr(_socket, opt), val)
    except OSError:
        pass


class _KAHTTPConnection(_http_client.HTTPConnection):
    def connect(self):
        super().connect(); _enable_tcp_keepalive(self.sock)


class _KAHTTPSConnection(_http_client.HTTPSConnection):
    def connect(self):
        super().connect(); _enable_tcp_keepalive(self.sock)


class _KAHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_KAHTTPConnection, req)


class _KAHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_KAHTTPSConnection, req, context=self._context)


_SHIM_OPENER = urllib.request.build_opener(_KAHTTPHandler(), _KAHTTPSHandler())


def _api_format():
    return (os.getenv('CHEMBENCH_API_FORMAT') or 'openai').strip().lower()


def _responses_body(body):
    """OpenAI chat body -> /v1/responses body (stateless)."""
    msgs = body.get("messages") or []
    instructions = "\n\n".join(m.get("content") or "" for m in msgs if m.get("role") == "system")
    items = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": m.get("tool_call_id") or m.get("id"),
                          "output": str(m.get("content") or "")})
            continue
        if role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                items.append({"type": "function_call", "call_id": tc.get("id"),
                              "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"})
            continue
        items.append({"role": role, "content": m.get("content") or ""})
    out = {"model": body.get("model"), "input": items, "store": False}
    if instructions:
        out["instructions"] = instructions
    tools = []
    for t in (body.get("tools") or []):
        fn = t.get("function") or {}
        if fn.get("name"):
            tools.append({"type": "function", "name": fn["name"],
                          "description": fn.get("description") or "",
                          "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    if tools:
        out["tools"] = tools
        out["tool_choice"] = body.get("tool_choice") or "auto"
    eff = body.get("reasoning_effort")
    if eff:
        out["reasoning"] = {"effort": eff}
    mt = body.get("max_completion_tokens") or body.get("max_tokens")
    if mt:
        out["max_output_tokens"] = int(mt)
    if body.get("temperature") is not None:
        out["temperature"] = float(body["temperature"])
    if body.get("top_p") is not None:
        out["top_p"] = float(body["top_p"])
    return out


def _responses_to_chat(d):
    """Responses output -> OpenAI-shaped dict (call_id mapped to tool_calls[].id)."""
    text_parts, tool_calls, reasoning = [], [], []
    for o in (d.get("output") or []):
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t == "message":
            for c in (o.get("content") or []):
                if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    text_parts.append(c["text"])
        elif t == "function_call":
            tool_calls.append({"id": o.get("call_id") or o.get("id"), "type": "function",
                               "function": {"name": o.get("name"),
                                            "arguments": o.get("arguments") or "{}"}})
        elif t == "reasoning":
            for s in (o.get("summary") or []):
                if isinstance(s, dict) and s.get("text"):
                    reasoning.append(s["text"])
    msg = {"role": "assistant", "content": "".join(text_parts)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"model": d.get("model"), "usage": d.get("usage"),
            "choices": [{"message": msg,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def _anthropic_body(body):
    """OpenAI chat body -> Anthropic /v1/messages body."""
    msgs = body.get("messages") or []
    system = "\n\n".join(m.get("content") or "" for m in msgs if m.get("role") == "system")
    convo = [{"role": m.get("role"), "content": m.get("content") or ""}
             for m in msgs if m.get("role") != "system"]
    out = {"model": body.get("model"), "messages": convo,
           "max_tokens": int(body.get("max_tokens") or 8192)}
    if system:
        out["system"] = system
    if body.get("temperature") is not None:
        out["temperature"] = float(body["temperature"])
    if body.get("stream"):
        out["stream"] = True
    for k in ("thinking", "output_config"):
        if body.get(k) is not None:
            out[k] = body[k]
    return out


def _anthropic_to_chat(d):
    """Anthropic response -> OpenAI-shaped dict."""
    blocks = d.get("content") or []
    text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    reasoning = "".join(b.get("thinking") or "" for b in blocks if b.get("type") == "thinking")
    return {"model": d.get("model"), "usage": d.get("usage"),
            "choices": [{"message": {"role": "assistant", "content": text,
                                     "reasoning_content": reasoning},
                         "finish_reason": d.get("stop_reason") or "stop"}]}


def _sse_read_anthropic(response, requested_model):
    content, reasoning, finish = [], [], None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "content_block_delta":
            dl = ev.get("delta") or {}
            if dl.get("type") == "text_delta" and isinstance(dl.get("text"), str):
                content.append(dl["text"])
            elif dl.get("type") == "thinking_delta" and isinstance(dl.get("thinking"), str):
                reasoning.append(dl["thinking"])
        elif t == "message_delta":
            finish = (ev.get("delta") or {}).get("stop_reason") or finish
        elif t == "message_stop":
            break
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    return {"model": requested_model, "choices": [{"message": msg, "finish_reason": finish or "stop"}]}


class ChemBenchAgent(BaseAgent):
    @staticmethod
    def name(): return "chembench-agent"
    @staticmethod
    def version(): return "0.2.0"

    async def setup(self, environment):
        return None

    def _endpoint(self):
        if _api_format() == 'responses':
            _b=(os.getenv('CHEMBENCH_API_BASE') or '').rstrip('/')
            return _b + '/responses'
        if _api_format() == 'anthropic':
            base=(os.getenv('CHEMBENCH_ANTHROPIC_BASE') or os.getenv('CHEMBENCH_API_BASE') or '').rstrip('/')
            return base + '/v1/messages'
        full=os.getenv('CHEMBENCH_API_URL')
        if full: return full
        base=os.getenv('CHEMBENCH_API_BASE') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
        return base.rstrip('/')+'/chat/completions'

    def _endpoint_reachable(self, attempts=3, connect_timeout=8.0):
        parsed=_urlparse.urlsplit(self._endpoint())
        host=parsed.hostname
        if not host:
            return True
        port=parsed.port or (443 if parsed.scheme=='https' else 80)
        for index in range(attempts):
            try:
                with _socket.create_connection((host,port),timeout=connect_timeout):
                    return True
            except OSError:
                if index+1<attempts:
                    time.sleep(min(15.0,2.0**index))
        return False

    def _request_json(self, body, timeout):
        last_error=None
        attempts=max(1,int(os.getenv('CHEMBENCH_REQUEST_RETRIES','5')))
        for attempt in range(attempts):
            body_without_key={k:v for k,v in body.items() if k!='__api_key'}
            body_without_key['stream']=True
            request_id=str(uuid.uuid4())
            started=time.time()
            if not self._endpoint_reachable():
                last_error=RuntimeError('LLM endpoint TCP connect failed after retries')
                if attempt+1<attempts:
                    time.sleep(min(30.0,2.0**(attempt+1)))
                    continue
                raise last_error
            _fmt=_api_format(); _anthropic=_fmt=='anthropic'; _responses=_fmt=='responses'
            if _responses:
                _send=_responses_body(body_without_key); _send['stream']=False
                _hdrs={'Content-Type':'application/json','Authorization':'Bearer '+body.get('__api_key','')}
            else:
                _send=_anthropic_body(body_without_key) if _anthropic else body_without_key
                _hdrs={'Content-Type':'application/json','Accept':'text/event-stream','Authorization':'Bearer '+body.get('__api_key','')}
            if _anthropic:
                _hdrs['anthropic-version']='2023-06-01'
            req=urllib.request.Request(self._endpoint(),data=json.dumps(_send).encode(),headers=_hdrs,method='POST')
            try:
                with _SHIM_OPENER.open(req,timeout=timeout) as r:
                    if _responses:
                        data=_responses_to_chat(json.loads(r.read().decode()))
                    elif _anthropic:
                        _ct=(r.headers.get('Content-Type') or '')
                        data=(_sse_read_anthropic(r, body_without_key.get('model'))
                              if 'text/event-stream' in _ct else _anthropic_to_chat(json.loads(r.read().decode())))
                    else:
                        data=_read_chat_completion(r)
                    if _response_aborted(data) and attempt+1<attempts:
                        self._record_provider_call({
                            'request_id':request_id,'attempt':attempt+1,'started_unix':started,
                            'elapsed_seconds':round(time.time()-started,6),'endpoint':self._endpoint(),
                            'request':body_without_key,'error':'PROVIDER_ABORT_RESPONSE',
                        })
                        last_error=RuntimeError('PROVIDER_ABORT_RESPONSE')
                        time.sleep(min(60.0,2.0**(attempt+1)))
                        continue
                    self._record_provider_call({
                        'request_id':request_id,'attempt':attempt+1,'started_unix':started,
                        'elapsed_seconds':round(time.time()-started,6),'endpoint':self._endpoint(),
                        'request':body_without_key,'http_status':getattr(r,'status',200),
                        'response':data,
                    })
                    return data
            except urllib.error.HTTPError as exc:
                try:
                    detail=exc.read().decode('utf-8','replace')
                except Exception:
                    detail=''
                # SiliconFlow can transiently return 429/5xx for long tool
                # requests.  Treat those like transport failures so the
                # configured retry budget is actually honoured; permanent 4xx
                # errors still fail immediately with the provider detail.
                self._record_provider_call({
                    'request_id':request_id,'attempt':attempt+1,'started_unix':started,
                    'elapsed_seconds':round(time.time()-started,6),'endpoint':self._endpoint(),
                    'request':body_without_key,'http_status':exc.code,'error':detail,
                })
                # matrixllm 会把长请求随机路由到不接受 max_tokens 的上游（瞬时 400）
                transient_400 = exc.code == 400 and 'unsupported_parameter' in detail
                if (exc.code in {406,408,429,500,502,503,504} or transient_400) and attempt+1<attempts:
                    last_error=RuntimeError(f'LLM HTTP {exc.code}: {detail}')
                    try:
                        retry_after=float(exc.headers.get('Retry-After','0'))
                    except (TypeError,ValueError):
                        retry_after=0.0
                    if exc.code in (429,406):
                        # SiliconFlow's TPM window can outlive the ordinary
                        # transport retry backoff; rapid retries only create
                        # more rejected billable requests.
                        time.sleep(min(120.0, max(retry_after, 65.0)))
                    else:
                        time.sleep(min(60.0, max(retry_after, 2.0**(attempt+1))))
                    continue
                raise RuntimeError(f'LLM HTTP {exc.code}: {detail}') from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last_error=exc
                self._record_provider_call({
                    'request_id':request_id,'attempt':attempt+1,'started_unix':started,
                    'elapsed_seconds':round(time.time()-started,6),'endpoint':self._endpoint(),
                    'request':body_without_key,'error':f'{type(exc).__name__}: {exc}',
                })
                if attempt+1<attempts:
                    time.sleep(min(60.0,2.0**(attempt+1)))
            except Exception as exc:
                self._record_provider_call({
                    'request_id':request_id,'attempt':attempt+1,'started_unix':started,
                    'elapsed_seconds':round(time.time()-started,6),'endpoint':self._endpoint(),
                    'request':body_without_key,'error':f'{type(exc).__name__}: {exc}',
                })
                raise
        raise last_error

    def _summarize_context(self, arg):
        """LLM digest call for context compaction (auxiliary; not an agent turn)."""
        messages = arg if isinstance(arg, list) else [{"role": "user", "content": str(arg)}]
        model = os.getenv('CHEMBENCH_MODEL_ID') or getattr(self, 'model_name', None)
        key = os.getenv('CHEMBENCH_API_KEY') or os.getenv('OPENAI_API_KEY', '')
        body = {'model': model, 'messages': messages, 'temperature': 0.0, 'max_tokens': 8192}
        top_p = os.getenv('CHEMBENCH_TOP_P')
        if top_p:
            body['top_p'] = float(top_p)
        thinking = os.getenv('CHEMBENCH_ENABLE_THINKING', '').strip().lower()
        if thinking in ('1', 'true', 'yes', 'on'):
            body['enable_thinking'] = True
        elif thinking in ('0', 'false', 'no', 'off'):
            body['enable_thinking'] = False
        body['reasoning_effort'] = 'low'
        body['__api_key'] = key
        data = self._request_json(body, float(os.getenv('CHEMBENCH_REQUEST_TIMEOUT', '300')))
        choice = (data.get('choices') or [{}])[0]
        if choice.get('finish_reason') not in (None, 'stop'):
            raise RuntimeError('summary finish_reason=%s' % choice.get('finish_reason'))
        content = (choice.get('message') or {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('summary empty')
        return content

    @staticmethod
    def _record_provider_call(record):
        """Append the complete redacted provider exchange without mutating history."""
        target=os.getenv('CHEMBENCH_LLM_CALL_LOG')
        if not target:
            return
        path=Path(target)
        path.parent.mkdir(parents=True,exist_ok=True)
        safe=json.loads(json.dumps(record,ensure_ascii=False,default=str))
        request=safe.get('request')
        if isinstance(request,dict):
            request.pop('__api_key',None)
        with path.open('a',encoding='utf-8') as fh:
            fh.write(json.dumps(safe,ensure_ascii=False,separators=(',',':'))+'\n')

    def _call(self,messages,temperature,max_tokens,json_schema=None):
        model=os.getenv('CHEMBENCH_MODEL_ID') or getattr(self,'model_name',None)
        key=os.getenv('CHEMBENCH_API_KEY') or os.getenv('OPENAI_API_KEY','')
        body={'model':model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}
        if os.getenv('CHEMBENCH_TOP_P'): body['top_p']=float(os.getenv('CHEMBENCH_TOP_P'))
        _thinking=os.getenv('CHEMBENCH_ENABLE_THINKING','').strip().lower()
        if _thinking in ('1','true','yes','on'): body['enable_thinking']=True
        elif _thinking in ('0','false','no','off'): body['enable_thinking']=False
        if os.getenv('CHEMBENCH_REASONING_EFFORT'): body['reasoning_effort']=os.getenv('CHEMBENCH_REASONING_EFFORT')
        if json_schema is not None:
            body['response_format']={'type':'json_schema','json_schema':{'name':'chembench_agent_output','schema':json_schema}}
        else:
            response_format=os.getenv('CHEMBENCH_RESPONSE_FORMAT')
            if response_format:
                body['response_format']={'type':response_format}
        body['__api_key']=key
        data=self._request_json(body,float(os.getenv('CHEMBENCH_REQUEST_TIMEOUT','1800')))
        self._last_finish_reason=((data.get('choices') or [{}])[0] or {}).get('finish_reason')
        _compactor=getattr(self,'_active_compactor',None)
        if _compactor is not None:
            _compactor.observe(data.get('usage') if isinstance(data,dict) else None)
        return data['choices'][0]['message']['content']

    def _call_tool_action(self,messages,tools,temperature,max_tokens):
        model=os.getenv('CHEMBENCH_MODEL_ID') or getattr(self,'model_name',None)
        key=os.getenv('CHEMBENCH_API_KEY') or os.getenv('OPENAI_API_KEY','')
        body={'model':model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens,'tools':tools,'tool_choice':'auto'}
        if os.getenv('CHEMBENCH_TOP_P'): body['top_p']=float(os.getenv('CHEMBENCH_TOP_P'))
        _thinking=os.getenv('CHEMBENCH_ENABLE_THINKING','').strip().lower()
        if _thinking in ('1','true','yes','on'): body['enable_thinking']=True
        elif _thinking in ('0','false','no','off'): body['enable_thinking']=False
        if os.getenv('CHEMBENCH_REASONING_EFFORT'): body['reasoning_effort']=os.getenv('CHEMBENCH_REASONING_EFFORT')
        body['__api_key']=key
        data=self._request_json(body,float(os.getenv('CHEMBENCH_REQUEST_TIMEOUT','1800')))
        self._last_finish_reason=((data.get('choices') or [{}])[0] or {}).get('finish_reason')
        _compactor=getattr(self,'_active_compactor',None)
        if _compactor is not None:
            _compactor.observe(data.get('usage') if isinstance(data,dict) else None)
        message=data['choices'][0]['message']
        tool_calls=message.get('tool_calls')
        if isinstance(tool_calls,list) and tool_calls:
            call=tool_calls[0]
            function=call.get('function') if isinstance(call,dict) else None
            if isinstance(function,dict):
                raw=function.get('arguments') or '{}'
                try:
                    arguments=json.loads(raw) if isinstance(raw,str) else raw
                except Exception:
                    arguments=None
                return function.get('name'),arguments,message.get('content') or '',message
        return None,None,message.get('content') or '',message

    @staticmethod
    def _compact_messages(messages, max_chars=100000, keep_reasoning_turns=2, keep_reasoning_chars=20000):
        """Keep the transcript inside the provider context budget.

        Per-turn reasoning and large tool blobs otherwise grow the prompt until
        the model spends its whole output budget on reasoning and returns a
        length-truncated response with no tool call.  Reasoning is kept only for
        the most recent turns, every payload is capped, and only the protocol
        head plus the recent turns that fit the budget are retained, preserving
        the assistant/tool-call pairing the provider requires.
        """
        with_reasoning=[i for i,msg in enumerate(messages)
                        if msg.get('role')=='assistant' and msg.get('reasoning_content')]
        keep=set(with_reasoning[-keep_reasoning_turns:]) if with_reasoning else set()
        compact=[]
        for index,msg in enumerate(messages):
            item=dict(msg)
            reasoning=item.get('reasoning_content')
            if index not in keep:
                item.pop('reasoning_content',None)
            elif isinstance(reasoning,str) and len(reasoning)>keep_reasoning_chars:
                item['reasoning_content']=reasoning[:keep_reasoning_chars]+'\n...[truncated for context budget]'
            content=item.get('content')
            if isinstance(content,str):
                limit=20000 if item.get('role')=='tool' else 12000
                if len(content)>limit:
                    item['content']=content[:limit]+'\n...[truncated for context budget]'
            compact.append(item)
        def payload_size(items):
            return sum(len(str(m.get('content','')))+len(str(m.get('reasoning_content',''))) for m in items)
        if payload_size(compact)<=max_chars:
            return compact
        head=compact[:2]
        budget=max_chars-payload_size(head)
        tail=[]
        used=0
        for index in range(len(compact)-1,1,-1):
            item=compact[index]
            length=payload_size([item])
            if tail and used+length>budget:
                break
            tail.insert(0,item)
            used+=length
        if tail and tail[0].get('role')=='tool':
            for index in range(len(compact)-len(tail)-1,1,-1):
                if compact[index].get('role')=='assistant' and compact[index].get('tool_calls'):
                    tail.insert(0,compact[index])
                    break
        retained=head+tail
        # OpenAI-compatible APIs require every role=tool message to be
        # immediately grounded in an earlier assistant message containing the
        # matching tool_calls.  Truncating by raw message count can otherwise
        # leave an orphaned tool response and DeepSeek rejects the next turn.
        while retained and retained[0].get('role')=='tool':
            retained.pop(0)
        pending_tool_ids=set()
        for idx,item in enumerate(list(retained)):
            if item.get('role')=='assistant' and item.get('tool_calls'):
                pending_tool_ids={
                    call.get('id') for call in item.get('tool_calls',[])
                    if isinstance(call,dict) and call.get('id')
                }
                continue
            if item.get('role')!='tool':
                continue
            call_id=item.get('tool_call_id')
            if call_id in pending_tool_ids:
                pending_tool_ids.discard(call_id)
                continue
            recovered=dict(item)
            recovered.pop('tool_call_id',None)
            recovered['role']='user'
            recovered['content']='Tool result (context recovery): '+str(item.get('content',''))
            retained[idx]=recovered
        return retained

    @staticmethod
    def _post_json(url, value):
        req=urllib.request.Request(url,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=float(os.getenv('CHEMBENCH_PROTOCOL_TIMEOUT','300'))) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail=exc.read().decode('utf-8','replace')
            except Exception:
                detail=''
            raise RuntimeError(f'sidecar HTTP {exc.code}: {detail}') from exc

    async def _fixed_read(self, environment, path, limit=30000):
        # Public input files are immutable for the lifetime of one agent run.
        # Cache only that namespace: protocol_state.json and round feedback are
        # deliberately excluded because task code may update them in-process.
        cache = getattr(self, "_immutable_read_cache", None)
        if cache is None:
            cache = self._immutable_read_cache = {}
        cache_key = (str(path), int(limit))
        if str(path).startswith("/workspace/input/") and cache_key in cache:
            return cache[cache_key]
        source=f"from pathlib import Path; print(Path({path!r}).read_text(encoding='utf-8')[-{int(limit)}:])"
        payload=base64.b64encode(source.encode()).decode()
        result=await environment.exec(command=f"python -c \"import base64; exec(base64.b64decode('{payload}'))\"",cwd='/workspace',timeout_sec=30)
        value = getattr(result,'stdout','')
        if str(path).startswith("/workspace/input/"):
            cache[cache_key] = value
        return value

    async def _write_workspace_file(self, environment, path, data):
        serialized=json.dumps(data,indent=2).encode()
        if hasattr(environment,'upload_file') and len(serialized)>2000:
            parent=Path(path).parent.as_posix()
            await environment.exec(command=f"mkdir -p {parent}",cwd='/workspace',timeout_sec=30)
            fd,temp=tempfile.mkstemp(prefix='chembench-agent-',suffix='.json')
            try:
                with os.fdopen(fd,'wb') as handle:
                    handle.write(serialized)
                await environment.upload_file(temp,path)
            finally:
                try: os.unlink(temp)
                except FileNotFoundError: pass
            return
        encoded=base64.b64encode(json.dumps(data,indent=2).encode()).decode()
        await environment.exec(command=f"python -c \"import base64; from pathlib import Path; Path('{path}').parent.mkdir(parents=True, exist_ok=True); open('{path}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)

    async def _import_sidecar_artifacts(self, environment, export_id, required_artifacts):
        """Import a sidecar evidence snapshot into /workspace.

        Sidecars and Harbor agents intentionally have distinct filesystems. The
        sidecar snapshots submissions under the organizer-only bind mount; this
        method stages every required model-authored artifact and atomically
        replaces each regular file only after the full source set was copied.
        It never copies hidden evaluator files into the agent workspace.
        """
        if not isinstance(export_id,str) or not export_id.strip():
            return {'ok':False,'reason':'missing_export_id'}
        safe=[]
        for item in required_artifacts or []:
            if not isinstance(item,str): continue
            path=Path(item)
            if path.is_absolute() or '..' in path.parts: continue
            # Public input is pre-mounted and must never be sidecar-derived.
            # `data` is different: Task 01's public contract requires it at
            # verifier time, but its controlled sidecar materializes that
            # public data in a separate filesystem. Import only explicitly
            # required data paths; no private evaluator material is exported.
            if path.parts and path.parts[0] == 'input': continue
            safe.append(path.as_posix())
        payload=base64.b64encode(json.dumps({'export_id':export_id,'artifacts':safe}).encode()).decode()
        script="""import base64,json,os,shutil,uuid
from pathlib import Path
spec=json.loads(base64.b64decode('__PAYLOAD__'))
root=Path('/workspace'); source=Path('/organizer_evidence')/spec['export_id']/ 'workspace'
stage=root/('.chembench-sidecar-import-'+uuid.uuid4().hex)
result={'ok':False,'export_id':spec['export_id'],'copied':[],'missing':[]}
try:
    if not source.is_dir(): raise FileNotFoundError(str(source))
    stage.mkdir(parents=True)
    for rel in spec['artifacts']:
        src=source/rel
        if not src.exists(): result['missing'].append(rel); continue
        dst=stage/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir(): shutil.copytree(src,dst,copy_function=shutil.copy2)
        else: shutil.copy2(src,dst)
    for rel in spec['artifacts']:
        staged=stage/rel
        if not staged.exists(): continue
        target=root/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if staged.is_file(): os.replace(staged,target)
        elif not target.exists(): os.replace(staged,target)
        else:
            # Directories cannot be atomically replaced over nonempty public
            # directories.  Their contents are still staged before commit and
            # each file replacement is atomic.
            for child in staged.rglob('*'):
                if child.is_dir(): continue
                rel_child=child.relative_to(staged); out=target/rel_child
                out.parent.mkdir(parents=True,exist_ok=True); os.replace(child,out)
        result['copied'].append(rel)
    result['ok']=True
finally:
    shutil.rmtree(stage,ignore_errors=True)
    (root/'sidecar_artifact_import.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
""".replace('__PAYLOAD__',payload)
        encoded=base64.b64encode(script.encode()).decode()
        try:
            response=await environment.exec(command=f"python -c \"import base64; exec(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=600)
            text=getattr(response,'stdout','') or ''
            return json.loads(text.strip().splitlines()[-1]) if text.strip() else {'ok':False,'reason':'empty_import_response'}
        except Exception as exc:
            return {'ok':False,'reason':f'{type(exc).__name__}: {exc}','export_id':export_id}

    async def _run_task15_round(self, instruction, environment, context, cfg):
        """Drive the legacy JSON/tool protocol without exposing an unrestricted shell to the model."""
        match=re.search(r'round\s+(\d+)',instruction,re.I)
        if not match: raise RuntimeError('Task 15 step instruction does not declare its round number')
        round_number=int(match.group(1))
        question=await self._fixed_read(environment,'/workspace/input/tasks/q1_lipophilic/question.md')
        state_path='/workspace/protocol_state.json'
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        messages=state.get('messages') or [
            {'role':'system','content':'Follow the supplied molecular-design protocol. Return exactly one JSON object per turn: either {"tool_calls":[...]} or {"candidate":{"name":"...","smiles":"...","rationale":"..."}}. You have no shell or arbitrary Python access.'},
            {'role':'user','content':question},
        ]
        feedback=''
        if round_number > 1:
            previous=f'/workspace/rounds/round-{round_number-1}/feedback.json'
            try: feedback=await self._fixed_read(environment,previous,12000)
            except Exception: feedback=''
        messages.append({'role':'user','content':instruction + ('\nPrevious-round public feedback:\n'+feedback if feedback else '')})
        calls_used=0; tool_messages=0; turns=[]; candidate=None
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0)))
        max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        for turn in range(1,4):
            text=self._call(messages,temperature,max_tokens); messages.append({'role':'assistant','content':text})
            try: obj=self._extract_json(text)
            except Exception:
                turns.append({'turn':turn,'kind':'parse_error'}); messages.append({'role':'user','content':'Invalid JSON. Submit one valid protocol object.'}); continue
            if isinstance(obj.get('candidate'),dict):
                value=obj['candidate']
                if all(isinstance(value.get(key),str) for key in ('name','smiles','rationale')):
                    candidate={key:value[key].strip() for key in ('name','smiles','rationale')}; turns.append({'turn':turn,'kind':'candidate'}); break
                messages.append({'role':'user','content':'Candidate requires string fields name, smiles, and rationale.'}); continue
            if 'tool_calls' in obj:
                tool_messages+=1
                if tool_messages > 2:
                    turns.append({'turn':turn,'kind':'tool_call_after_budget'}); break
                payload=base64.b64encode(json.dumps(obj['tool_calls']).encode()).decode()
                command=f"python /workspace/chembench_protocol/run_tools.py --payload {payload} --used {calls_used}"
                result=await environment.exec(command=command,cwd='/workspace',timeout_sec=120)
                try: bridge=json.loads(getattr(result,'stdout',''))
                except Exception: bridge={'responses':[{'ok':False,'error':{'code':'tool_runtime_error'}}],'calls_used':calls_used}
                calls_used=int(bridge.get('calls_used',calls_used)); responses=bridge.get('responses',[])
                turns.append({'turn':turn,'kind':'tool_calls','responses':responses})
                messages.append({'role':'user','content':json.dumps({'tool_responses':responses,'calls_used':calls_used,'calls_remaining':max(0,6-calls_used)})})
                continue
            messages.append({'role':'user','content':'Object must contain tool_calls or candidate.'})
        if candidate is not None:
            target=f'/workspace/rounds/round-{round_number}/candidate.json'
            encoded=base64.b64encode(json.dumps({'candidate':candidate},indent=2).encode()).decode()
            await environment.exec(command=f"python -c \"import base64; from pathlib import Path; Path('{target}').parent.mkdir(parents=True, exist_ok=True); open('{target}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        persisted={'messages':messages,'task15':{'last_round':round_number,'calls_used':calls_used,'tool_messages':tool_messages}}
        encoded=base64.b64encode(json.dumps(persisted).encode()).decode()
        await environment.exec(command=f"python -c \"import base64; open('{state_path}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        if hasattr(context,'metadata'): context.metadata={'round':round_number,'tool_calls':calls_used,'tool_messages':tool_messages,'candidate_submitted':candidate is not None}
        return None

    async def _run_candidate_only_round(self, instruction, environment, context, cfg):
        """One model response and one candidate: malformed output consumes the round."""
        match=re.search(r'round\s+(\d+)',instruction,re.I)
        if not match: raise RuntimeError('Step instruction does not declare its round number')
        round_number=int(match.group(1)); state_path='/workspace/protocol_state.json'
        question=await self._fixed_read(environment,cfg['question_path'])
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        messages=state.get('messages') or [
            {'role':'system','content':'Follow the supplied protocol. You have no shell or arbitrary Python access. Return exactly one JSON object: {"candidate":{"name":"...","smiles":"...","rationale":"..."}}.'},
            {'role':'user','content':question},
        ]
        feedback=''
        if round_number > 1:
            try: feedback=await self._fixed_read(environment,f'/workspace/rounds/round-{round_number-1}/feedback.json',12000)
            except Exception: feedback=''
        messages.append({'role':'user','content':instruction + ('\nPrevious-round public feedback:\n'+feedback if feedback else '')})
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0)))
        max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        text=self._call(messages,temperature,max_tokens); messages.append({'role':'assistant','content':text})
        candidate=None; failure=None
        try:
            obj=self._extract_json(text); value=obj.get('candidate')
            if not isinstance(value,dict) or not all(isinstance(value.get(key),str) for key in ('name','smiles','rationale')): raise ValueError('invalid candidate schema')
            candidate={key:value[key].strip() for key in ('name','smiles','rationale')}
        except Exception as exc: failure=type(exc).__name__
        if candidate is not None:
            target=f'/workspace/rounds/round-{round_number}/candidate.json'
            encoded=base64.b64encode(json.dumps({'candidate':candidate},indent=2).encode()).decode()
            await environment.exec(command=f"python -c \"import base64; open('{target}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        persisted={'messages':messages,'candidate_only':{'last_round':round_number,'candidate_submitted':candidate is not None,'failure':failure}}
        encoded=base64.b64encode(json.dumps(persisted).encode()).decode()
        await environment.exec(command=f"python -c \"import base64; open('{state_path}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        if hasattr(context,'metadata'): context.metadata={'round':round_number,'candidate_submitted':candidate is not None,'failure':failure}
        return None

    async def _run_task03(self, instruction, environment, context, cfg):
        """Drive Task 03's five structured candidates and separate final recommendation."""
        state_path='/workspace/protocol_state.json'; question=await self._fixed_read(environment,cfg['question_path'])
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        messages=state.get('messages') or [
            {'role':'system','content':'Follow the xTB-DIPRO design protocol. You have no shell or arbitrary Python access. Return exactly the requested JSON object.'},
            {'role':'user','content':question},
        ]
        is_recommendation='recommended_candidate_id' in instruction
        match=re.search(r'R([1-5])',instruction) if not is_recommendation else None
        if not is_recommendation and not match: raise RuntimeError('Task 03 round instruction lacks R1-R5')
        feedback=''
        if is_recommendation:
            try: feedback=await self._fixed_read(environment,'/workspace/rounds/public_protocol_state.json',30000)
            except Exception: feedback=''
        elif int(match.group(1)) > 1:
            previous=int(match.group(1))-1
            try: feedback=await self._fixed_read(environment,f'/workspace/rounds/round-{previous}/feedback.json',16000)
            except Exception: feedback=''
        messages.append({'role':'user','content':instruction + ('\nEvaluator evidence:\n'+feedback if feedback else '')})
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0))); max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        text=self._call(messages,temperature,max_tokens); messages.append({'role':'assistant','content':text})
        failure=None; output=None
        try:
            obj=self._extract_json(text)
            if is_recommendation:
                value=obj.get('recommended_candidate_id')
                if value not in {'R1','R2','R3','R4','R5'}: raise ValueError('invalid recommendation')
                output={'recommended_candidate_id':value}; target='/workspace/final_answer.json'
            else:
                value=obj.get('candidate'); required=('candidate_id','parent_candidate_id','proposal_mode','smiles','structural_modification','modification_reasoning')
                if not isinstance(value,dict) or any(key not in value for key in required): raise ValueError('invalid candidate schema')
                expected='R'+match.group(1)
                if value.get('candidate_id') != expected or value.get('proposal_mode') not in {'direct_smiles','rdkit_edit_plan'} or not isinstance(value.get('smiles'),str) or not value['smiles'].strip(): raise ValueError('invalid candidate values')
                output={'candidate':{key:value[key] for key in required}}; target=f'/workspace/rounds/round-{match.group(1)}/candidate.json'
            await self._write_workspace_file(environment,target,output)
        except Exception as exc: failure=type(exc).__name__
        persisted={'messages':messages,'task03':{'last_action':'recommendation' if is_recommendation else 'R'+match.group(1),'written':output is not None and failure is None,'failure':failure}}
        await self._write_workspace_file(environment,state_path,persisted)
        if hasattr(context,'metadata'): context.metadata=persisted['task03']
        return None

    async def _run_private_response_round(self,instruction,environment,context,cfg):
        """Five-round restricted protocol whose measurements come from an external private sidecar."""
        match=re.search(r'round\s+(\d+)',instruction,re.I)
        if not match: raise RuntimeError('private-response step lacks round number')
        round_number=int(match.group(1)); state_path='/workspace/protocol_state.json'; unit=cfg['candidate_unit']
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        sidecar=state.get('private_response',{}); url=os.getenv(cfg.get('sidecar_url_env','CHEMBENCH_PROTOCOL_URL'),cfg.get('sidecar_url','http://127.0.0.1:8787')).rstrip('/')
        if not sidecar.get('session_id'):
            started=self._post_json(url+'/session',{}); sidecar={'session_id':started['session_id'],'baseline':started['baseline_measurement']}
        question=await self._fixed_read(environment,cfg['question_path'],30000)
        public_cfg=await self._fixed_read(environment,cfg['public_config_path'],30000)
        messages=state.get('messages') or [
            {'role':'system','content':'Follow the supplied five-round protocol. You have no shell or arbitrary Python access. Return exactly one JSON object: tool_calls or candidate.'},
            {'role':'user','content':question+'\nPublic task configuration:\n'+public_cfg+'\nBaseline measurement:\n'+json.dumps(sidecar['baseline'])},
        ]
        if round_number>1:
            try: feedback=await self._fixed_read(environment,f'/workspace/rounds/round-{round_number-1}/feedback.json',18000)
            except Exception: feedback=''
            messages.append({'role':'user','content':instruction+'\nPrevious evaluator feedback:\n'+feedback})
        else: messages.append({'role':'user','content':instruction})
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0))); max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        tool_turns=0; calls_used=0; candidate=None; failure=None
        for turn_index in range(3):
            text=self._call(messages,temperature,max_tokens); messages.append({'role':'assistant','content':text})
            try: obj=self._extract_json(text)
            except Exception: failure='json_parse'; break
            value=obj.get('candidate')
            if isinstance(value,dict):
                if not isinstance(value.get('name'),str) or not isinstance(value.get('rationale'),str) or not isinstance(value.get(unit),dict): failure='invalid_candidate_schema'; break
                candidate={'name':value['name'].strip(),'rationale':value['rationale'].strip(),unit:value[unit]}; break
            if 'tool_calls' not in obj or tool_turns>=2: failure='tool_phase_exhausted'; break
            payload=base64.b64encode(json.dumps({'calls':obj['tool_calls'],'used':calls_used}).encode()).decode()
            result=await environment.exec(command=f"python /workspace/chembench_protocol/run_public_tools.py --payload {payload}",cwd='/workspace',timeout_sec=120)
            try: batch=json.loads(getattr(result,'stdout','')); calls_used=int(batch.get('calls_used',calls_used)); responses=batch.get('results',[])
            except Exception: responses=[{'ok':False,'error':'tool_runtime_error'}]
            tool_turns+=1; messages.append({'role':'user','content':json.dumps({'tool_results':responses,'tool_turns_used':tool_turns,'calls_used':calls_used})})
        feedback=None
        if candidate is not None:
            measured=self._post_json(url+'/evaluate',{'session_id':sidecar['session_id'],'candidate':candidate})
            if measured.get('ok'):
                feedback=measured['feedback']; target=f'/workspace/rounds/round-{round_number}'
                for filename,value in (('candidate.json',{'candidate':candidate}),('feedback.json',feedback)):
                    encoded=base64.b64encode(json.dumps(value,indent=2).encode()).decode(); await environment.exec(command=f"python -c \"import base64; from pathlib import Path; Path('{target}').mkdir(parents=True,exist_ok=True); open('{target}/{filename}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
            else: failure=(measured.get('error') or {}).get('code','sidecar_error')
        persisted={'messages':messages,'private_response':sidecar,'last_round':round_number,'last_failure':failure}
        encoded=base64.b64encode(json.dumps(persisted).encode()).decode(); await environment.exec(command=f"python -c \"import base64; open('{state_path}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        if hasattr(context,'metadata'): context.metadata={'round':round_number,'candidate_submitted':candidate is not None,'tool_turns':tool_turns,'tool_calls':calls_used,'failure':failure}
        return None

    @staticmethod
    def _hidden_suite_tool_spec(cfg):
        tools=[]
        for item in cfg.get('hidden_suite_tools') or []:
            if not isinstance(item,dict) or not item.get('name'):
                continue
            tools.append({
                'type':'function',
                'function':{
                    'name':str(item['name']),
                    'description':str(item.get('description','Execute the allowlisted benchmark tool.')),
                    'parameters':item.get('parameters',{'type':'object','properties':{},'additionalProperties':True}),
                },
            })
        required=list(cfg.get('candidate_required_fields') or ['final_status','rationale'])
        properties={field:{'type':'string'} for field in required if isinstance(field,str)}
        tools.append({
            'type':'function',
            'function':{
                'name':'submit_candidate',
                'description':'Submit the final candidate only after completing all required work and producing result.json.',
                'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False},
            },
        })
        return tools

    @staticmethod
    def _normalize_hidden_tool_calls(obj,tool_names):
        calls=obj.get('tool_calls') if isinstance(obj,dict) else None
        if not isinstance(calls,list):
            return []
        normalized=[]
        for call in calls:
            if not isinstance(call,dict):
                continue
            tool=call.get('tool')
            arguments=call.get('arguments')
            if not tool and len(call)==1:
                tool=next(iter(call)); arguments=call[tool]
            if tool in {'python','pyscf','xtb','psi4'}:
                tool='run_python'
            if tool not in tool_names:
                continue
            if not isinstance(arguments,dict):
                arguments={}
            if tool=='run_python':
                if 'script' not in arguments and 'code' in arguments:
                    arguments['script']=arguments.pop('code')
                if 'timeout_seconds' not in arguments:
                    arguments['timeout_seconds']=900
            normalized.append({'tool':tool,'arguments':arguments})
        return normalized

    async def _run_hidden_suite_case(self,instruction,environment,context,cfg):
        """Run one organizer-side hidden case using only declared suite tools."""
        case_number=resolve_case_number(instruction, int(suite.get('case_count', 0)))
        state_path='/workspace/protocol_state.json'
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        suite=state.get('hidden_suite',{}); url=os.getenv(cfg['sidecar_url_env'],cfg['sidecar_url']).rstrip('/')
        if not suite.get('session_id'):
            started=self._post_json(url+'/session',{}); suite={'session_id':started['session_id'],'case':started['case'],'case_count':started['case_count']}
        case=suite.get('case')
        if not isinstance(case,dict): raise RuntimeError('hidden-suite sidecar did not provide a case')
        required=cfg.get('candidate_required_fields',['final_status','rationale'])
        if not isinstance(required,list) or not required:
            raise RuntimeError('hidden-suite config must declare candidate_required_fields')
        # The public task statement is the actual problem specification.  The
        # hidden-case envelope contains only transport/budget metadata, so include
        # the statement explicitly while bounding its size for provider context.
        try:
            task_statement=await self._fixed_read(environment,'/workspace/input/source_question.md',limit=4000)
        except Exception:
            task_statement=''
        tools=self._hidden_suite_tool_spec(cfg)
        required_artifacts = cfg.get('required_artifacts') or []
        if not isinstance(required_artifacts, list):
            raise RuntimeError('hidden-suite config required_artifacts must be a list')
        tool_names={item['function']['name'] for item in tools}
        tool_names.discard('submit_candidate')
        tool_instructions=', '.join(sorted(tool_names))
        native_override=os.getenv('CHEMBENCH_USE_NATIVE_TOOL_CALLS')
        if native_override is None:
            use_native_tools=bool(cfg.get('use_native_tool_calls',True))
        else:
            use_native_tools=native_override.strip().lower() in {'1','true','yes','on'}
        system=(
            'You are an expert scientific agent operating through a restricted sidecar. '
            'Use only the provided tools. No shell, network, arbitrary Python, or undeclared tool is available. '
            'Before the first computation, read input/TOOL_POLICY.md and input/submission_contract.md with read_file '
            '(use run_python only if a computation is required). '
            f'Allowed tools: {tool_instructions}. Prefer read_file for file inspection; reserve run_python for writes and computation. '
            'After all required artifacts are complete, call submit_candidate with exactly '
            + json.dumps(required,separators=(',',':')) + '.'
            ' Every artifact must be written inside the workspace at the exact relative path listed below; '
            'The tool process current working directory is the workspace root. Use relative paths such as '
            'input/... and result.json; do not use the literal absolute path /workspace (it is not mounted '
            'inside the restricted tool process). '
            'a statement in rationale is not an artifact. Before submitting, use run_python to check that every '
            'required path exists and is readable. Required artifact paths: '
            + json.dumps(required_artifacts,separators=(',',':'))
            + '. If an artifact cannot be produced, do not claim completion: report the exact missing path and reason.'
        )
        if len(required_artifacts) >= 6:
            system += (
                ' This is a large artifact task: batch read-only inspection into as few calls as possible. '
                'Start writing scoreable artifacts early (create the generator and training metadata before '
                'model-selection experiments), reserve the final calls for predictions, candidate library, '
                'result validation, and submit_candidate. Do not spend the whole budget comparing optional '
                'models; a documented, deterministic baseline that satisfies every schema is preferable to an '
                'unfinished optimization study.'
            )
        if not use_native_tools:
            system += (
                ' Native function calling is disabled for this provider. For every computation turn, '
                'respond with exactly one JSON object and no XML, Markdown, prose, or tool tags: '
                '{"tool_calls":[{"tool":"run_python","arguments":{"script":"...",'
                '"timeout_seconds":900}}]}. Other declared computation tools use the same tool/arguments '
                'shape. Only after every required artifact exists, finish with exactly '
                '{"candidate":{"final_status":"completed","rationale":"..."}}.'
            )
        user_content=instruction+'\nProtected case specification:\n'+json.dumps(case)
        if task_statement:
            user_content+='\nPublic task statement excerpt (the complete statement is available at /workspace/input/source_question.md; read it with run_python before solving):\n'+task_statement
        messages=[{'role':'system','content':system},{'role':'user','content':user_content}]
        trace=[]; candidate=None; failure=None; result_json_seen=False
        budget=case.get('budget',{}) or {}
        raw_turns=budget.get('max_agent_turns')
        raw_calls=budget.get('max_tool_calls')
        max_turns=int(raw_turns) if raw_turns not in (None,'unlimited') else None
        # 2026-09-23 (P1-3): env may only RAISE the turn ceiling -- thinking
        # profiles need room to finish (task11/12 died exploring under 6);
        # non-thinking runs keep task-declared budgets untouched.
        _forced_turns=os.getenv('CHEMBENCH_MAX_AGENT_TURNS')
        if _forced_turns and str(_forced_turns).strip().isdigit():
            _forced=int(_forced_turns)
            max_turns=_forced if max_turns is None else max(max_turns,_forced)
        max_tool_calls=int(raw_calls) if raw_calls not in (None,'unlimited') else None
        max_wall_time_seconds=int(budget.get('max_wall_time_seconds') or 21600)
        primary_action_limit=budget.get('primary_action_limit')
        repair_action_limit=budget.get('repair_action_limit')
        read_file_call_limit=budget.get('read_file_call_limit')
        primary_action_limit=int(primary_action_limit) if isinstance(primary_action_limit,int) else None
        repair_action_limit=int(repair_action_limit) if isinstance(repair_action_limit,int) else None
        read_file_call_limit=int(read_file_call_limit) if isinstance(read_file_call_limit,int) else None

        def _compute_used():
            return sum(1 for t in trace if (t.get('request') or {}).get('tool') != 'read_file')

        def _reads_used():
            return sum(1 for t in trace if (t.get('request') or {}).get('tool') == 'read_file')

        def _budget_line():
            if primary_action_limit is not None:
                line='Budget: compute actions remaining %d/%d' % (max(0, primary_action_limit - _compute_used()), primary_action_limit)
                if read_file_call_limit is not None:
                    line += '; read_file remaining %d/%d' % (max(0, read_file_call_limit - _reads_used()), read_file_call_limit)
                if repair_action_limit is not None:
                    line += '; output-only repairs %d (no training/generation allowed)' % repair_action_limit
            elif max_tool_calls is not None:
                line='Budget: tool calls remaining %d/%d' % (max(0, max_tool_calls - len(trace)), max_tool_calls)
            else:
                line='Budget: unlimited tool calls'
            arts=cfg.get('required_artifacts') or []
            if arts:
                shown=', '.join(str(a) for a in arts[:10])
                line += '; required artifacts: ' + shown + ('…' if len(arts)>10 else '')
            if max_turns is not None:
                line += '; turns remaining %d/%d' % (max(0, max_turns - turn_index), max_turns)
            if _in_submit_reserve():
                line += '; ACTION REQUIRED: call submit_candidate now with your best candidate'
            return line

        # Submission guarantee: don't lose an otherwise-complete run to exploration.
        # T1 warns once near the budget end; T2 reserves the final turns for submit.
        def _env_num(name, default, cast):
            raw=os.getenv(name)
            if raw is None or str(raw).strip()=='':
                return default
            try:
                return cast(raw)
            except (TypeError,ValueError):
                return default
        submit_reserve_turns=_env_num('CHEMBENCH_SUBMIT_RESERVE_TURNS',2,int)
        status_card_fraction=_env_num('CHEMBENCH_STATUS_CARD_FRACTION',0.75,float)
        status_card_sent=False

        def _in_submit_reserve():
            return (submit_reserve_turns>0 and max_turns is not None
                    and turn_index>max_turns-submit_reserve_turns)

        def _status_card():
            arts=cfg.get('required_artifacts') or []
            text=('[HARNESS STATUS] Turn %d/%s. Budget is nearly exhausted.'
                  % (turn_index, max_turns if max_turns is not None else 'unlimited'))
            if arts:
                text+=' Required artifacts: '+', '.join(str(a) for a in arts)+'.'
            text+=' If your required artifacts are already written, call submit_candidate NOW with your best candidate; work that is not submitted is lost.'
            return text

        run_deadline=time.monotonic()+max_wall_time_seconds
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0))); max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        turn_index=0
        consecutive_truncations=0
        consecutive_invalid=0
        consecutive_sidecar_failures = 0
        abort = False
        compactor=ContextCompactor(summarize=self._summarize_context)
        self._active_compactor=compactor
        while max_turns is None or turn_index < max_turns:
            if time.monotonic() >= run_deadline:
                failure='wall_time_exhausted'
                break
            turn_index += 1
            if (not status_card_sent and status_card_fraction>0 and max_turns is not None
                    and turn_index>=int(max_turns*status_card_fraction)):
                messages.append({'role':'user','content':_status_card()})
                status_card_sent=True
            wire=compactor.project(messages)+[{'role':'system','content':_budget_line()}]
            obj=None; raw_message=None; native_tool=None; native_arguments=None
            if use_native_tools:
                native_tool,native_arguments,text,raw_message=self._call_tool_action(wire,tools,temperature,max_tokens)
                if isinstance(raw_message,dict):
                    # Some OpenAI-compatible providers occasionally emit an
                    # unterminated JSON string in function.arguments.  Sending
                    # that assistant message back verbatim makes the provider
                    # reject the *next* request with HTTP 400.  Keep the tool
                    # call identity but replace unparsable arguments with an
                    # empty object; the normal invalid-call feedback then lets
                    # the model retry using valid JSON.
                    sanitized=json.loads(json.dumps(raw_message))
                    for raw_call in sanitized.get('tool_calls',[]) or []:
                        function=raw_call.get('function') if isinstance(raw_call,dict) else None
                        if not isinstance(function,dict):
                            continue
                        raw_args=function.get('arguments')
                        if isinstance(raw_args,str):
                            try: json.loads(raw_args)
                            except (TypeError,ValueError): function['arguments']='{}'
                    if not sanitized.get('tool_calls'):
                        body=str(sanitized.get('content') or '')
                        sanitized['content']='[harness: non-action output discarded; preview: '+body[:120].replace('\n',' ')+']'
                    raw_message=sanitized
                    messages.append(raw_message)
                finish_reason=getattr(self,'_last_finish_reason',None)
                # A length-truncated response is safe to execute only when
                # the native tool call and its JSON arguments are complete.
                # Otherwise allow one recovery request; a second consecutive
                # truncation ends the protocol instead of replaying work.
                if finish_reason == 'length' and (native_tool is None or native_arguments is None):
                    consecutive_truncations += 1
                    if consecutive_truncations >= 3:
                        failure='length_truncation_exhausted'
                        break
                    truncated_ids=[c.get('id') for c in ((raw_message or {}).get('tool_calls') or []) if isinstance(c,dict) and c.get('id')]
                    if truncated_ids:
                        for truncated_id in truncated_ids:
                            messages.append({'role':'tool','tool_call_id':truncated_id,'content':json.dumps({'ok':False,'error':'OUTPUT_LIMIT_TRUNCATED','detail':'Previous response hit the output-token limit before this call completed. Re-issue with minimal reasoning and complete arguments.'})})
                    else:
                        messages.append({'role':'user','content':'Your previous response consumed the output-token budget before emitting a tool call and was discarded. Do NOT restate reasoning. Emit exactly ONE complete tool call now (or, if finished, a short final answer).'})
                    continue
                # A provider may label a complete native function call as
                # ``length`` because hidden reasoning reached the output
                # limit after the call was serialized.  That call is safe to
                # execute and must not consume the consecutive-truncation
                # recovery allowance.
                if finish_reason != 'length' or (native_tool is not None and native_arguments is not None):
                    consecutive_truncations=0
                raw_calls=(raw_message or {}).get('tool_calls') or []
                if raw_calls:
                    consecutive_invalid=0
                    for raw_call in raw_calls:
                        if not isinstance(raw_call,dict):
                            continue
                        function=raw_call.get('function') or {}
                        call_name=function.get('name')
                        raw_args=function.get('arguments') or '{}'
                        try:
                            call_arguments=json.loads(raw_args) if isinstance(raw_args,str) else raw_args
                        except Exception:
                            call_arguments=None
                        call_id=raw_call.get('id') or ''
                        if _in_submit_reserve() and call_name!='submit_candidate':
                            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':False,'tool':call_name,'error':'TOOL_BUDGET_RESERVED_FOR_SUBMIT','detail':'Budget is reserved for submission now. Call submit_candidate with your best candidate; other actions are disabled.'})})
                            continue
                        if call_name=='submit_candidate' and isinstance(call_arguments,dict):
                            if all(isinstance(call_arguments.get(field),str) and call_arguments[field].strip() for field in required):
                                candidate=call_arguments
                                # Every native tool call in an assistant message
                                # must receive a matching tool response before
                                # the next provider request.  Even though the
                                # sidecar submission is performed below, add a
                                # protocol acknowledgement here so providers
                                # (notably DeepSeek Flash) do not reject the
                                # conversation with an incomplete tool-call
                                # sequence.
                                messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':True,'tool':call_name,'status':'SUBMIT_CANDIDATE_ACCEPTED_FOR_SIDEcar'})})
                                continue
                            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':False,'tool':call_name,'error':'INVALID_CANDIDATE_SCHEMA_RETRY_WITH_REQUIRED_FIELDS'})})
                            continue
                        if call_name not in tool_names or not isinstance(call_arguments,dict):
                            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':False,'tool':call_name,'error':'INVALID_NATIVE_TOOL_CALL'})})
                            continue
                        # OpenAI-compatible providers occasionally emit a
                        # native call with omitted/empty arguments.  No tool
                        # action is executed in that case, so do not send it
                        # to the sidecar (which would consume a quota call and
                        # create an empty script).  Return protocol feedback
                        # and let the model retry with a real argument object.
                        if not call_arguments or (
                            call_name == 'run_python' and not str(call_arguments.get('script','')).strip()
                        ):
                            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':False,'tool':call_name,'error':'EMPTY_TOOL_ARGUMENTS_RETRY_WITH_NONEMPTY_ARGUMENTS'})})
                            continue
                        if max_tool_calls is not None and len(trace) >= max_tool_calls:
                            messages.append({'role':'tool','tool_call_id':call_id,'content':json.dumps({'ok':False,'tool':call_name,'error':'TOOL_BUDGET_EXHAUSTED'})})
                            continue
                        try:
                            response=self._post_json(url+'/tool',{'session_id':suite['session_id'],'tool':call_name,'arguments':call_arguments})
                        except Exception as exc:
                            consecutive_sidecar_failures += 1
                            response={'ok':False,'tool':call_name,'error':str(exc)}
                            if consecutive_sidecar_failures >= 3:
                                failure='sidecar_infrastructure_failure'
                                abort=True
                        else:
                            consecutive_sidecar_failures = 0
                        trace.append({'request':{'tool':call_name,'arguments':call_arguments},'response':response})
                        if response.get('result',{}).get('result_json_exists') is True:
                            result_json_seen=True
                        tool_message_content=json.dumps(response)+'\n'+_budget_line()
                        messages.append({'role':'tool','tool_call_id':call_id,'content':tool_message_content})
                        if abort:
                            break
                    if abort:
                        break
                    if candidate is not None:
                        break
                    continue
                if native_tool=='submit_candidate' and isinstance(native_arguments,dict):
                    if all(isinstance(native_arguments.get(field),str) and native_arguments[field].strip() for field in required):
                        candidate=native_arguments
                    else:
                        messages.append({'role':'user','content':f'submit_candidate requires non-empty string fields: {required}. Retry with all required fields.'})
                elif text:
                    try: obj=self._extract_json(text)
                    except Exception: obj=None
            else:
                text=self._call(messages,temperature,max_tokens); messages.append({'role':'assistant','content':text})
                try: obj=self._extract_json(text)
                except Exception: obj=None
            if obj is None and candidate is None and native_tool is None:
                consecutive_invalid += 1
                if consecutive_invalid >= 6:
                    failure='invalid_response_exhausted'
                    break
                if consecutive_invalid == 1:
                    nudge='Invalid response. Use a provided tool or submit_candidate.'
                elif consecutive_invalid == 2:
                    nudge='Your last %d responses were unusable. Output EXACTLY ONE complete tool call and nothing else.' % consecutive_invalid
                else:
                    nudge='Repeated unusable output (%d in a row). Stop reasoning; emit ONE minimal valid native tool call now, or call submit_candidate if the required artifacts are already written.' % consecutive_invalid
                messages.append({'role':'user','content':nudge})
                continue
            if candidate is not None:
                consecutive_invalid=0
                break
            value=obj.get('candidate') if isinstance(obj,dict) else None
            if isinstance(value,dict):
                if not all(isinstance(value.get(field),str) and value[field].strip() for field in required):
                    messages.append({'role':'user','content':f'Candidate must contain non-empty string fields: {required}.'})
                    continue
                candidate={field:value[field].strip() for field in required}
                break
            calls=self._normalize_hidden_tool_calls(obj,tool_names)
            if not calls:
                consecutive_invalid += 1
                if consecutive_invalid >= 6:
                    failure='invalid_response_exhausted'
                    break
                messages.append({'role':'user','content':('Return a provided tool call or submit_candidate. Output EXACTLY ONE complete tool call, nothing else.' if consecutive_invalid < 3 else 'Repeated unusable output (%d in a row). Emit ONE minimal valid native tool call, or call submit_candidate.' % consecutive_invalid)})
                continue
            consecutive_invalid=0
            if _in_submit_reserve():
                allowed=[c for c in calls if c.get('tool')=='submit_candidate']
                if not allowed:
                    messages.append({'role':'user','content':'Budget is reserved for submission. Call submit_candidate now with your best candidate; other actions are disabled.'})
                    continue
                calls=allowed
            results=[]
            for call in calls:
                try:
                    response=self._post_json(url+'/tool',{'session_id':suite['session_id'],'tool':call['tool'],'arguments':call['arguments']})
                except Exception as exc:
                    consecutive_sidecar_failures += 1
                    response={'ok':False,'tool':call['tool'],'error':str(exc)}
                    if consecutive_sidecar_failures >= 3:
                        failure='sidecar_infrastructure_failure'
                        abort=True
                else:
                    consecutive_sidecar_failures = 0
                results.append(response); trace.append({'request':call,'response':response})
                if response.get('result',{}).get('result_json_exists') is True:
                    result_json_seen=True
                if abort:
                    break
            if abort:
                break
            messages.append({'role':'user','content':'Tool results:\n'+json.dumps(results)+'\n'+_budget_line()})
        if candidate is None:
            failure=failure or 'protocol_turn_exhausted'
            target=f'/workspace/cases/case-{case_number}'
            artifact={'case_id':case.get('case_id'),'public_case':case,'candidate':None,'tool_count':len(trace),'submission':None,'failure':failure}
            await self._write_workspace_file(environment,f'{target}/case_result.json',artifact)
        submitted=None
        if candidate is not None:
            if not all(isinstance(candidate.get(field),str) and candidate[field].strip() for field in required):
                failure='invalid_candidate_schema'; candidate=None
        if candidate is not None:
            if cfg.get('require_result_json') and not result_json_seen:
                # Submission artifacts are exclusively model-authored.  Never
                # fabricate a degraded result.json on the model's behalf.
                failure = failure or 'result_json_missing'
            try:
                submitted=self._post_json(url+'/submit',{'session_id':suite['session_id'],'candidate':candidate})
            except Exception as exc:
                submitted={'ok':False,'error':str(exc)}
                failure='sidecar_infrastructure_failure'
            # /submit snapshots the restricted workspace before invoking the
            # task adapter.  Import the public/model-authored files on both
            # success and error paths; this prevents a transport failure from
            # turning a completed result.json into MISSING_RESULT_JSON.
            export_id=submitted.get('export_id') or submitted.get('evidence_id') or f"{suite['session_id']}/last_submission"
            submitted['artifact_import']=await self._import_sidecar_artifacts(
                environment, export_id, cfg.get('required_artifacts') or []
            )
            if not submitted.get('ok') and failure != 'sidecar_infrastructure_failure':
                failure='sidecar_submission_rejected'
            evidence_id=submitted.get('evidence_id')
            # Evidence is already bind-mounted for the organizer/verifier. Do
            # not copy the entire evidence workspace back into the agent
            # workspace: it duplicates large artifacts and can overwrite the
            # live protocol state or model-produced files.
            suite['case']=submitted.get('next_case'); suite['last_feedback']=submitted.get('feedback')
            target=f'/workspace/cases/case-{case_number}'
            artifact={'case_id':case.get('case_id'),'public_case':case,'candidate':candidate,'tool_count':len(trace),'submission':submitted}
            await self._write_workspace_file(environment,f'{target}/case_result.json',artifact)
        persisted={'hidden_suite':suite,'last_case':case_number,'last_failure':failure,'result_json_seen':result_json_seen}
        await self._write_workspace_file(environment,state_path,persisted)
        # Keep a redaction-free-of-secrets protocol trace for post-run diagnosis.
        # The trace contains model messages and tool request/response metadata,
        # but never the provider API key (that key is not part of `messages`).
        await self._write_workspace_file(
            environment,
            '/workspace/agent_trajectory.json',
            {
                'task_id': cfg.get('task_id', 'task'),
                'case_number': case_number,
                'messages': messages,
                'tool_events': trace,
                'candidate': candidate,
                'submission': submitted,
                'failure': failure,
            },
        )
        if hasattr(context,'metadata'): context.metadata={'case':case_number,'tool_calls':len(trace),'candidate_submitted':candidate is not None,'failure':failure,'result_json_seen':result_json_seen}
        return None

    async def _run_adaptive_hidden_suite_case(self,instruction,environment,context,cfg):
        """Run one organizer-side adaptive torsion case through two closed actions."""
        match=re.search(r'case\s+(\d+)',instruction,re.I)
        if not match: raise RuntimeError('adaptive-suite step lacks case number')
        case_number=int(match.group(1)); state_path='/workspace/protocol_state.json'
        try: state=json.loads(await self._fixed_read(environment,state_path))
        except Exception: state={}
        suite=state.get('adaptive_suite',{}); url=os.getenv(cfg['sidecar_url_env'],cfg['sidecar_url']).rstrip('/')
        if not suite.get('session_id'):
            started=self._post_json(url+'/session',{}); suite={'session_id':started['session_id'],'case':started['case'],'case_count':started['case_count']}
        case=suite.get('case')
        if not isinstance(case,dict): raise RuntimeError('adaptive sidecar did not provide a case')
        compact_case={'case_id':case.get('case_id'),'task':case.get('task'),'budget':case.get('budget'),'constraints':case.get('constraints'),'output':case.get('output')}
        messages=[{'role':'system','content':'Follow the closed adaptive torsion protocol. Return exactly one JSON object and nothing else. Example run_scan_batch: {"action":"run_scan_batch","angles_deg":[0,60,120,180,240,300],"finalize":false,"confidence":null,"rationale":"Establish six evenly separated sectors."}. Example inspect_failed_attempt: {"action":"inspect_failed_attempt","angle_deg":15}. You have no shell or arbitrary Python access.'},{'role':'user','content':instruction+'\nProtected case specification:\n'+json.dumps(compact_case)}]
        trace=[]; candidate=None; failure=None; max_turns=int(case.get('budget',{}).get('max_agent_turns',8))
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0))); max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        _nto=os.getenv('CHEMBENCH_USE_NATIVE_TOOL_CALLS')
        use_native_tools=(bool(cfg.get('use_native_tool_calls',True)) if _nto is None else _nto.strip().lower() in {'1','true','yes','on'})
        adaptive_schema={
            'type':'object',
            'additionalProperties':False,
            'properties':{
                'action':{'type':'string','enum':['run_scan_batch','inspect_failed_attempt']},
                'angles_deg':{'type':'array','items':{'type':'integer','minimum':0,'maximum':355,'multipleOf':5},'minItems':1,'maxItems':6,'uniqueItems':True},
                'finalize':{'type':'boolean'},
                'confidence':{'type':['string','null'],'enum':['low','medium','high',None]},
                'angle_deg':{'type':'integer','minimum':0,'maximum':355,'multipleOf':5},
                'rationale':{'type':'string','minLength':1,'maxLength':400},
            },
            'required':['action'],
        }
        adaptive_tools=[
            {'type':'function','function':{
                'name':'run_scan_batch',
                'description':'Submit one adaptive torsion scan batch.',
                'parameters':{
                    'type':'object',
                    'additionalProperties':False,
                    'properties':{
                        'action':{'type':'string','const':'run_scan_batch'},
                        'angles_deg':{'type':'array','items':{'type':'integer','minimum':0,'maximum':355,'multipleOf':5},'minItems':1,'maxItems':6,'uniqueItems':True},
                        'finalize':{'type':'boolean'},
                        'confidence':{'type':['string','null'],'enum':['low','medium','high',None]},
                        'rationale':{'type':'string','minLength':1,'maxLength':400},
                    },
                    'required':['action','angles_deg','finalize','confidence','rationale'],
                },
            }},
            {'type':'function','function':{
                'name':'inspect_failed_attempt',
                'description':'Inspect a failed native angle.',
                'parameters':{
                    'type':'object',
                    'additionalProperties':False,
                    'properties':{'action':{'type':'string','const':'inspect_failed_attempt'},'angle_deg':{'type':'integer','minimum':0,'maximum':355,'multipleOf':5}},
                    'required':['action','angle_deg'],
                },
            }},
        ]
        last_good_state={}
        for turn_index in range(1,max_turns+1):
            if use_native_tools:
                name,arguments,text,_ = self._call_tool_action(messages,adaptive_tools,temperature,max_tokens)
                if name in {'run_scan_batch','inspect_failed_attempt'} and isinstance(arguments,dict):
                    obj=arguments
                elif text:
                    try: obj=self._extract_json(text)
                    except Exception: obj=None
                else:
                    obj=None
            else:
                text=self._call(messages,temperature,max_tokens,adaptive_schema); messages.append({'role':'assistant','content':text})
                try: obj=self._extract_json(text)
                except Exception: obj=None
                obj=self._normalize_adaptive_action(obj)
            if obj is None:
                messages.append({'role':'user','content':'Invalid JSON. Return exactly one valid adaptive protocol action object.'})
                continue
            obj=self._coerce_adaptive_action(obj,turn_index,last_good_state)
            if obj.get('action') in {'run_scan_batch','inspect_failed_attempt'}:
                response=self._post_json(url+'/tool',{'session_id':suite['session_id'],'tool':obj.get('action'),'arguments':obj})
                trace.append({'request':obj,'response':response}); result=response.get('result') or {}
                if isinstance(result.get('state'),dict): last_good_state=result['state']
                messages.append({'role':'user','content':'Tool result:\n'+json.dumps(result)})
                if result.get('terminal',{}).get('provider_must_stop') is True:
                    if result.get('terminal',{}).get('accepted') is True:
                        candidate=obj
                    else:
                        failure=result.get('status','sidecar_terminal_rejected')
                    break
                continue
            messages.append({'role':'user','content':'Return exactly one action with action equal to run_scan_batch or inspect_failed_attempt.'})
        if candidate is None and failure is None:
            failure='protocol_turn_exhausted'
        submitted=None
        if candidate is not None:
            submitted=self._post_json(url+'/submit',{'session_id':suite['session_id'],'candidate':candidate})
            if not submitted.get('ok'): failure='sidecar_submission_rejected'
            suite['case']=submitted.get('next_case'); suite['last_feedback']=submitted.get('feedback')
            target=f'/workspace/cases/case-{case_number}'
            artifact={'case_id':case.get('case_id'),'public_case':case,'candidate':candidate,'tool_events':trace,'submission':submitted}
            encoded=base64.b64encode(json.dumps(artifact,indent=2).encode()).decode(); await environment.exec(command=f"python -c \"import base64; from pathlib import Path; Path('{target}').mkdir(parents=True,exist_ok=True); open('{target}/case_result.json','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        persisted={'adaptive_suite':suite,'last_case':case_number,'last_failure':failure}
        encoded=base64.b64encode(json.dumps(persisted).encode()).decode(); await environment.exec(command=f"python -c \"import base64; open('{state_path}','wb').write(base64.b64decode('{encoded}'))\"",cwd='/workspace',timeout_sec=30)
        if hasattr(context,'metadata'): context.metadata={'case':case_number,'tool_calls':len(trace),'candidate_submitted':candidate is not None,'failure':failure}
        return None

    @staticmethod
    def _normalize_adaptive_action(obj):
        if not isinstance(obj,dict): return obj
        if 'action' in obj: return obj
        if 'tool_calls' in obj:
            calls=obj.get('tool_calls')
            if isinstance(calls,list) and calls and isinstance(calls[0],dict):
                call=calls[0]
                function=call.get('function') if isinstance(call,dict) else None
                if isinstance(function,dict):
                    name=function.get('name')
                    args=function.get('arguments')
                    if isinstance(args,str):
                        try: args=json.loads(args)
                        except Exception: args=None
                    if name in {'run_scan_batch','inspect_failed_attempt'} and isinstance(args,dict):
                        return {'action':name,**args}
        for value in obj.values():
            if isinstance(value,dict) and any(key in value for key in ('angles_deg','angles')):
                action={'action':'run_scan_batch',**value}
                if 'angles' in action and 'angles_deg' not in action:
                    action['angles_deg']=action.pop('angles')
                return action
        return obj

    @staticmethod
    def _coerce_adaptive_action(obj,turn_index,last_result):
        if not isinstance(obj,dict) or 'action' not in obj:
            return obj
        action=obj['action']
        state=last_result if isinstance(last_result,dict) else None
        if action=='inspect_failed_attempt' and not (state or {}).get('failed_angles_deg'):
            action='run_scan_batch'; obj['action']='run_scan_batch'
        if action=='run_scan_batch':
            angles=obj.get('angles_deg',obj.get('angles'))
            if not isinstance(angles,list):
                angles=[]
            clean=[]
            for value in angles:
                if isinstance(value,bool) or not isinstance(value,int) or value<0 or value>=360 or value%5 or value in clean:
                    continue
                clean.append(value)
            sampled=set((state or {}).get('sampled_angles_deg',[]))
            clean=[value for value in clean if value not in sampled]
            phase=(state or {}).get('next_phase','phase0')
            needed=6 if phase=='phase0' else 3 if phase in {'phase1','phase2'} else 2
            fallback=[0,60,120,180,240,300] if phase=='phase0' else [30,150,270] if phase=='phase1' else [90,210,330] if phase=='phase2' else [90,210,330]
            for value in fallback:
                if len(clean)>=needed:
                    break
                if value not in clean:
                    clean.append(value)
            obj['angles_deg']=clean[:needed]
            if not isinstance(obj.get('finalize'),bool):
                obj['finalize']=phase in {'phase2','recovery'}
            if obj['finalize']:
                if obj.get('confidence') not in {'low','medium','high'}:
                    obj['confidence']='high'
            else:
                obj['confidence']=None
            rationale=obj.get('rationale')
            if not isinstance(rationale,str) or not 1<=len(rationale)<=400:
                obj['rationale']='Continue the adaptive torsion sampling protocol with legal five-degree grid angles.'
        elif action=='inspect_failed_attempt':
            angle=obj.get('angle_deg')
            if isinstance(angle,bool) or not isinstance(angle,int) or angle<0 or angle>=360 or angle%5:
                obj['angle_deg']=0
        return obj

    @staticmethod
    def _extract_json(text):
        t=text.strip()
        m=re.search(r'```(?:json)?\s*(\{.*?\})\s*```',t,re.S)
        if m: t=m.group(1)
        try: return json.loads(t)
        except Exception:
            a=t.find('{'); b=t.rfind('}')
            if a>=0 and b>a: return json.loads(t[a:b+1])
            raise

    def _no_payload(self, text):
        # 2026-09-27 fix: max-thinking models can spend the whole output budget on reasoning and return empty/truncated/pseudo-format text; such a turn has no usable payload
        if getattr(self,'_last_finish_reason',None)=='length':
            return True
        t=(text or '').strip()
        if not t:
            return True
        if t.startswith('```') and not re.search(r'```(?:json)?\s*(\{.*?\})\s*```',t,re.S):
            return True
        return '# ' in t[:40] and '{' not in t

    async def run(self, instruction, environment, context):
        # Cache scope is one agent invocation. A new Harbor step starts with a
        # fresh cache, so files written between rounds cannot become stale.
        self._immutable_read_cache = {}
        cfg_res=await environment.exec(command='cat /workspace/.chembench_agent.json',cwd='/workspace',timeout_sec=30)
        cfg=json.loads(cfg_res.stdout) if getattr(cfg_res,'return_code',0)==0 and cfg_res.stdout.strip() else {}
        if cfg.get('protocol_mode') == 'task15_diagnostic_round':
            return await self._run_task15_round(instruction,environment,context,cfg)
        if cfg.get('protocol_mode') == 'candidate_only_round':
            return await self._run_candidate_only_round(instruction,environment,context,cfg)
        if cfg.get('protocol_mode') == 'task03_design_round':
            return await self._run_task03(instruction,environment,context,cfg)
        if cfg.get('protocol_mode') == 'private_response_round':
            return await self._run_private_response_round(instruction,environment,context,cfg)
        if cfg.get('protocol_mode') == 'hidden_suite_case':
            return await self._run_hidden_suite_case(instruction,environment,context,cfg)
        if cfg.get('protocol_mode') == 'adaptive_hidden_suite_case':
            return await self._run_adaptive_hidden_suite_case(instruction,environment,context,cfg)
        _scale=max(1,int(os.getenv('CHEMBENCH_ITERATION_SCALE','1') or '1'))
        max_iter=int(cfg.get('max_iterations',6))*_scale; max_tools=int(cfg.get('max_tool_calls',12))*_scale; per_tool=float(cfg.get('timeout_per_tool_sec',600));
        temperature=float(os.getenv('CHEMBENCH_TEMPERATURE',cfg.get('temperature',0.0))); max_tokens=int(os.getenv('CHEMBENCH_MAX_TOKENS',cfg.get('max_tokens',36000)))
        system=("You are an expert scientific agent operating in /workspace.\nReturn exactly one JSON object with keys reasoning, status, commands, read_files, final_answer. commands[].command MUST be valid Python source, not shell. Use only files inside /workspace. Set status=complete with final_answer when finished.")
        state={}
        state_res=await environment.exec(command='cat /workspace/protocol_state.json',cwd='/workspace',timeout_sec=30)
        if getattr(state_res,'return_code',1)==0 and getattr(state_res,'stdout','').strip():
            try: state=json.loads(state_res.stdout)
            except Exception: state={}
        messages=state.get('messages') or [{'role':'system','content':system}]
        if not any(m.get('role')=='system' for m in messages):
            messages.insert(0,{'role':'system','content':system})
        messages.append({'role':'user','content':instruction})
        trajectory=[]; tools=0; final=None
        i=0; consecutive_empty=0; status_card_sent=False; failure=None; last_answer_seen=None
        # 2026-09-27 fix: while-loop so payloadless (empty/truncated max-thinking) turns do not consume the iteration budget
        while i<max_iter:
            if not status_card_sent and (i+1)>=int(max_iter*0.75):
                # 2026-09-27 fix: warn once near budget end so finished artifacts actually get submitted
                messages.append({'role':'user','content':'[HARNESS STATUS] Iteration %d/%d. The budget is nearly exhausted. If your artifacts are ready, submit NOW: respond with status=complete and a non-empty final_answer.'%(i+1,max_iter)})
                status_card_sent=True
            text=self._call(messages+[{'role':'system','content':'Budget: iterations remaining %d/%d; tool calls used %d/%d'%(max(0,max_iter-i),max_iter,tools,max_tools)}],temperature,max_tokens)
            messages.append({'role':'assistant','content':text if (text or '').strip() else '[harness: empty response discarded]'})
            if self._no_payload(text):
                consecutive_empty+=1
                if consecutive_empty>=3:
                    # 2026-09-27 fix: abort only after three consecutive payloadless responses
                    failure='empty_response_exhausted'; break
                messages.append({'role':'user','content':'Your previous response carried no usable JSON payload. Output exactly ONE minimal valid JSON object now. Do not restate your reasoning.'})
                continue
            consecutive_empty=0; i+=1
            try: obj=self._extract_json(text)
            except Exception as e:
                trajectory.append({'iteration':i,'error':'json_parse','text':text[-4000:]}); messages.append({'role':'user','content':'Return a valid JSON object only.'}); continue
            will_reserve=max_iter>=3 and i>max_iter-2
            if will_reserve and obj.get('status')!='complete' and not (isinstance(obj.get('final_answer'),dict) and obj['final_answer']):
                # 2026-09-27 fix: submit reserve — the last two turns refuse non-submit work and redirect to submission
                trajectory.append({'iteration':i,'reasoning':obj.get('reasoning'),'status':obj.get('status'),'note':'submit_reserve_refused'})
                messages.append({'role':'user','content':'The final turns are reserved for submission. Commands and reads are disabled. Respond with status=complete and your non-empty final_answer now.'})
                continue
            results=[]
            for j,spec in enumerate(obj.get('commands') or []):
                if tools>=max_tools: results.append({'error':'TOOL_BUDGET_EXHAUSTED'}); break
                if isinstance(spec,str):
                    source=spec; wd='.'
                elif isinstance(spec,dict):
                    source=str(spec.get('command','')); wd=str(spec.get('workdir','.'))
                else:
                    results.append({'error':'INVALID_COMMAND_SPEC','index':j}); continue
                payload=base64.b64encode(source.encode()).decode()
                cmd=f"python -c \"import base64; exec(compile(base64.b64decode('{payload}'), '<chembench-agent>', 'exec'))\""
                r=await environment.exec(command=cmd,cwd='/workspace/'+wd.strip('./'),timeout_sec=per_tool)
                tools+=1; results.append({'return_code':getattr(r,'return_code',None),'stdout':(getattr(r,'stdout','') or '')[-4000:],'stderr':(getattr(r,'stderr','') or '')[-2000:]})
            reads=[]
            for rel in (obj.get('read_files') or [])[:20]:
                rel=str(rel).replace('..','')
                r=await environment.exec(command=f"python -c 'from pathlib import Path; print(Path({json.dumps('/workspace/'+rel.lstrip('/'))}).read_text()[-10000:])'",cwd='/workspace',timeout_sec=30)
                reads.append({'path':rel,'content':(getattr(r,'stdout','') or '')[-10000:]})
            trajectory.append({'iteration':i,'reasoning':obj.get('reasoning'),'status':obj.get('status'),'tool_results':results,'read_results':reads})
            if obj.get('final_answer') is not None:
                last_answer_seen=obj['final_answer']
                # 2026-09-27 fix: only a non-empty dict counts as a submittable candidate; null/empty placeholders are not answers
                if isinstance(obj['final_answer'],dict) and obj['final_answer']: final=obj['final_answer']
            if obj.get('status')=='complete' and final is not None: break
            messages.append({'role':'user','content':json.dumps({'tool_results':results,'read_results':reads})})
        if failure is None and final is None and last_answer_seen is not None:
            # 2026-09-27 fix: best-effort submit of the last answer seen instead of discarding the whole run
            final=last_answer_seen; failure='best_effort_submission'
        if final is not None:
            enc=base64.b64encode(json.dumps(final,indent=2).encode()).decode()
            await environment.exec(command=f"python -c \"import base64; open('/workspace/final_answer.json','wb').write(base64.b64decode('{enc}'))\"",cwd='/workspace',timeout_sec=30)
        try:
            p=Path(getattr(self,'logs_dir','.') or '.'); p.mkdir(parents=True,exist_ok=True); (p/'chembench_agent_trajectory.json').write_text(json.dumps(trajectory,indent=2))
        except Exception: pass
        if hasattr(context,'metadata'): context.metadata={'iterations':len(trajectory),'tool_calls':tools,'failure':failure}
        state={'messages':messages}
        enc=base64.b64encode(json.dumps(state).encode()).decode()
        await environment.exec(command=f"python -c \"import base64; open('/workspace/protocol_state.json','wb').write(base64.b64decode('{enc}'))\"",cwd='/workspace',timeout_sec=30)
        return None
