"""Unified streaming LLM client (stdlib only).

Reads parameters from a unified config dict (see common/config). All tasks share
this client; no task keeps its own LLM parameter copy.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
import http.client
import socket


class LLMError(RuntimeError):
    pass


def _retry_wait(exc, default=65.0, cap=300.0):
    """Backoff for 406/429: honor Retry-After when the provider sends it."""
    try:
        headers = getattr(exc, "headers", None)
        header = headers.get("Retry-After") if headers else None
        if header:
            return min(cap, max(default, float(header)))
    except (TypeError, ValueError):
        pass
    return default


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enable_tcp_keepalive(sock):
    """Aggressive TCP keepalive so a proxy's idle timer never fires while the
    upstream model is 'thinking' (matrixllm cuts an idle connection at ~90s)."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, val in (("TCP_KEEPIDLE", 20), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 9)):
            if hasattr(socket, opt):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
    except OSError:
        pass


class _KAHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        _enable_tcp_keepalive(self.sock)


class _KAHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        super().connect()
        _enable_tcp_keepalive(self.sock)


class _KAHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_KAHTTPConnection, req)


class _KAHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_KAHTTPSConnection, req, context=self._context)


class LLMClient:
    def __init__(self, cfg: dict):
        self.api_format = str(cfg.get("api_format") or "openai").strip().lower()
        base = cfg.get("api_base", "").rstrip("/")
        if self.api_format == "anthropic":
            self.endpoint = base if base.endswith("/v1/messages") else base + "/v1/messages"
        else:
            self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.api_key = cfg.get("api_key") or cfg.get("_api_key", "")
        self.model = cfg.get("model")
        self.stream = _truthy(cfg.get("stream", True))
        self.enable_thinking = cfg.get("enable_thinking")
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.thinking = cfg.get("thinking")          # Anthropic thinking control
        self.output_config = cfg.get("output_config")  # Anthropic: {"effort": "low|medium|high"}
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = cfg.get("top_p")
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.frequency_penalty = cfg.get("frequency_penalty")
        self.presence_penalty = cfg.get("presence_penalty")
        self.seed = cfg.get("seed")
        self.stop = cfg.get("stop")
        self.timeout = float(cfg.get("request_timeout", 1800))
        self.retries = int(cfg.get("request_retries", 10))
        self.connection_close = _truthy(cfg.get("connection_close", True))
        self._opener = urllib.request.build_opener(_KAHTTPHandler(), _KAHTTPSHandler())

    def _headers(self):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key}
        if self.api_format == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        if self.connection_close:
            headers["Connection"] = "close"
        return headers

    def _anthropic_body(self, messages):
        """Anthropic /v1/messages payload: system is a top-level field."""
        system = "\n\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
        convo = [{"role": m.get("role"), "content": m.get("content") or ""}
                 for m in messages if m.get("role") != "system"]
        body = {"model": self.model, "messages": convo, "max_tokens": self.max_tokens}
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = float(self.top_p)
        if self.output_config is not None:
            body["output_config"] = self.output_config  # e.g. {"effort": "low|medium|high"}
        if self.thinking is not None:
            body["thinking"] = self.thinking           # extended-thinking control
        if self.stream:
            body["stream"] = True          # Anthropic SSE (message_start / content_block_delta / ...)
        return body

    def _body(self, messages, tools=None, json_schema=None):
        if self.api_format == "anthropic":
            return self._anthropic_body(messages)
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            body["top_p"] = float(self.top_p)
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = float(self.frequency_penalty)
        if self.presence_penalty is not None:
            body["presence_penalty"] = float(self.presence_penalty)
        if self.seed is not None:
            body["seed"] = self.seed
        if self.stop:
            body["stop"] = self.stop
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "agent_output", "schema": json_schema}}
        if self.stream:
            body["stream"] = True
        if self.enable_thinking is not None:
            body["enable_thinking"] = _truthy(self.enable_thinking)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    @staticmethod
    def _assemble_anthropic_sse(response):
        """Parse Anthropic /v1/messages SSE into the shared OpenAI-shaped dict."""
        content_parts, reasoning_parts = [], []
        finish_reason, model_name = None, None
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[len("data:"):].strip())
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            kind = ev.get("type")
            if kind == "message_start":
                model_name = (ev.get("message") or {}).get("model") or model_name
            elif kind == "content_block_delta":
                delta = ev.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    content_parts.append(delta["text"])
                elif delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str):
                    reasoning_parts.append(delta["thinking"])
            elif kind == "message_delta":
                stop = (ev.get("delta") or {}).get("stop_reason")
                if stop:
                    finish_reason = stop
            elif kind == "message_stop":
                break
        message = {"role": "assistant", "content": "".join(content_parts),
                   "reasoning_content": "".join(reasoning_parts)}
        return {"model": model_name, "choices": [{"message": message,
                "finish_reason": finish_reason or "stop"}]}

    @staticmethod
    def _assemble_sse(response, requested_model):
        role, content_parts, reasoning_parts, tc_map = None, [], [], {}
        finish_reason, model_name = None, None
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("model"), str) and chunk["model"]:
                model_name = chunk["model"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("role"):
                role = delta["role"]
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                entry = tc_map.setdefault(index, {"id": None, "type": "function",
                                                  "function": {"name": "", "arguments": ""}})
                if isinstance(call.get("id"), str) and call["id"]:
                    entry["id"] = call["id"]
                if isinstance(call.get("type"), str) and call["type"]:
                    entry["type"] = call["type"]
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                if isinstance(function.get("name"), str) and function["name"]:
                    entry["function"]["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    entry["function"]["arguments"] += function["arguments"]
        tool_calls = [tc_map[k] for k in sorted(tc_map)]
        for position, call in enumerate(tool_calls):
            if not call.get("id"):
                call["id"] = "call_%d_%d" % (position, int(time.time() * 1000))
        message = {"role": role or "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"model": model_name or requested_model,
                "choices": [{"message": message, "finish_reason": finish_reason or "stop"}]}

    def chat(self, messages, tools=None, json_schema=None):
        body = self._body(messages, tools=tools, json_schema=json_schema)
        request_id = str(uuid.uuid4())
        retryable = {406, 408, 429, 500, 502, 503, 504}
        last_error = None
        for attempt in range(max(1, self.retries)):
            req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                         headers=self._headers(), method="POST")
            started = time.monotonic()
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    ctype = response.headers.get("Content-Type") or ""
                    if "text/event-stream" in ctype:
                        data = (self._assemble_anthropic_sse(response)
                                if self.api_format == "anthropic"
                                else self._assemble_sse(response, body["model"]))
                    else:
                        data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:
                    pass
                # Some gateways (e.g. matrixllm) randomly route a request to an
                # upstream that rejects a normally-valid parameter (HTTP 400
                # "unsupported_parameter"). That is transient: re-sending usually
                # lands on a healthy upstream. Opt-in via env so other models keep
                # the strict no-retry behaviour on 400.
                transient_400 = (exc.code == 400 and "unsupported_parameter" in detail
                                 and _truthy(os.environ.get("MSEBENCH_RETRY_TRANSIENT_400")))
                if (exc.code in retryable or transient_400) and attempt + 1 < self.retries:
                    last_error = "HTTP %s: %s" % (exc.code, detail[:200])
                    time.sleep(_retry_wait(exc) if exc.code in (406, 429) else min(60.0, 2.0 ** (attempt + 1)))
                    continue
                raise LLMError("LLM HTTP %s: %s" % (exc.code, detail[:300])) from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt + 1 < self.retries:
                    last_error = "%s: %s" % (type(exc).__name__, exc)
                    time.sleep(min(60.0, 2.0 ** (attempt + 1)))
                    continue
                raise LLMError("%s: %s" % (type(exc).__name__, exc)) from exc
            if (self.api_format == "anthropic" and isinstance(data, dict)
                    and isinstance(data.get("content"), list)):
                if not isinstance(data.get("content"), list):
                    raise LLMError("empty/invalid anthropic response: %s" % str(data)[:200])
                text = "".join(b.get("text") or "" for b in data["content"]
                               if isinstance(b, dict) and b.get("type") == "text")
                reasoning = "".join(b.get("thinking") or "" for b in data["content"]
                                    if isinstance(b, dict) and b.get("type") == "thinking")
                return {
                    "request_id": request_id,
                    "model": data.get("model") or self.model,
                    "content": text,
                    "reasoning_content": reasoning,
                    "tool_calls": None,
                    "finish_reason": data.get("stop_reason"),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            if not isinstance(data, dict) or not data.get("choices"):
                raise LLMError("empty/invalid provider response")
            message = data["choices"][0].get("message") or {}
            return {
                "request_id": request_id,
                "model": data.get("model") or body["model"],
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or "",
                "tool_calls": message.get("tool_calls"),
                "finish_reason": data["choices"][0].get("finish_reason"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        raise LLMError("retries exhausted: %s" % last_error)
