"""Harbor participant for public workspace-tool contracts; no scientific strategy."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from harbor.agents.base import BaseAgent


class RetryableProviderError(RuntimeError):
    """No tool from this response has executed; another counted turn may retry."""


class WorkspaceSidecarAgent(BaseAgent):
    @staticmethod
    def name():
        return "workspace-sidecar"

    @staticmethod
    def version():
        return "1.0.0"

    def __init__(self, *args, extra_env=None, **kwargs):
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self._participant_env = dict(extra_env or {})

    def _setting(self, key, default=None):
        return self._participant_env.get(key, os.environ.get(key, default))

    async def setup(self, environment):
        pass

    @staticmethod
    async def _read_config(environment):
        result = await environment.exec(
            command="cat /workspace/.chembench_agent.json",
            cwd="/workspace",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError("MISSING_PUBLIC_AGENT_CONFIG")
        return json.loads(result.stdout)

    def _request(self, url, body=None):
        request = urllib.request.Request(
            url,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=12600 if url.endswith(("/actions", "/tool")) else 60
            ) as response:
                value = json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise RuntimeError("SIDECAR_UNAVAILABLE_OR_PROTOCOL_ERROR") from exc
        if not isinstance(value, dict):
            raise TypeError("INVALID_SIDECAR_RESPONSE")
        return value

    def _provider_log(self, record):
        """Observable provider exchange on the organizer host, never workspace."""

        def observable(value):
            if isinstance(value, dict):
                return {
                    k: observable(v)
                    for k, v in value.items()
                    if k
                    not in {
                        "reasoning_content",
                        "reasoning",
                        "reasoning_details",
                        "thinking",
                        "Authorization",
                        "authorization",
                        "api_key",
                    }
                }
            if isinstance(value, list):
                return [observable(v) for v in value]
            return value

        directory = getattr(self, "logs_dir", None)
        if directory is None:
            return
        path = Path(directory) / "provider_exchanges.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(observable(record), ensure_ascii=False)
        key = self._setting("CHEMBENCH_API_KEY", "")
        if key:
            encoded = encoded.replace(key, "[REDACTED]")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    @staticmethod
    def _provider_tools(public_tools):
        result = []
        for item in public_tools:
            tool = item.get("function", item)
            function = {
                "name": tool["name"],
                "parameters": tool.get("parameters", tool.get("input_schema", {})),
            }
            if tool.get("description"):
                function["description"] = tool["description"]
            result.append({"type": "function", "function": function})
        return result

    @staticmethod
    def _native_message(choice):
        if choice.get("finish_reason") == "length":
            raise RetryableProviderError("PROVIDER_RESPONSE_LENGTH")
        message = choice.get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or not calls:
            raise RetryableProviderError("PROVIDER_NO_NATIVE_TOOL_CALLS")
        ids = set()
        for call in calls:
            try:
                cid, function = call["id"], call["function"]
                if (
                    call["type"] != "function"
                    or not isinstance(cid, str)
                    or not cid
                    or cid in ids
                    or not isinstance(function["name"], str)
                    or not function["name"]
                    or not isinstance(function["arguments"], str)
                    or not isinstance(json.loads(function["arguments"]), dict)
                ):
                    raise ValueError("invalid native function call")
                ids.add(cid)
            except (KeyError, TypeError, ValueError) as exc:
                raise RetryableProviderError("PROVIDER_INVALID_TOOL_ARGUMENTS") from exc
        # Keep native IDs and provider metadata. Never turn tool calls into prose.
        result = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": calls,
        }
        # Some providers require this opaque state on the next request. It stays
        # in memory and goes back only to that provider, never logs or sidecar.
        if isinstance(message.get("reasoning_content"), str):
            result["reasoning_content"] = message["reasoning_content"]
        return result

    def _request_json_stream(self, body, timeout):
        """Streaming provider round trip with transport-level retries.

        The gateway intermittently answers HTTP 406/5xx or resets connections;
        identical retried requests succeed. Retrying here preserves the turn
        budget that turn-level RetryableProviderError would otherwise consume.
        """
        endpoint = self._setting("CHEMBENCH_API_URL")
        if not endpoint:
            raise RuntimeError("MISSING_PROVIDER_CONFIGURATION")
        attempts = max(1, int(self._setting("CHEMBENCH_REQUEST_RETRIES", "10")))
        retryable_http = {406, 408, 429, 500, 502, 503, 504}
        last_transport = None
        for attempt in range(attempts):
            try:
                return self._consume_stream(endpoint, body, timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in retryable_http and attempt + 1 < attempts:
                    if exc.code == 429:
                        time.sleep(65.0)
                    else:
                        time.sleep(min(60.0, 2.0 ** (attempt + 1)))
                    continue
                raise
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last_transport = exc
                if attempt + 1 < attempts:
                    time.sleep(min(60.0, 2.0 ** (attempt + 1)))
                    continue
                raise
        if last_transport is not None:
            raise last_transport
        raise RetryableProviderError("PROVIDER_TRANSPORT_FAILURE")

    def _consume_stream(self, endpoint, body, timeout):
        """Streaming (SSE) port of the provider round trip.

        The gateway drops idle non-streaming connections after ~90s, which kills
        long reasoning generations. Streaming keeps bytes flowing; this reader
        reassembles deltas into the exact response shape `_complete` consumed.
        Falls back to plain JSON when the provider ignores `stream=true`.
        """
        request_body = dict(body)
        request_body["stream"] = True
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._setting("CHEMBENCH_API_KEY", ""),
            },
        )
        role = None
        content_parts = []
        reasoning_parts = []
        tc_map = {}
        finish_reason = None
        model_name = None
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type") or ""
            if "text/event-stream" not in content_type:
                raw = response.read().decode("utf-8")
                try:
                    return json.loads(raw)
                except ValueError as exc:
                    raise RetryableProviderError(
                        "PROVIDER_INVALID_JSON_RESPONSE"
                    ) from exc
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    if payload == "[DONE]":
                        break
                    continue
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
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    index = call.get("index", 0)
                    if not isinstance(index, int):
                        index = 0
                    entry = tc_map.setdefault(
                        index,
                        {"id": None, "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
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
        tool_calls = [tc_map[key] for key in sorted(tc_map)]
        for position, call in enumerate(tool_calls):
            if not call.get("id"):
                call["id"] = "call_stream_%d_%d" % (position, int(time.time() * 1000))
        message = {"role": role or "assistant", "content": "".join(content_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "model": model_name or body.get("model"),
            "choices": [
                {"message": message, "finish_reason": finish_reason or "stop"}
            ],
        }

    def _complete(self, messages, cfg, public_tools):
        endpoint = self._setting("CHEMBENCH_API_URL")
        requested = self._setting("CHEMBENCH_MODEL_ID") or getattr(
            self, "model_name", None
        )
        if not endpoint or not requested:
            raise RuntimeError("MISSING_PROVIDER_CONFIGURATION")
        body = {
            "model": requested,
            "messages": messages,
            "temperature": cfg["temperature"],
            "max_tokens": int(self._setting("CHEMBENCH_MAX_TOKENS", cfg["max_tokens"])),
            "tools": self._provider_tools(public_tools),
        }
        thinking = self._setting("CHEMBENCH_ENABLE_THINKING")
        if thinking:
            body["enable_thinking"] = thinking.strip().lower() in (
                "1", "true", "yes", "on"
            )
        effort = self._setting("CHEMBENCH_REASONING_EFFORT")
        if effort:
            body["reasoning_effort"] = effort
        started = time.monotonic()
        request_timeout = int(self._setting("CHEMBENCH_REQUEST_TIMEOUT_SECONDS", 300))
        record = {"request": body, "request_timeout_seconds": request_timeout}
        try:
            value = self._request_json_stream(body, request_timeout)
            record["response"] = value
        except urllib.error.HTTPError as exc:
            record.update(
                http_status=exc.code, error=exc.read().decode("utf-8", errors="replace")
            )
            error = (
                RetryableProviderError
                if exc.code in {406, 408, 429, 500, 502, 503, 504}
                else RuntimeError
            )
            raise error(f"PROVIDER_HTTP_{exc.code}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            record["error"] = type(exc).__name__
            raise RetryableProviderError("PROVIDER_TRANSPORT_FAILURE") from exc
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            self._provider_log(record)
        if not isinstance(value, dict):
            raise RetryableProviderError("PROVIDER_INVALID_RESPONSE")
        actual = value.get("model")
        if not isinstance(actual, str) or not actual:
            raise RuntimeError("MISSING_PROVIDER_ACTUAL_MODEL_ID")
        choices = value.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise RetryableProviderError("PROVIDER_EMPTY_CHOICES")
        return self._native_message(choices[0]), actual

    @staticmethod
    async def _write_outputs(environment, outputs, required):
        if not isinstance(outputs, dict) or set(outputs) - set(required):
            raise RuntimeError("UNDECLARED_PUBLIC_ARTIFACT")
        for name, encoded in outputs.items():
            if PurePosixPath(name).name != name or name in {".", ".."}:
                raise RuntimeError("INVALID_PUBLIC_ARTIFACT_PATH")
            base64.b64decode(encoded, validate=True)
            source = (
                "import base64,pathlib; "
                f"pathlib.Path('/workspace/{name}').write_bytes(base64.b64decode({encoded!r}))"
            )
            payload = base64.b64encode(source.encode()).decode()
            result = await environment.exec(
                command=f"python -c \"import base64;exec(base64.b64decode('{payload}'))\"",
                cwd="/workspace",
                timeout_sec=30,
            )
            if result.return_code != 0:
                raise RuntimeError("PUBLIC_ARTIFACT_WRITE_FAILED")

    async def run(self, instruction, environment, context, cfg=None):
        cfg = cfg if cfg is not None else await self._read_config(environment)
        if cfg.get("protocol_mode") != "workspace_sidecar_case":
            raise RuntimeError("UNSUPPORTED_PROTOCOL")
        env_name = cfg.get("sidecar_url_env")
        if not isinstance(env_name, str) or not env_name:
            raise RuntimeError("MISSING_SIDECAR_ENV_DECLARATION")
        url = (self._setting(env_name) or "").strip().rstrip("/")
        if not url:
            raise RuntimeError(f"MISSING_SIDECAR_ENDPOINT:{env_name}")
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
        ):
            raise RuntimeError("INVALID_SIDECAR_ENDPOINT")
        health = self._request(url + "/health")
        if health.get("ok") is not True or health.get("task_id") != cfg["task_id"]:
            raise RuntimeError("SIDECAR_TASK_MISMATCH")
        attempt = self._setting("CHEMBENCH_ATTEMPT_ID")
        model = self._setting("CHEMBENCH_MODEL_ID") or getattr(self, "model_name", None)
        if not attempt or not model:
            raise RuntimeError("MISSING_ORGANIZER_ATTEMPT_METADATA")
        started = self._request(
            url + "/session", {"attempt_id": attempt, "requested_model_id": model}
        )
        if started.get("ok") is not True:
            raise RuntimeError("SESSION_REJECTED")
        sid, case = started["session_id"], started["case"]
        if case["task"]["id"] != cfg["task_id"]:
            raise RuntimeError("SIDECAR_TASK_MISMATCH")
        contract_hash = hashlib.sha256(
            json.dumps(
                case["public_tools"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if contract_hash != cfg["tool_contract_sha256"]:
            raise RuntimeError("PUBLIC_TOOL_SCHEMA_MISMATCH")
        budget = case["budget"]
        if (
            budget["max_agent_turns"] != cfg["max_iterations"]
            or budget["max_tool_calls"] != cfg["max_tool_calls"]
        ):
            raise RuntimeError("PUBLIC_BUDGET_MISMATCH")
        names = [
            t.get("name") or t.get("function", {}).get("name")
            for t in case["public_tools"]
        ]
        if len(names) != len(set(names)) or set(names) != set(cfg["allowed_tools"]):
            raise RuntimeError("PUBLIC_TOOL_CONTRACT_MISMATCH")
        messages = [
            {
                "role": "system",
                "content": "Use the declared native function tools to interact with the task. "
                "Write the required submission files, "
                "then submit. Follow the public budgets and finalization restrictions.",
            },
            {
                "role": "user",
                "content": instruction + "\nPublic case:\n" + json.dumps(case),
            },
        ]
        submitted = None
        consecutive_failures = 0
        try:
            for _ in range(budget["max_agent_turns"]):
                turn = self._request(url + "/turn", {"session_id": sid})
                if not turn.get("ok"):
                    break
                messages.append(
                    {"role": "user", "content": "Budget status:\n" + json.dumps(turn)}
                )
                try:
                    message, actual = self._complete(
                        messages, cfg, case["public_tools"]
                    )
                except RetryableProviderError as exc:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError(
                            "PROVIDER_RETRY_EXHAUSTED:" + str(exc)
                        ) from exc
                    messages.append(
                        {
                            "role": "user",
                            "content": "The last provider request did not produce a complete native tool call. "
                            "No tools from that response executed. Use a compact, complete native "
                            "tool call. Do not repeat tools whose results were already returned.",
                        }
                    )
                    continue
                consecutive_failures = 0
                self._request(
                    url + "/provider",
                    {
                        "session_id": sid,
                        "actual_model_id": actual,
                        "assistant_content": json.dumps(
                            {
                                k: v
                                for k, v in message.items()
                                if k != "reasoning_content"
                            },
                            ensure_ascii=False,
                        ),
                    },
                )
                messages.append(message)
                calls = message["tool_calls"]
                action = {
                    "tool_calls": [
                        {
                            "tool": call["function"]["name"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        }
                        for call in calls
                    ]
                }
                response = self._request(
                    url + "/actions", {"session_id": sid, "action": action}
                )
                if response.get("terminal"):
                    submitted = response
                    break
                results = response.get("results")
                if not isinstance(results, list) or len(results) != len(calls):
                    raise RuntimeError("SIDECAR_TOOL_RESULT_COUNT_MISMATCH")
                for call, result in zip(calls, results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            if submitted is None:
                submitted = self._request(
                    url + "/finalize", {"session_id": sid, "reason": "agent_turn_limit"}
                )
        except Exception:
            self._request(
                url + "/finalize",
                {"session_id": sid, "reason": "provider_or_agent_failure"},
            )
            raise
        await self._write_outputs(
            environment,
            submitted.get("public_artifacts", {}),
            cfg["required_artifacts"],
        )
        if hasattr(context, "metadata"):
            context.metadata = {
                "attempt_id": attempt,
                "termination_reason": submitted.get("termination_reason"),
                "evidence_id": submitted.get("evidence_id"),
            }
