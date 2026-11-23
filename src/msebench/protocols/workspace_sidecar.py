"""workspace_sidecar protocol plugin (tasks 04/05/06/18).

Faithful port of the chembench WorkspaceSidecarAgent.run() master loop; LLM
transport is delegated to the unified base agent's streaming _request_json.
Error strings are kept byte-identical: the WorkspaceSidecarVerifier and the
package regression fixtures depend on them.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from msebench.context.compactor import ContextCompactor


class RetryableProviderError(RuntimeError):
    """No tool from this response has executed; another counted turn may retry."""


def _setting(agent, key, default=None):
    for source in (getattr(agent, "_participant_env", None)
                   or getattr(agent, "extra_env", None) or {}, os.environ):
        value = source.get(key) if callable(getattr(source, "get", None)) else None
        if value is not None and value != "":
            return value
    return default


def _request(agent, url, body=None):
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
    result = {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
    if isinstance(message.get("reasoning_content"), str):
        result["reasoning_content"] = message["reasoning_content"]
    return result


def _provider_log(agent, record):
    def observable(value):
        if isinstance(value, dict):
            return {
                k: observable(v)
                for k, v in value.items()
                if k not in {"reasoning_content", "reasoning", "reasoning_details",
                             "thinking", "Authorization", "authorization", "api_key"}
            }
        if isinstance(value, list):
            return [observable(v) for v in value]
        return value

    directory = getattr(agent, "logs_dir", None)
    if directory is None:
        return
    path = Path(directory) / "provider_exchanges.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(observable(record), ensure_ascii=False)
    key = str(_setting(agent, "CHEMBENCH_API_KEY", "") or "")
    if key:
        encoded = encoded.replace(key, "[REDACTED]")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _compact_messages(messages, max_chars=100000, keep_reasoning_turns=2, keep_reasoning_chars=20000):
    """Keep the transcript inside the provider context budget.

    Per-turn reasoning and large tool blobs inflate the prompt until the model
    spends its whole output budget on reasoning and returns
    finish_reason=length with no tool call.  Reasoning is kept only for the most
    recent turns, every payload is capped, and only the protocol head plus the
    recent turns that fit the budget are retained, preserving the
    assistant/tool-call pairing the provider requires.
    """
    with_reasoning = [i for i, msg in enumerate(messages)
                      if msg.get("role") == "assistant" and msg.get("reasoning_content")]
    keep = set(with_reasoning[-keep_reasoning_turns:]) if with_reasoning else set()
    compact = []
    for index, msg in enumerate(messages):
        item = dict(msg)
        reasoning = item.get("reasoning_content")
        if index not in keep:
            item.pop("reasoning_content", None)
        elif isinstance(reasoning, str) and len(reasoning) > keep_reasoning_chars:
            item["reasoning_content"] = reasoning[:keep_reasoning_chars] + "\n...[truncated for context budget]"
        content = item.get("content")
        if isinstance(content, str):
            limit = 20000 if item.get("role") == "tool" else 12000
            if len(content) > limit:
                item["content"] = content[:limit] + "\n...[truncated for context budget]"
        compact.append(item)

    def payload_size(items):
        return sum(len(str(m.get("content", ""))) + len(str(m.get("reasoning_content", ""))) for m in items)

    if payload_size(compact) <= max_chars:
        return compact
    head = compact[:2]
    budget = max_chars - payload_size(head)
    tail = []
    used = 0
    for index in range(len(compact) - 1, 1, -1):
        item = compact[index]
        length = payload_size([item])
        if tail and used + length > budget:
            break
        tail.insert(0, item)
        used += length
    if tail and tail[0].get("role") == "tool":
        for index in range(len(compact) - len(tail) - 1, 1, -1):
            if compact[index].get("role") == "assistant" and compact[index].get("tool_calls"):
                tail.insert(0, compact[index])
                break
    retained = head + tail
    while retained and retained[0].get("role") == "tool":
        retained.pop(0)
    pending = set()
    for idx, item in enumerate(list(retained)):
        if item.get("role") == "assistant" and item.get("tool_calls"):
            pending = {call.get("id") for call in item.get("tool_calls", [])
                       if isinstance(call, dict) and call.get("id")}
            continue
        if item.get("role") != "tool":
            continue
        call_id = item.get("tool_call_id")
        if call_id in pending:
            pending.discard(call_id)
            continue
        recovered = dict(item)
        recovered.pop("tool_call_id", None)
        recovered["role"] = "user"
        recovered["content"] = "Tool result (context recovery): " + str(item.get("content", ""))
        retained[idx] = recovered
    return retained


def _complete(agent, messages, cfg, public_tools):
    model = _setting(agent, "CHEMBENCH_MODEL_ID") or getattr(agent, "model_name", None)
    key = _setting(agent, "CHEMBENCH_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    body = {
        "model": model,
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": int(_setting(agent, "CHEMBENCH_MAX_TOKENS", cfg["max_tokens"])),
        "tools": _provider_tools(public_tools),
    }
    thinking = _setting(agent, "CHEMBENCH_ENABLE_THINKING")
    if thinking:
        body["enable_thinking"] = str(thinking).strip().lower() in ("1", "true", "yes", "on")
    effort = _setting(agent, "CHEMBENCH_REASONING_EFFORT")
    if effort:
        body["reasoning_effort"] = effort
    body["__api_key"] = key
    timeout = float(_setting(agent, "CHEMBENCH_REQUEST_TIMEOUT_SECONDS", "900"))
    started = time.monotonic()
    data = None
    try:
        data = agent._request_json(body, timeout)
    except RuntimeError as exc:
        text = str(exc)
        for code in (406, 408, 429, 500, 502, 503, 504):
            if f"LLM HTTP {code}" in text:
                raise RetryableProviderError(f"PROVIDER_HTTP_{code}") from exc
        lower = text.lower()
        if ("empty" in lower or "json" in lower or "timed_out" in lower
                or "timed out" in lower or "urlerror" in lower or "connection" in lower):
            raise RetryableProviderError("PROVIDER_TRANSPORT_FAILURE") from exc
        raise
    finally:
        _provider_log(agent, {
            "request": {k: v for k, v in body.items() if k != "__api_key"},
            "request_timeout_seconds": int(timeout),
            "response": data,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
    if not isinstance(data, dict) or not data.get("choices"):
        raise RetryableProviderError("PROVIDER_EMPTY_CHOICES")
    actual = data.get("model")
    if not isinstance(actual, str) or not actual:
        actual = model
    return _native_message(data["choices"][0]), actual


def _summarize(agent, arg):
    """Auxiliary LLM digest call for context compaction (not an agent turn)."""
    messages = arg if isinstance(arg, list) else [{"role": "user", "content": str(arg)}]
    model = _setting(agent, "CHEMBENCH_MODEL_ID") or getattr(agent, "model_name", None)
    key = _setting(agent, "CHEMBENCH_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 8192}
    top_p = _setting(agent, "CHEMBENCH_TOP_P")
    if top_p:
        body["top_p"] = float(top_p)
    thinking = str(_setting(agent, "CHEMBENCH_ENABLE_THINKING") or "").strip().lower()
    if thinking in ("1", "true", "yes", "on"):
        body["enable_thinking"] = True
    elif thinking in ("0", "false", "no", "off"):
        body["enable_thinking"] = False
    body["reasoning_effort"] = "low"
    body["__api_key"] = key
    data = agent._request_json(
        body, float(_setting(agent, "CHEMBENCH_REQUEST_TIMEOUT_SECONDS", "900"))
    )
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") not in (None, "stop"):
        raise RuntimeError("summary finish_reason=%s" % choice.get("finish_reason"))
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("summary empty")
    return content


async def _write_outputs(agent, environment, outputs, required):
    if not isinstance(outputs, dict) or set(outputs) - set(required):
        raise RuntimeError("UNDECLARED_PUBLIC_ARTIFACT")
    for name, encoded in outputs.items():
        if PurePosixPath(name).name != name or name in {".", ".."}:
            raise RuntimeError("INVALID_PUBLIC_ARTIFACT_PATH")
        base64.b64decode(encoded, validate=True)
        payload = (
            "import base64; from pathlib import Path; "
            f"Path('/workspace/{name}').write_bytes(base64.b64decode({json.dumps(encoded)}))"
        )
        await environment.exec(
            command=f"python -c {json.dumps(payload)}", cwd="/workspace", timeout_sec=30
        )


async def run(agent, instruction, environment, context, cfg):
    if cfg.get("protocol_mode") != "workspace_sidecar_case":
        raise RuntimeError("UNSUPPORTED_PROTOCOL")
    env_name = cfg.get("sidecar_url_env")
    if not isinstance(env_name, str) or not env_name:
        raise RuntimeError("MISSING_SIDECAR_ENV_DECLARATION")
    url = (str(_setting(agent, env_name) or "")).strip().rstrip("/")
    if not url:
        raise RuntimeError(f"MISSING_SIDECAR_ENDPOINT:{env_name}")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise RuntimeError("INVALID_SIDECAR_ENDPOINT")
    health = _request(agent, url + "/health")
    if health.get("ok") is not True or health.get("task_id") != cfg["task_id"]:
        raise RuntimeError("SIDECAR_TASK_MISMATCH")
    attempt = _setting(agent, "CHEMBENCH_ATTEMPT_ID")
    model = _setting(agent, "CHEMBENCH_MODEL_ID") or getattr(agent, "model_name", None)
    if not attempt or not model:
        raise RuntimeError("MISSING_ORGANIZER_ATTEMPT_METADATA")
    started = _request(agent, url + "/session",
                       {"attempt_id": attempt, "requested_model_id": model})
    if started.get("ok") is not True:
        raise RuntimeError("SESSION_REJECTED")
    sid, case = started["session_id"], started["case"]
    if case["task"]["id"] != cfg["task_id"]:
        raise RuntimeError("SIDECAR_TASK_MISMATCH")
    contract_hash = hashlib.sha256(
        json.dumps(case["public_tools"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if contract_hash != cfg["tool_contract_sha256"]:
        raise RuntimeError("PUBLIC_TOOL_SCHEMA_MISMATCH")
    budget = case["budget"]
    if (budget["max_agent_turns"] != cfg["max_iterations"]
            or budget["max_tool_calls"] != cfg["max_tool_calls"]):
        raise RuntimeError("PUBLIC_BUDGET_MISMATCH")
    names = [t.get("name") or t.get("function", {}).get("name") for t in case["public_tools"]]
    if len(names) != len(set(names)) or set(names) != set(cfg["allowed_tools"]):
        raise RuntimeError("PUBLIC_TOOL_CONTRACT_MISMATCH")
    messages = [
        {"role": "system",
         "content": "Use the declared native function tools to interact with the task. "
                    "Write the required submission files, "
                    "then submit. Follow the public budgets and finalization restrictions."},
        {"role": "user", "content": instruction + "\nPublic case:\n" + json.dumps(case)},
    ]
    submitted = None
    consecutive_failures = 0
    compactor = ContextCompactor(summarize=lambda arg: _summarize(agent, arg))
    try:
        for _ in range(budget["max_agent_turns"]):
            turn = _request(agent, url + "/turn", {"session_id": sid})
            if not turn.get("ok"):
                break
            messages.append({"role": "user", "content": "Budget status:\n" + json.dumps(turn)})
            wire = compactor.project(messages)
            try:
                message, actual = _complete(agent, wire, cfg, case["public_tools"])
            except RetryableProviderError as exc:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise RuntimeError("PROVIDER_RETRY_EXHAUSTED:" + str(exc)) from exc
                messages.append({
                    "role": "user",
                    "content": ("The last provider request did not produce a complete native tool call. "
                               "No tools from that response executed. Use a compact, complete native "
                               "tool call. Do not repeat tools whose results were already returned.")
                               if consecutive_failures == 1 else
                               ("Still no complete native tool call (%d/3). Output EXACTLY ONE complete "
                                "native tool call and nothing else." % consecutive_failures),
                })
                continue
            consecutive_failures = 0
            _request(agent, url + "/provider", {
                "session_id": sid,
                "actual_model_id": actual,
                "assistant_content": json.dumps(
                    {k: v for k, v in message.items() if k != "reasoning_content"},
                    ensure_ascii=False),
            })
            messages.append(message)
            calls = message["tool_calls"]
            action = {"tool_calls": [
                {"tool": call["function"]["name"],
                 "arguments": json.loads(call["function"]["arguments"])}
                for call in calls]}
            response = _request(agent, url + "/actions", {"session_id": sid, "action": action})
            if response.get("terminal"):
                submitted = response
                break
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(calls):
                raise RuntimeError("SIDECAR_TOOL_RESULT_COUNT_MISMATCH")
            for call, result in zip(calls, results):
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
        if submitted is None:
            submitted = _request(agent, url + "/finalize",
                                 {"session_id": sid, "reason": "agent_turn_limit"})
    except Exception:
        _request(agent, url + "/finalize",
                 {"session_id": sid, "reason": "provider_or_agent_failure"})
        raise
    await _write_outputs(agent, environment,
                         submitted.get("public_artifacts", {}),
                         cfg["required_artifacts"])
    if hasattr(context, "metadata"):
        context.metadata = {
            "attempt_id": attempt,
            "termination_reason": submitted.get("termination_reason"),
            "evidence_id": submitted.get("evidence_id"),
        }
    return None
