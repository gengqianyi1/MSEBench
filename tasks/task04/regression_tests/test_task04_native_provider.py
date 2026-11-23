"""Exercise real request encoding and multi-turn Agent with an offline provider."""

import asyncio
import io
import json
import threading
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import test_workspace_sidecar_agent as fixtures
from test_workspace_sidecar_agent import config, server

agent_module = fixtures.agent_module
protocol = fixtures.protocol


def native(name, call_id, arguments="{}"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def response(message, finish="tool_calls"):
    return {
        "id": "fixture-response",
        "object": "chat.completion",
        "created": 0,
        "model": "actual/model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


@pytest.fixture
def run_offline(agent_module, protocol, monkeypatch, tmp_path):
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(protocol))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    real_open = agent_module.urllib.request.urlopen
    requests = []
    timeouts = []
    replies = []

    def provider_open(request, **kwargs):
        if request.full_url != "https://offline.invalid/chat/completions":
            return real_open(request, **kwargs)
        requests.append(json.loads(request.data))
        timeouts.append(kwargs["timeout"])
        assert replies, "unexpected extra provider request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(agent_module.urllib.request, "urlopen", provider_open)
    agent = agent_module.WorkspaceSidecarAgent(
        extra_env={
            "TEST_SIDECAR_URL": f"http://127.0.0.1:{http.server_port}",
            "CHEMBENCH_API_URL": "https://offline.invalid/chat/completions",
            "CHEMBENCH_API_KEY": "synthetic-key-not-a-real-credential",
            "CHEMBENCH_ATTEMPT_ID": "attempt-one",
            "CHEMBENCH_MODEL_ID": "requested/model",
        }
    )
    agent.logs_dir = tmp_path / "agent-logs"
    writes = {}

    async def capture(environment, outputs, required):
        writes.update(outputs)

    agent._write_outputs = capture

    def run(values):
        replies.extend(values)
        asyncio.run(
            agent.run(
                "Use the public tools and submit.",
                None,
                SimpleNamespace(metadata={}),
                cfg=config(),
            )
        )

    yield SimpleNamespace(
        run=run,
        requests=requests,
        timeouts=timeouts,
        agent=agent,
        protocol=protocol,
        writes=writes,
    )
    http.shutdown()
    http.server_close()
    thread.join()


def test_three_turn_native_calls_keep_ids_and_results(run_offline):
    r = run_offline
    r.run(
        [
            response(native("read_file", "read_1")),
            response(native("xtb_run", "job_1")),
            response(native("submit", "submit_1")),
        ]
    )
    assert r.protocol.adapter.calls == ["read_file", "xtb_run"]
    assert set(r.writes) == {"final_answer.json"}
    assert r.requests[0]["tools"][0] == {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "additionalProperties": False},
        },
    }
    third = r.requests[2]["messages"]
    calls = [m for m in third if m.get("tool_calls")]
    assert [m["tool_calls"][0]["id"] for m in calls] == ["read_1", "job_1"]
    results = [m for m in third if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["read_1", "job_1"]
    assert all(json.loads(m["content"])["ok"] for m in results)


def test_request_output_and_timeout_overrides_preserve_task_budgets(run_offline):
    r = run_offline
    r.agent._participant_env.update(
        CHEMBENCH_MAX_TOKENS="65536", CHEMBENCH_REQUEST_TIMEOUT_SECONDS="900"
    )
    r.run([response(native("submit", "finish"))])
    assert r.requests[0]["max_tokens"] == 65536
    assert r.timeouts == [900]
    record = json.loads((r.protocol.root / "current.json").read_text())
    assert record["public_case"]["budget"]["max_agent_turns"] == 3
    assert record["public_case"]["budget"]["max_tool_calls"] == 4
    assert record["agent_metrics"]["agent_turns_used"] == 1


def test_no_request_override_keeps_declared_default(run_offline):
    r = run_offline
    r.run([response(native("submit", "finish"))])
    assert r.requests[0]["max_tokens"] == 512
    assert r.timeouts == [300]


def test_thinking_protocol_state_is_returned_in_memory_not_logged(run_offline):
    r = run_offline
    first = native("read_file", "thinking_call")
    first["reasoning_content"] = "SYNTHETIC_PRIVATE_PROTOCOL_STATE"
    r.run([response(first), response(native("submit", "done"))])
    previous = next(m for m in r.requests[1]["messages"] if m.get("tool_calls"))
    assert previous["reasoning_content"] == "SYNTHETIC_PRIVATE_PROTOCOL_STATE"
    # DeepSeek V4 thinking rejects tool_choice; native auto is the API default.
    assert "tool_choice" not in r.requests[0]
    logs = (r.agent.logs_dir / "provider_exchanges.jsonl").read_text(encoding="utf-8")
    evidence = list(r.protocol.root.rglob("participant_events.json"))[0].read_text()
    assert "SYNTHETIC_PRIVATE_PROTOCOL_STATE" not in logs + evidence


def test_entire_native_batch_is_checked_before_any_execution(run_offline):
    r = run_offline
    batch = native("xtb_run", "must_not_run")
    batch["tool_calls"].extend(native("read_file", "bad_last", "{")["tool_calls"])
    r.run([response(batch), response(native("submit", "finish"))])
    assert r.protocol.adapter.calls == []
    assert not any(
        m.get("tool_call_id") == "must_not_run" for m in r.requests[1]["messages"]
    )


def test_multiple_native_results_are_ordered_and_model_errors_are_returned(run_offline):
    r = run_offline
    batch = native("read_file", "one")
    batch["tool_calls"].extend(
        native("xtb_run", "two", '{"unknown":true}')["tool_calls"]
    )
    r.run([response(batch), response(native("submit", "finish"))])
    results = [m for m in r.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["one", "two"]
    assert json.loads(results[0]["content"])["ok"] is True
    assert json.loads(results[1]["content"])["error"] == "INVALID_TOOL_ARGUMENTS"
    assert r.protocol.adapter.calls == ["read_file"]


@pytest.mark.parametrize(
    "bad",
    [
        response({"role": "assistant", "content": ""}, "stop"),
        response(native("xtb_run", "truncated", '{"x":'), "length"),
        response(native("xtb_run", "complete_but_length"), "length"),
        response(native("xtb_run", "bad_arguments", '{"x":'), "tool_calls"),
        TimeoutError("simulated provider request timeout"),
    ],
)
def test_retry_consumes_turn_but_never_reexecutes_successful_job(run_offline, bad):
    r = run_offline
    r.run(
        [
            response(native("xtb_run", "good_job")),
            bad,
            response(native("submit", "finish")),
        ]
    )
    assert r.protocol.adapter.calls == ["xtb_run"]
    record = json.loads((r.protocol.root / "current.json").read_text())
    assert record["agent_metrics"]["agent_turns_used"] == 3
    assert record["agent_metrics"]["tool_calls_used"] == 2
    assert len(r.requests) == 3
    assert [
        m["tool_call_id"] for m in r.requests[2]["messages"] if m["role"] == "tool"
    ] == ["good_job"]


def test_three_invalid_responses_stop_without_executing_tools(run_offline):
    r = run_offline
    bad = response(native("xtb_run", "broken", "{"))
    with pytest.raises(RuntimeError, match="PROVIDER_RETRY_EXHAUSTED"):
        r.run([bad, bad, bad])
    assert len(r.requests) == 3
    assert r.protocol.adapter.calls == []
    record = json.loads((r.protocol.root / "current.json").read_text())
    assert record["agent_metrics"]["termination_reason"] == "provider_or_agent_failure"


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_or_balance_error_stops_without_key_switch(run_offline, status):
    r = run_offline
    error = urllib.error.HTTPError(
        "https://offline.invalid",
        status,
        "denied",
        {},
        io.BytesIO(b'{"error":"denied"}'),
    )
    with pytest.raises(RuntimeError, match=f"PROVIDER_HTTP_{status}"):
        r.run([error])
    assert len(r.requests) == 1
    assert r.protocol.adapter.calls == []


def test_logs_include_failed_response_but_never_hidden_reasoning_or_key(run_offline):
    r = run_offline
    bad = native("xtb_run", "bad", "{")
    bad["reasoning_content"] = "SECRET_HIDDEN_THOUGHT"
    bad["content"] = "synthetic-key-not-a-real-credential"
    r.run([response(bad, "length"), response(native("submit", "done"))])
    files = list(r.agent.logs_dir.glob("provider*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert '"finish_reason": "length"' in text
    assert '"arguments": "{"' in text
    assert "SECRET_HIDDEN_THOUGHT" not in text
    assert "reasoning_content" not in text
    assert "synthetic-key-not-a-real-credential" not in text
    assert "Authorization" not in text


@pytest.mark.parametrize("mode", ["valid", "invalid"])
def test_scripted_fixture_accepts_native_request_and_result_messages(mode):
    root = Path(__file__).resolve().parents[1]
    public = root / "tasks/task-04-histidine-microstate-ensemble/environment/public"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/participant_fixture.py"),
            "--public",
            str(public),
            "--mode",
            mode,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("fixture did not start")
                time.sleep(0.05)
        body = {"model": "fixture", "messages": [], "tools": [{"type": "function"}]}

        def ask():
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as answer:
                return json.load(answer)

        first = ask()["choices"][0]["message"]
        assert first["tool_calls"]
        if mode == "valid":
            body["messages"].append(first)
            for call in first["tool_calls"]:
                result = {
                    "ok": True,
                    "tool": call["function"]["name"],
                    "result": {
                        "stdout": "SANDBOX_OK",
                        "status": "completed",
                        "job_id": "job_fixture",
                    },
                }
                body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result),
                    }
                )
            final = ask()["choices"][0]["message"]
        else:
            final = first
        assert final["tool_calls"][-1]["function"]["name"] == "submit"
    finally:
        process.terminate()
        process.wait(timeout=10)
