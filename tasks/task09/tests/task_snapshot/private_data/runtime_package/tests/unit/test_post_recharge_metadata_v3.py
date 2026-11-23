from __future__ import annotations

from types import SimpleNamespace

import scripts.run_suite_models_v3 as runner
from framework.schemas import ModelConfig
from framework.siliconflow_adapter import ProviderCallError
from framework.siliconflow_adapter import load_runtime_credential
from framework.trajectory import TrajectoryRecorder
from scripts.run_suite_models_v3 import _response_parts_v3


def test_explicit_env_file_replaces_stale_inherited_key(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GJLD_API_KEY=fresh-test-value\n", encoding="utf-8")
    monkeypatch.setenv("GJLD_API_KEY", "stale-test-value")
    monkeypatch.setenv("ANT_BENCHMARK_ENV_FILE", str(env_file))

    assert load_runtime_credential(tmp_path) == "fresh-test-value"


def test_v3_usage_projection_records_optional_reasoning_and_cache_counts() -> None:
    response = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4-Pro",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            cache_read_input_tokens=4,
            cache_creation_input_tokens=2,
        ),
    )

    actual, content, usage = _response_parts_v3(response)

    assert actual == "deepseek-ai/DeepSeek-V4-Pro"
    assert content == '{"ok":true}'
    assert usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "reasoning_tokens": 3,
        "cached_input_tokens": 5,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 2,
    }


def test_provider_retry_recreates_client_after_connection_failure(monkeypatch) -> None:
    instances: list[object] = []

    class FakeTransport:
        def __init__(self, config, api_key) -> None:
            assert config.model == "deepseek-ai/DeepSeek-V4-Pro"
            assert api_key == "test-key"
            instances.append(self)

        def complete(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            if len(instances) == 1:
                raise ProviderCallError(
                    {
                        "status": "NETWORK_CONNECTION_FAILED",
                        "retryable": True,
                    }
                )
            return SimpleNamespace(
                model="deepseek-ai/DeepSeek-V4-Pro",
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=SimpleNamespace(
                    prompt_tokens=2,
                    completion_tokens=3,
                    total_tokens=5,
                ),
            )

    monkeypatch.setattr(runner, "SiliconFlowChatTransport", FakeTransport)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    recorder = TrajectoryRecorder(
        run_id="fresh-client-test",
        protocol=runner.PROTOCOL,
    )
    recorder.append("run_start", {})

    actual, content, usage, retries = runner._provider_turn(
        ModelConfig.for_label("deepseek"),
        [{"role": "user", "content": "{}"}],
        api_key="test-key",
        recorder=recorder,
        turn=1,
    )

    assert len(instances) == 2
    assert retries == 1
    assert actual == "deepseek-ai/DeepSeek-V4-Pro"
    assert content == '{"ok":true}'
    assert usage == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
