"""SiliconFlow OpenAI-compatible chat-completions transport.

The adapter keeps provider-specific validation and request construction outside
the generic agent loop. Provider retries remain in :class:`AgentRunner`, where
every attempt is recorded in the trajectory; the SDK's opaque retries are
therefore disabled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .schemas import (
    ConfigurationError,
    ModelConfig,
    ProviderError,
    SILICONFLOW_API_KEY_ENV,
    SILICONFLOW_MODEL_IDS,
    TRUSTED_PROVIDER_BASE_URLS,
)


SILICONFLOW_BASE_URL = TRUSTED_PROVIDER_BASE_URLS["siliconflow"]


class SiliconFlowChatTransport:
    """Non-streaming SiliconFlow function-calling transport."""

    requires_api_key = True
    is_mock = False

    def __init__(
        self,
        config: ModelConfig,
        api_key: str,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if (
            config.provider != "siliconflow"
            or config.transport != "openai_compatible"
            or config.base_url != SILICONFLOW_BASE_URL
            or config.api_key_env != SILICONFLOW_API_KEY_ENV
            or config.model not in SILICONFLOW_MODEL_IDS
        ):
            raise ConfigurationError(
                "SiliconFlow transport requires the official endpoint, credential "
                "environment variable, and an approved exact model ID"
            )
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("SiliconFlow credential must be a non-empty string")

        uses_default_client = client_factory is None
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise ConfigurationError(
                    "the 'openai' package is required for SiliconFlow model runs"
                ) from exc
            client_factory = OpenAI

        self._config = config
        self._uses_default_client = uses_default_client
        try:
            self._client = client_factory(
                api_key=api_key,
                base_url=SILICONFLOW_BASE_URL,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            raise ConfigurationError(
                "SiliconFlow client initialization failed "
                f"({type(exc).__name__})"
            ) from None

    @property
    def uses_canonical_client(self) -> bool:
        """Whether the transport constructed the pinned default SDK client."""

        return self._uses_default_client

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": min(float(timeout_seconds), self._config.timeout_seconds),
            "stream": False,
        }
        for name, value in self._config.provider_options.items():
            if name == "thinking":
                request["extra_body"] = {"thinking": dict(value)}
            else:
                request[name] = value
        try:
            return self._client.chat.completions.create(**request)
        except Exception as exc:
            # Never reflect provider exception text: SDK/network errors can
            # include request headers. The runner still records the safe type,
            # retry attempt, and terminal failure.
            raise ProviderError(
                "SiliconFlow chat completion failed "
                f"({type(exc).__name__})"
            ) from None


_CANONICAL_SILICONFLOW_CHAT_TRANSPORT = SiliconFlowChatTransport


__all__ = [
    "SILICONFLOW_BASE_URL",
    "SiliconFlowChatTransport",
]
