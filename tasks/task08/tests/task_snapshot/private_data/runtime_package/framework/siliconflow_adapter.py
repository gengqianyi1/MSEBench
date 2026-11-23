"""Strict SiliconFlow OpenAI-compatible transport with no opaque SDK retry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .schemas import (
    ConfigurationError,
    MissingCredentialError,
    ModelConfig,
    ProviderError,
    SILICONFLOW_API_KEY_ENV,
    SILICONFLOW_BASE_URL,
)


MISSING_GJLD_API_KEY = "MISSING_GJLD_API_KEY"


class ProviderCallError(ProviderError):
    """Provider exception wrapper that intentionally discards exception text."""

    def __init__(self, original_type: str, safe_status: str = "PROVIDER_REQUEST_FAILED") -> None:
        self.original_type = original_type or "ProviderException"
        self.safe_status = safe_status
        super().__init__(safe_status)


def load_runtime_credential(project_root: str | Path | None = None) -> str:
    """Load only the configured or immediate-parent ``.env``.

    The secret is returned only to the caller constructing the SDK client.  A
    missing value is represented by the single stable public error code.  The
    default is deliberately bounded to one parent so an unrelated ancestor
    project cannot inject credentials when this package is extracted elsewhere.
    """

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    explicit = os.environ.get("ANT_BENCHMARK_ENV_FILE", "").strip()
    if explicit:
        env_path = Path(explicit).expanduser().resolve()
    else:
        env_path = root.parent / ".env"
    try:
        load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        # A dotenv parser/path failure is deliberately indistinguishable from a
        # missing credential at the public boundary.
        raise MissingCredentialError(MISSING_GJLD_API_KEY) from None
    value = os.environ.get(SILICONFLOW_API_KEY_ENV, "")
    if not isinstance(value, str) or not value.strip():
        raise MissingCredentialError(MISSING_GJLD_API_KEY)
    return value


class SiliconFlowChatTransport:
    """Non-streaming chat-completions transport for one exact configured model."""

    requires_api_key = True
    is_mock = False
    transport_name = "siliconflow_openai_compatible"

    def __init__(
        self,
        config: ModelConfig | Mapping[str, Any],
        api_key: str,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
        if (
            self.config.provider != "siliconflow"
            or self.config.transport != "openai_compatible"
            or self.config.base_url != SILICONFLOW_BASE_URL
            or self.config.api_key_env != SILICONFLOW_API_KEY_ENV
            or self.config.sdk_max_retries != 0
        ):
            raise ConfigurationError("invalid formal SiliconFlow transport configuration")
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingCredentialError(MISSING_GJLD_API_KEY)

        canonical = client_factory is None
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - pinned runtime dependency
                raise ConfigurationError("OpenAI-compatible client is unavailable") from exc
            client_factory = OpenAI
        try:
            self._client = client_factory(
                api_key=api_key,
                base_url=SILICONFLOW_BASE_URL,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            raise ProviderCallError(type(exc).__name__, "PROVIDER_CLIENT_INIT_FAILED") from None
        self._uses_canonical_client = canonical

    @property
    def uses_canonical_client(self) -> bool:
        return self._uses_canonical_client

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": min(float(timeout_seconds), self.config.timeout_seconds),
            "stream": False,
        }
        if tools:
            request["tools"] = [dict(tool) for tool in tools]
            request["tool_choice"] = "auto"
        try:
            return self._client.chat.completions.create(**request)
        except Exception as exc:
            # The original text/args/headers are never reflected or chained.
            raise ProviderCallError(type(exc).__name__) from None


# Short alias for callers that prefer a provider-neutral noun.
SiliconFlowAdapter = SiliconFlowChatTransport


__all__ = [
    "MISSING_GJLD_API_KEY",
    "ProviderCallError",
    "SiliconFlowAdapter",
    "SiliconFlowChatTransport",
    "load_runtime_credential",
]
