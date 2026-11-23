"""Clean-room OpenAI-compatible chat-completions client.

Only the standard library is used.  Credentials are read from named environment
variables and are never returned or written to logs.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from common.io_utils import redact


class ModelClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    api_key_env: str
    model: str
    api_key_env_fallbacks: tuple[str, ...] = ()
    temperature: float = 0.2
    timeout_seconds: int = 180
    provider: str = "openai_compatible"
    max_tokens: int = 8192
    max_retries: int = 3


class ModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        api_key_env_fallbacks: Sequence[str] | None = None,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: int = 180,
        provider: str = "openai_compatible",
        extra_body: Mapping[str, Any] | None = None,
        default_max_tokens: int = 8192,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        opener: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        localhost = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in ({"https"} if not localhost else {"http", "https"}):
            raise ValueError("base_url must use HTTPS (HTTP is allowed only for localhost tests)")
        if not parsed.netloc:
            raise ValueError("base_url must include a host")
        if not api_key_env or not model:
            raise ValueError("api_key_env and model are required")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key_env_fallbacks = tuple(str(value) for value in (api_key_env_fallbacks or ()) if str(value))
        self.model = model
        self.temperature = float(temperature)
        self.timeout_seconds = int(timeout_seconds)
        self.provider = str(provider)
        self.extra_body = dict(extra_body or {})
        self.default_max_tokens = int(default_max_tokens)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._opener = opener or urllib.request.build_opener()
        self._sleeper = sleeper

    @property
    def metadata(self) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(self.base_url)
        return {
            "provider": self.provider,
            "base_url_origin": f"{parsed.scheme}://{parsed.netloc}",
            "api_key_envs": [self.api_key_env, *self.api_key_env_fallbacks],
            "model": self.model,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "default_max_tokens": self.default_max_tokens,
            "extra_body": redact(self.extra_body),
        }

    @staticmethod
    def _balance_exhausted(status: int, detail: str) -> bool:
        # The preregistered campaign switches away from the voucher credential
        # only for an explicit 403 balance-exhausted response. Never switch for
        # 401, generic permission failures, or status without balance wording.
        if status != 403:
            return False
        lowered = detail.lower()
        return any(token in lowered for token in ("balance", "insufficient", "quota exhausted", "余额", "欠费"))

    def _credential_chain(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for name in (self.api_key_env, *self.api_key_env_fallbacks):
            value = os.environ.get(name)
            if value:
                result.append((name, value))
        if not result:
            names = ", ".join((self.api_key_env, *self.api_key_env_fallbacks))
            raise ModelClientError(f"missing API credential environment variable(s): {names}")
        return result

    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if seed is not None:
            payload["seed"] = int(seed)
        payload["max_tokens"] = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        for key, value in self.extra_body.items():
            if key in {"model", "messages", "tools", "tool_choice"}:
                raise ModelClientError(f"extra_body may not override reserved field: {key}")
            payload[key] = value
        raw: bytes | None = None
        credential_used: str | None = None
        credentials = self._credential_chain()
        for credential_index, (credential_name, api_key) in enumerate(credentials):
            switch_to_fallback = False
            for attempt in range(self.max_retries + 1):
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "xtb-llm-benchmark/2.0",
                    },
                    method="POST",
                )
                try:
                    with self._opener.open(request, timeout=self.timeout_seconds) as response:
                        raw = response.read()
                    credential_used = credential_name
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read(2000).decode("utf-8", errors="replace")
                    if self._balance_exhausted(exc.code, detail) and credential_index + 1 < len(credentials):
                        switch_to_fallback = True
                        break
                    if exc.code in {429, 503, 504} and attempt < self.max_retries:
                        self._sleeper(self.retry_backoff_seconds * (2**attempt))
                        continue
                    raise ModelClientError(f"model API returned HTTP {exc.code}: {redact(detail)}") from exc
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    ssl.SSLError,
                    http.client.RemoteDisconnected,
                    http.client.IncompleteRead,
                    ConnectionError,
                ) as exc:
                    if attempt < self.max_retries:
                        self._sleeper(self.retry_backoff_seconds * (2**attempt))
                        continue
                    raise ModelClientError(f"model API request failed: {exc}") from exc
            if raw is not None:
                break
            if not switch_to_fallback:
                break
        if raw is None:
            raise ModelClientError("all configured model API credentials were exhausted")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelClientError("model API returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ModelClientError("model API response must be an object")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelClientError("model API response has no choices[0]")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelClientError("model API response has no choices[0].message")
        # Hidden reasoning fields must not become observable trajectory content.
        clean_message = {
            key: value
            for key, value in message.items()
            if key not in {"reasoning", "reasoning_content", "chain_of_thought", "analysis"}
        }
        return {
            "id": result.get("id"),
            "model": result.get("model") or self.model,
            "created": result.get("created"),
            "message": clean_message,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {},
            "credential_env": credential_used,
        }
