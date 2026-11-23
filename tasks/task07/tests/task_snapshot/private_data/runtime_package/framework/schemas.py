"""Small, dependency-free schemas used by the benchmark agent runner.

The runtime deliberately does not depend on Pydantic.  Importing this module is
therefore safe in the offline test environment, even when the optional OpenAI
client is not installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


class RunnerError(RuntimeError):
    """Base class for expected runner failures."""


class ConfigurationError(RunnerError):
    """Raised when a task or model configuration is unsafe or invalid."""


class MissingCredentialError(RunnerError):
    """Raised before any network activity when a configured key is absent."""


class ProviderError(RunnerError):
    """Raised after all configured provider retries fail."""


class OutputParseError(RunnerError):
    """Raised when the model's final answer is not the exact required JSON."""


class BudgetExceededError(RunnerError):
    """Raised when the wall-clock or turn budget prevents continuation."""


@dataclass(frozen=True)
class BudgetConfig:
    max_tool_calls: int
    max_unique_evaluated_candidates: int
    max_agent_turns: int
    timeout_seconds: float

    @classmethod
    def from_task_config(cls, task_config: Mapping[str, Any]) -> "BudgetConfig":
        raw = task_config.get("budget")
        if not isinstance(raw, Mapping):
            raise ConfigurationError("task configuration must contain a 'budget' object")
        try:
            max_tool_calls = raw["max_tool_calls"]
            max_candidates = raw["max_unique_evaluated_candidates"]
            max_turns = raw["max_agent_turns"]
            timeout = raw["timeout_seconds"]
        except KeyError as exc:
            raise ConfigurationError(f"invalid task budget: {exc}") from exc
        for label, value in (
            ("max_tool_calls", max_tool_calls),
            ("max_unique_evaluated_candidates", max_candidates),
            ("max_agent_turns", max_turns),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"budget.{label} must be an integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ConfigurationError("budget.timeout_seconds must be a number")
        if not math.isfinite(float(timeout)):
            raise ConfigurationError("budget.timeout_seconds must be finite")
        result = cls(
            max_tool_calls=max_tool_calls,
            max_unique_evaluated_candidates=max_candidates,
            max_agent_turns=max_turns,
            timeout_seconds=float(timeout),
        )
        if (
            result.max_tool_calls < 0
            or result.max_unique_evaluated_candidates < 0
            or result.max_agent_turns < 1
            or result.timeout_seconds <= 0
            or result.max_tool_calls > 1_000_000
            or result.max_unique_evaluated_candidates > 1_000_000
            or result.max_agent_turns > 10_000
            or result.timeout_seconds > 86_400
        ):
            raise ConfigurationError("task budget limits are outside supported safe ranges")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INLINE_SECRET_KEYS = {
    "api_key",
    "apikey",
    "api_token",
    "access_token",
    "secret",
    "password",
}


TRUSTED_PROVIDER_BASE_URLS = {
    "moonshot": "https://api.moonshot.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "siliconflow": "https://api.siliconflow.cn/v1",
}

PROVIDER_MODEL_PREFIXES = {
    "moonshot": ("kimi-", "moonshot-"),
    "deepseek": ("deepseek-",),
    "zhipu": ("glm-",),
}

SILICONFLOW_API_KEY_ENV = "GJLD_API_KEY"
SILICONFLOW_MODEL_IDS = frozenset(
    {
        "zai-org/GLM-5.2",
        "deepseek-ai/DeepSeek-V4-Pro",
        "Pro/moonshotai/Kimi-K2.6",
    }
)
PROVIDER_EXACT_MODEL_IDS = {
    "siliconflow": SILICONFLOW_MODEL_IDS,
}

_SUPPORTED_PROVIDERS = frozenset((*TRUSTED_PROVIDER_BASE_URLS, "custom"))
_SUPPORTED_TRANSPORTS = frozenset({"openai_compatible"})

# These fields determine the audited request and may never be supplied through
# provider_options.  Keeping this list broader than the current request also
# prevents SDK escape hatches from silently replacing headers or request data.
_RESERVED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "timeout",
        "timeout_seconds",
        "stream",
        "api_key",
        "base_url",
        "headers",
        "extra_headers",
        "extra_body",
    }
)

# Options here are deliberately narrow and are copied only after all audited
# request fields have been fixed by the transport.  ``thinking`` is translated
# by the transport to the SDK's extra_body escape hatch; callers never control
# extra_body itself.
_SAFE_PROVIDER_OPTIONS = frozenset(
    {
        "seed",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "parallel_tool_calls",
        "reasoning_effort",
        "thinking",
    }
)


def _find_inline_secret_key(value: Any, prefix: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key)
            lowered = text.lower()
            location = f"{prefix}.{text}" if prefix else text
            secret_like = (
                lowered in _INLINE_SECRET_KEYS
                or lowered in {"authorization", "proxy_authorization", "cookie"}
                or lowered.endswith("_api_key")
                or lowered.endswith("_access_token")
                or lowered.endswith("_secret")
                or lowered.endswith("_password")
            )
            if secret_like and not lowered.endswith("_env"):
                return location
            nested = _find_inline_secret_key(item, location)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _find_inline_secret_key(item, f"{prefix}[{index}]")
            if nested:
                return nested
    return None


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    transport: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    provider_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        if not isinstance(raw, Mapping):
            raise ConfigurationError("model configuration must be an object")

        lowered = {str(key).lower() for key in raw}
        forbidden = sorted(lowered & _INLINE_SECRET_KEYS)
        if forbidden:
            raise ConfigurationError(
                "inline credentials are forbidden; configure only api_key_env "
                f"(found {', '.join(forbidden)})"
            )
        secret_location = _find_inline_secret_key(raw)
        if secret_location:
            raise ConfigurationError(
                "inline credentials are forbidden; configure only environment-variable "
                f"names (secret-like field {secret_location!r})"
            )

        known = {
            "provider",
            "transport",
            "model",
            "model_env",
            "base_url",
            "api_key_env",
            "temperature",
            "max_tokens",
            "timeout_seconds",
            "max_retries",
            "retry_backoff_seconds",
            "provider_options",
            "enabled",
        }
        unknown = sorted(str(key) for key in raw if str(key) not in known)
        if unknown:
            raise ConfigurationError(
                "unknown model configuration field(s) are forbidden: "
                + ", ".join(unknown)
            )

        if "provider" not in raw:
            raise ConfigurationError(
                "provider must explicitly identify moonshot, deepseek, zhipu, "
                "siliconflow, or custom"
            )
        provider = str(raw.get("provider", "")).strip().lower()
        if provider not in _SUPPORTED_PROVIDERS:
            raise ConfigurationError(
                f"unsupported provider {provider!r}; expected one of "
                + ", ".join(sorted(_SUPPORTED_PROVIDERS))
            )

        if "transport" not in raw:
            raise ConfigurationError("transport must explicitly be 'openai_compatible'")
        transport = str(raw.get("transport", "")).strip().lower()
        if transport not in _SUPPORTED_TRANSPORTS:
            raise ConfigurationError(
                f"unsupported transport {transport!r}; expected 'openai_compatible'"
            )

        api_key_env = str(raw.get("api_key_env", "")).strip()
        if not _ENV_NAME.fullmatch(api_key_env):
            raise ConfigurationError("api_key_env must be a valid environment-variable name")
        if provider == "siliconflow" and api_key_env != SILICONFLOW_API_KEY_ENV:
            raise ConfigurationError(
                "siliconflow api_key_env must be exactly "
                f"{SILICONFLOW_API_KEY_ENV!r}"
            )

        model = str(raw.get("model", "")).strip()
        base_url = str(raw.get("base_url", "")).strip()
        if not model:
            raise ConfigurationError("model must be a non-empty string")
        if not base_url:
            raise ConfigurationError("base_url must be a non-empty string")

        if any(character.isspace() for character in base_url):
            raise ConfigurationError("base_url must not contain whitespace")
        parsed_url = urlsplit(base_url)
        if (
            not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ConfigurationError(
                "base_url must be an absolute endpoint without credentials, query, or fragment"
            )
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed_url.scheme != "https" and not (
            parsed_url.scheme == "http" and parsed_url.hostname in loopback_hosts
        ):
            raise ConfigurationError(
                "base_url must use HTTPS (HTTP is permitted only for a loopback development endpoint)"
            )

        exact_models = PROVIDER_EXACT_MODEL_IDS.get(provider)
        prefixes = PROVIDER_MODEL_PREFIXES.get(provider)
        if exact_models is not None and model not in exact_models:
            raise ConfigurationError(
                f"model {model!r} is incompatible with provider {provider!r}; "
                "expected one of the exact configured model IDs"
            )
        if exact_models is None and prefixes is not None and not model.startswith(prefixes):
            raise ConfigurationError(
                f"model {model!r} is incompatible with provider {provider!r}; "
                f"expected prefix {prefixes}"
            )

        raw_options = raw.get("provider_options", {})
        if not isinstance(raw_options, Mapping):
            raise ConfigurationError("provider_options must be an object")
        option_keys = {str(key) for key in raw_options}
        reserved = sorted(
            key for key in option_keys if key.lower() in _RESERVED_REQUEST_FIELDS
        )
        if reserved:
            raise ConfigurationError(
                "provider_options cannot override audited request field(s): "
                + ", ".join(reserved)
            )
        unsafe = sorted(option_keys - _SAFE_PROVIDER_OPTIONS)
        if unsafe:
            raise ConfigurationError(
                "provider_options contains unsupported field(s): " + ", ".join(unsafe)
            )
        provider_options = {str(key): value for key, value in raw_options.items()}
        cls._validate_provider_options(provider_options)

        if "enabled" in raw and not isinstance(raw["enabled"], bool):
            raise ConfigurationError("enabled must be a boolean")

        temperature = raw.get("temperature", 0.2)
        max_tokens = raw.get("max_tokens", 4096)
        timeout_seconds = raw.get("timeout_seconds", 60)
        max_retries = raw.get("max_retries", 2)
        retry_backoff_seconds = raw.get("retry_backoff_seconds", 2)
        for label, value in (
            ("temperature", temperature),
            ("timeout_seconds", timeout_seconds),
            ("retry_backoff_seconds", retry_backoff_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"{label} must be a number")
            if not math.isfinite(float(value)):
                raise ConfigurationError(f"{label} must be finite")
        for label, value in (
            ("max_tokens", max_tokens),
            ("max_retries", max_retries),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"{label} must be an integer")

        if not 0 <= float(temperature) <= 2:
            raise ConfigurationError("temperature must be between 0 and 2")
        if not 1 <= max_tokens <= 1_000_000:
            raise ConfigurationError("max_tokens must be between 1 and 1000000")
        if not 0 < float(timeout_seconds) <= 86_400:
            raise ConfigurationError("timeout_seconds must be greater than 0 and at most 86400")
        if not 0 <= max_retries <= 10:
            raise ConfigurationError("max_retries must be between 0 and 10")
        if not 0 <= float(retry_backoff_seconds) <= 3_600:
            raise ConfigurationError(
                "retry_backoff_seconds must be between 0 and 3600"
            )

        result = cls(
            provider=provider,
            transport=transport,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=float(temperature),
            max_tokens=max_tokens,
            timeout_seconds=float(timeout_seconds),
            max_retries=max_retries,
            retry_backoff_seconds=float(retry_backoff_seconds),
            provider_options=provider_options,
        )
        return result

    @staticmethod
    def _validate_provider_options(options: Mapping[str, Any]) -> None:
        def number(name: str) -> None:
            if name not in options:
                return
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"provider_options.{name} must be a number")
            if not math.isfinite(float(value)):
                raise ConfigurationError(f"provider_options.{name} must be finite")

        for name in ("top_p", "frequency_penalty", "presence_penalty"):
            number(name)
        if "top_p" in options and not 0 <= float(options["top_p"]) <= 1:
            raise ConfigurationError("provider_options.top_p must be between 0 and 1")
        for name in ("frequency_penalty", "presence_penalty"):
            if name in options and not -2 <= float(options[name]) <= 2:
                raise ConfigurationError(
                    f"provider_options.{name} must be between -2 and 2"
                )
        if "seed" in options and (
            isinstance(options["seed"], bool) or not isinstance(options["seed"], int)
        ):
            raise ConfigurationError("provider_options.seed must be an integer")
        if "seed" in options and not -(2**63) <= options["seed"] < 2**63:
            raise ConfigurationError("provider_options.seed must fit in a signed 64-bit integer")
        if "parallel_tool_calls" in options and not isinstance(
            options["parallel_tool_calls"], bool
        ):
            raise ConfigurationError(
                "provider_options.parallel_tool_calls must be a boolean"
            )
        if "reasoning_effort" in options and (
            not isinstance(options["reasoning_effort"], str)
            or not options["reasoning_effort"].strip()
            or len(options["reasoning_effort"]) > 64
        ):
            raise ConfigurationError(
                "provider_options.reasoning_effort must be a non-empty string"
            )
        if "stop" in options:
            stop = options["stop"]
            if not (
                isinstance(stop, str)
                or (
                    isinstance(stop, list)
                    and len(stop) <= 4
                    and all(isinstance(item, str) and len(item) <= 512 for item in stop)
                )
            ):
                raise ConfigurationError(
                    "provider_options.stop must be a string or at most four strings"
                )
            if isinstance(stop, str) and len(stop) > 512:
                raise ConfigurationError(
                    "provider_options.stop strings must not exceed 512 characters"
                )
        if "thinking" in options:
            thinking = options["thinking"]
            if (
                not isinstance(thinking, Mapping)
                or set(thinking) != {"type"}
                or thinking.get("type") not in {"enabled", "disabled", "auto"}
            ):
                raise ConfigurationError(
                    "provider_options.thinking must be exactly "
                    "{'type': 'enabled'|'disabled'|'auto'}"
                )

    @property
    def is_trusted_endpoint(self) -> bool:
        """Whether provider, transport, endpoint, and requested model are attested."""

        expected = TRUSTED_PROVIDER_BASE_URLS.get(self.provider)
        exact_models = PROVIDER_EXACT_MODEL_IDS.get(self.provider)
        prefixes = PROVIDER_MODEL_PREFIXES.get(self.provider)
        model_is_trusted = (
            self.model in exact_models
            if exact_models is not None
            else prefixes is not None and self.model.startswith(prefixes)
        )
        return (
            self.transport == "openai_compatible"
            and expected is not None
            and self.base_url == expected
            and model_is_trusted
        )

    def response_model_compatible(self, returned_model: Any) -> bool:
        """Verify the provider-returned model against the configured model ID."""

        if not isinstance(returned_model, str) or returned_model != returned_model.strip():
            return False
        if not returned_model:
            return False
        exact_models = PROVIDER_EXACT_MODEL_IDS.get(self.provider)
        if exact_models is not None:
            # Providers registered with exact IDs must echo the configured ID
            # byte-for-byte.  Version-suffix tolerance is retained only for
            # legacy providers governed by a model-prefix allowlist.
            return self.model in exact_models and returned_model == self.model
        else:
            prefixes = PROVIDER_MODEL_PREFIXES.get(self.provider)
            if prefixes is not None and not returned_model.startswith(prefixes):
                return False
        if returned_model == self.model:
            return True
        if not returned_model.startswith(self.model):
            return False
        suffix = returned_model[len(self.model) :]
        return bool(suffix) and suffix[0] in {"-", "_", "@", ":", "/"}

    def sanitized_dict(self) -> dict[str, Any]:
        """Return the only model configuration representation written to disk."""

        return {
            "provider": self.provider,
            "transport": self.transport,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "provider_options": dict(self.provider_options),
        }


@dataclass(frozen=True)
class FinalCandidate:
    name: str
    smiles: str
    rationale: str

    @classmethod
    def from_exact_object(
        cls,
        value: Any,
        output_config: Mapping[str, Any] | None = None,
    ) -> "FinalCandidate":
        if not isinstance(value, dict):
            raise OutputParseError("final_candidate must be a JSON object")
        required = {"name", "smiles", "rationale"}
        actual = set(value)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise OutputParseError(
                f"final_candidate keys must be exactly {sorted(required)}; "
                f"missing={missing}, extra={extra}"
            )
        if not all(isinstance(value[key], str) for key in required):
            raise OutputParseError("all final_candidate fields must be strings")

        candidate = cls(
            name=value["name"],
            smiles=value["smiles"],
            rationale=value["rationale"],
        )
        if (
            not candidate.name.strip()
            or not candidate.smiles.strip()
            or not candidate.rationale.strip()
        ):
            raise OutputParseError("all final candidate strings must be non-empty")

        limits = output_config or {}
        checks = (
            ("name", candidate.name, int(limits.get("max_name_length", 128))),
            ("smiles", candidate.smiles, int(limits.get("max_smiles_length", 512))),
            (
                "rationale",
                candidate.rationale,
                int(limits.get("max_rationale_length", 2000)),
            ),
        )
        for label, text, maximum in checks:
            if len(text) > maximum:
                raise OutputParseError(f"final candidate {label} exceeds {maximum} characters")
        return candidate

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    status: str
    output_dir: Path
    final_answer: dict[str, Any]
    evaluation: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"
