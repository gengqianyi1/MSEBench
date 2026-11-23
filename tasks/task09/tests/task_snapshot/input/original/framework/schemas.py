"""Closed schemas and exact model configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY_ENV = "GJLD_API_KEY"
TASK_ID = "adaptive_xtb_torsion_profile_agent_v2"
BENCHMARK_VERSION = "2.0.0"
PROTOCOL = "adaptive_torsion_sampling_tool_loop/2.0"
CASE_ID = "2_fluorobiphenyl_central_torsion_v1"
FORMAL_MODELS = {
    "kimi": "Pro/moonshotai/Kimi-K2.6",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "glm": "zai-org/GLM-5.2",
}
ACTIVE_PREFLIGHT_REPORT = (
    "reports/PROVIDER_PREFLIGHT_RECHARGE_01.json"
)
ACTIVE_REPLAY_REPORT = "reports/replay_validation_recharge_01.json"
FORMAL_CAMPAIGN_ID = "formal_campaign_v2_recharge_01"
FORMAL_VALIDATION_REPORTS = {
    label: f"reports/{label}_formal_validation_recharge_01.json"
    for label in FORMAL_MODELS
}
FORMAL_TEMPERATURE = 0.0
FORMAL_MAX_OUTPUT_TOKENS = 384
FORMAL_MAX_PROVIDER_TURNS = 5
FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN = 4
FORMAL_MAXIMUM_CUMULATIVE_RETRY_DELAY_SECONDS = 60.0
PREFLIGHT_MAX_OUTPUT_TOKENS = 24
PREFLIGHT_PROVIDER_TURNS = 1
PREFLIGHT_NATIVE_XTB_CALLS = 0
CONTEXT_MODE = "canonical_state_v2"


class SchemaError(ValueError):
    pass


class ConfigurationError(RuntimeError):
    pass


class MissingCredentialError(ConfigurationError):
    pass


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    label: str
    model: str
    provider: str = "siliconflow"
    transport: str = "openai_compatible"
    base_url: str = SILICONFLOW_BASE_URL
    api_key_env: str = SILICONFLOW_API_KEY_ENV
    temperature: float = FORMAL_TEMPERATURE
    max_tokens: int = FORMAL_MAX_OUTPUT_TOKENS
    timeout_seconds: float = 180.0
    # The SDK never retries.  This is the number of application retries after
    # the initial attempt: 3 retries == 4 total attempts per provider turn.
    max_retries: int = FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN - 1
    sdk_max_retries: int = 0
    max_provider_turns: int = FORMAL_MAX_PROVIDER_TURNS
    context_mode: str = CONTEXT_MODE
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        expected = FORMAL_MODELS.get(self.label)
        if expected is None or self.model != expected:
            raise ConfigurationError("formal model identity is not allowlisted")
        if (
            self.provider != "siliconflow"
            or self.transport != "openai_compatible"
            or self.base_url != SILICONFLOW_BASE_URL
            or self.api_key_env != SILICONFLOW_API_KEY_ENV
            or self.sdk_max_retries != 0
            or self.temperature != FORMAL_TEMPERATURE
            or self.max_tokens != FORMAL_MAX_OUTPUT_TOKENS
            or self.max_provider_turns != FORMAL_MAX_PROVIDER_TURNS
            or self.context_mode != CONTEXT_MODE
            or self.fallback_allowed is not False
        ):
            raise ConfigurationError("formal provider configuration changed")
        if self.max_retries != FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN - 1:
            raise ConfigurationError("provider retry budget changed")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ConfigurationError("provider timeout is invalid")

    @classmethod
    def for_label(cls, label: str) -> "ModelConfig":
        try:
            model = FORMAL_MODELS[label]
        except KeyError as exc:
            raise ConfigurationError("unknown formal model label") from exc
        return cls(label=label, model=model)


@dataclass(frozen=True)
class ProviderPreflightConfig:
    """Frozen one-turn, no-native-compute provider availability probe."""

    label: str
    model: str
    provider: str = "siliconflow"
    transport: str = "openai_compatible"
    base_url: str = SILICONFLOW_BASE_URL
    api_key_env: str = SILICONFLOW_API_KEY_ENV
    temperature: float = FORMAL_TEMPERATURE
    max_tokens: int = PREFLIGHT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = 30.0
    max_provider_turns: int = PREFLIGHT_PROVIDER_TURNS
    native_xtb_calls: int = PREFLIGHT_NATIVE_XTB_CALLS
    formal_trajectory: bool = False
    sdk_max_retries: int = 0
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        expected = FORMAL_MODELS.get(self.label)
        if expected is None or self.model != expected:
            raise ConfigurationError("preflight model identity is not allowlisted")
        if (
            self.provider != "siliconflow"
            or self.transport != "openai_compatible"
            or self.base_url != SILICONFLOW_BASE_URL
            or self.api_key_env != SILICONFLOW_API_KEY_ENV
            or self.temperature != FORMAL_TEMPERATURE
            or self.max_tokens != PREFLIGHT_MAX_OUTPUT_TOKENS
            or self.max_provider_turns != PREFLIGHT_PROVIDER_TURNS
            or self.native_xtb_calls != PREFLIGHT_NATIVE_XTB_CALLS
            or self.formal_trajectory is not False
            or self.sdk_max_retries != 0
            or self.fallback_allowed is not False
        ):
            raise ConfigurationError("provider preflight configuration changed")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ConfigurationError("preflight timeout is invalid")

    @classmethod
    def for_label(cls, label: str) -> "ProviderPreflightConfig":
        try:
            model = FORMAL_MODELS[label]
        except KeyError as exc:
            raise ConfigurationError("unknown formal model label") from exc
        return cls(label=label, model=model)


def validate_scan_batch_action(value: Any) -> dict[str, Any]:
    """Validate the only normal v2 model action.

    Phase-specific batch sizes are enforced by the runtime because they depend
    on current trajectory state.  This closed schema deliberately has no
    profile ID, curve, extrema, barrier, path, model, or runtime-statistic
    fields.  With ``finalize=true`` the runtime performs the final automatic
    fit/analysis and creates the trusted closed envelope without another model
    turn.
    """

    if not isinstance(value, Mapping):
        raise SchemaError("scan batch action must be an object")
    expected = {
        "action",
        "angles_deg",
        "finalize",
        "confidence",
        "rationale",
    }
    if set(value) != expected:
        raise SchemaError("scan batch action keys do not match the closed schema")
    if value["action"] != "run_scan_batch":
        raise SchemaError("action must be run_scan_batch")

    raw_angles = value["angles_deg"]
    if not isinstance(raw_angles, list) or not 1 <= len(raw_angles) <= 6:
        raise SchemaError("angles_deg must contain one to six angles")
    angles: list[int] = []
    for raw in raw_angles:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise SchemaError("angles_deg values must be integers")
        if not 0 <= raw < 360 or raw % 5:
            raise SchemaError("angles_deg values must use the frozen 5-degree grid")
        angles.append(raw)
    if len(set(angles)) != len(angles):
        raise SchemaError("angles_deg contains duplicate angles")

    finalize = value["finalize"]
    if not isinstance(finalize, bool):
        raise SchemaError("finalize must be boolean")
    confidence = value["confidence"]
    if finalize:
        if confidence not in {"low", "medium", "high"}:
            raise SchemaError("terminal confidence is invalid")
    elif confidence is not None:
        raise SchemaError("confidence must be null before terminal finalization")

    rationale = value["rationale"]
    if (
        not isinstance(rationale, str)
        or not rationale.strip()
        or len(rationale) > 400
    ):
        raise SchemaError("rationale is invalid")
    return {
        "action": "run_scan_batch",
        "angles_deg": angles,
        "finalize": finalize,
        "confidence": confidence,
        "rationale": rationale.strip(),
    }


def validate_submission(value: Any) -> dict[str, Any]:
    """Fail-closed migration shim: v1.x ``submit_final`` is disabled in v2."""

    del value
    raise SchemaError(
        "submit_final is not a v2 action; use run_scan_batch(finalize=true)"
    )


__all__ = [
    "ACTIVE_PREFLIGHT_REPORT",
    "ACTIVE_REPLAY_REPORT",
    "BENCHMARK_VERSION",
    "CASE_ID",
    "CONTEXT_MODE",
    "ConfigurationError",
    "FORMAL_MAXIMUM_CUMULATIVE_RETRY_DELAY_SECONDS",
    "FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN",
    "FORMAL_MAX_OUTPUT_TOKENS",
    "FORMAL_MAX_PROVIDER_TURNS",
    "FORMAL_CAMPAIGN_ID",
    "FORMAL_MODELS",
    "FORMAL_TEMPERATURE",
    "FORMAL_VALIDATION_REPORTS",
    "MissingCredentialError",
    "ModelConfig",
    "PREFLIGHT_MAX_OUTPUT_TOKENS",
    "PREFLIGHT_NATIVE_XTB_CALLS",
    "PREFLIGHT_PROVIDER_TURNS",
    "PROTOCOL",
    "ProviderPreflightConfig",
    "ProviderError",
    "SILICONFLOW_API_KEY_ENV",
    "SILICONFLOW_BASE_URL",
    "SchemaError",
    "TASK_ID",
    "validate_scan_batch_action",
]
