from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

import framework.schemas as schema_module
from framework.schemas import (
    ACTIVE_PREFLIGHT_REPORT,
    ACTIVE_REPLAY_REPORT,
    FORMAL_MODELS,
    FORMAL_CAMPAIGN_ID,
    FORMAL_VALIDATION_REPORTS,
    ConfigurationError,
    MissingCredentialError,
    ModelConfig,
    ProviderPreflightConfig,
    SchemaError,
    validate_scan_batch_action,
)
from framework.siliconflow_adapter import (
    PROVIDER_ERROR_CLASSIFICATION_VERSION,
    ProviderCallError,
    SiliconFlowChatTransport,
    classify_provider_exception,
)
from scripts.provider_preflight import (
    ROOT as PREFLIGHT_ROOT,
    _resolve_output_path,
    preflight_one,
    run_preflight,
)


SECRET = "sk" + "-THIS-MUST-NEVER-BE-PERSISTED-123456"


def _status_error(
    status_code: int,
    *,
    code: str | int | None = None,
    message: str = "provider failure",
) -> openai.APIStatusError:
    request = httpx.Request(
        "POST",
        "https://provider.invalid/v1/chat/completions",
    )
    response = httpx.Response(status_code, request=request)
    error: dict[str, object] = {"message": message}
    if code is not None:
        error["code"] = code
    return openai.APIStatusError(
        "raw exception text must not be reflected",
        response=response,
        body={"error": error},
    )


def _response(
    model: str,
    *,
    content: str = '{"ok":true}',
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=[])
            )
        ],
        usage=usage,
    )


class CaptureTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def test_v2_formal_and_preflight_configs_are_frozen() -> None:
    assert (
        ACTIVE_PREFLIGHT_REPORT
        == "reports/PROVIDER_PREFLIGHT_RECHARGE_01.json"
    )
    assert ACTIVE_REPLAY_REPORT == "reports/replay_validation_recharge_01.json"
    assert FORMAL_CAMPAIGN_ID == "formal_campaign_v2_recharge_01"
    assert FORMAL_VALIDATION_REPORTS == {
        label: f"reports/{label}_formal_validation_recharge_01.json"
        for label in FORMAL_MODELS
    }
    for label, model in FORMAL_MODELS.items():
        formal = ModelConfig.for_label(label)
        assert formal.model == model
        assert formal.temperature == 0
        assert formal.max_tokens == 384
        assert formal.max_provider_turns == 5
        assert formal.max_retries == 3
        assert formal.sdk_max_retries == 0
        assert formal.context_mode == "canonical_state_v2"
        assert formal.fallback_allowed is False

        preflight = ProviderPreflightConfig.for_label(label)
        assert preflight.model == model
        assert preflight.temperature == 0
        assert preflight.max_tokens == 24
        assert preflight.max_provider_turns == 1
        assert preflight.native_xtb_calls == 0
        assert preflight.formal_trajectory is False
        assert preflight.sdk_max_retries == 0
        assert preflight.fallback_allowed is False


def test_v2_schema_accepts_only_scan_batch_and_terminal_finalize() -> None:
    initial = validate_scan_batch_action(
        {
            "action": "run_scan_batch",
            "angles_deg": [0, 60, 120, 180, 240, 300],
            "finalize": False,
            "confidence": None,
            "rationale": "Broad periodic coverage.",
        }
    )
    assert initial["angles_deg"] == [0, 60, 120, 180, 240, 300]
    assert initial["finalize"] is False

    terminal = validate_scan_batch_action(
        {
            "action": "run_scan_batch",
            "angles_deg": [85, 95],
            "finalize": True,
            "confidence": "high",
            "rationale": "Refine the final basin and finalize.",
        }
    )
    assert terminal["finalize"] is True
    assert terminal["confidence"] == "high"


def test_v2_schema_rejects_manual_profile_extrema_and_submit_final() -> None:
    manual_submission = {
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "final_status": "completed",
        "profile_id": "profile_model_selected",
        "global_minimum_deg": 45,
        "local_minima_deg": [45, 135],
        "local_maxima_deg": [0, 90],
        "forward_barrier_kcal_mol": 1.5,
        "reverse_barrier_kcal_mol": 3.8,
        "confidence": 0.9,
        "rationale": "manual final",
    }
    with pytest.raises(SchemaError):
        validate_scan_batch_action(manual_submission)
    with pytest.raises(SchemaError):
        schema_module.validate_submission(manual_submission)
    assert "validate_submission" not in schema_module.__all__

    with pytest.raises(SchemaError):
        validate_scan_batch_action(
            {
                "action": "run_scan_batch",
                "angles_deg": [85, 95],
                "finalize": True,
                "confidence": "high",
                "rationale": "terminal",
                "profile_id": "forbidden-model-profile",
            }
        )


@pytest.mark.parametrize(
    "action",
    [
        {
            "action": "run_scan_batch",
            "angles_deg": [0, 0],
            "finalize": False,
            "confidence": None,
            "rationale": "duplicate",
        },
        {
            "action": "run_scan_batch",
            "angles_deg": [7],
            "finalize": False,
            "confidence": None,
            "rationale": "off grid",
        },
        {
            "action": "run_scan_batch",
            "angles_deg": [360],
            "finalize": False,
            "confidence": None,
            "rationale": "outside range",
        },
        {
            "action": "run_scan_batch",
            "angles_deg": [90],
            "finalize": False,
            "confidence": "high",
            "rationale": "early confidence",
        },
        {
            "action": "run_scan_batch",
            "angles_deg": [90],
            "finalize": True,
            "confidence": None,
            "rationale": "missing terminal confidence",
        },
    ],
)
def test_v2_scan_batch_schema_rejects_invalid_actions(
    action: dict[str, object],
) -> None:
    with pytest.raises(SchemaError):
        validate_scan_batch_action(action)


def test_model_config_rejects_alias_retry_and_sampling_overrides() -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(label="kimi", model="alias-or-fallback")
    with pytest.raises(ConfigurationError):
        ModelConfig(
            label="kimi",
            model=FORMAL_MODELS["kimi"],
            temperature=0.1,
        )
    with pytest.raises(ConfigurationError):
        ModelConfig(
            label="kimi",
            model=FORMAL_MODELS["kimi"],
            max_tokens=385,
        )
    with pytest.raises(ConfigurationError):
        ModelConfig(
            label="kimi",
            model=FORMAL_MODELS["kimi"],
            max_retries=4,
        )
    with pytest.raises(ConfigurationError):
        ModelConfig(
            label="kimi",
            model=FORMAL_MODELS["kimi"],
            fallback_allowed=True,
        )


def test_transport_fixes_model_sampling_and_disables_sdk_retries() -> None:
    client_kwargs: dict[str, object] = {}
    request_kwargs: dict[str, object] = {}

    class Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            request_kwargs.update(kwargs)
            return _response(FORMAL_MODELS["deepseek"])

    class Client:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=Completions())

    transport = SiliconFlowChatTransport(
        ModelConfig.for_label("deepseek"),
        "placeholder-key",
        client_factory=Client,
    )
    transport.complete(
        messages=[{"role": "system", "content": "compact state"}],
        tools=[],
        timeout_seconds=10,
    )
    assert client_kwargs["max_retries"] == 0
    assert request_kwargs["model"] == FORMAL_MODELS["deepseek"]
    assert request_kwargs["temperature"] == 0
    assert request_kwargs["max_tokens"] == 384
    assert request_kwargs["stream"] is False
    assert "fallback" not in json.dumps(request_kwargs).casefold()


def test_transport_never_reflects_or_stores_raw_provider_exception() -> None:
    class Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            raise RuntimeError(SECRET)

    class Client:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    transport = SiliconFlowChatTransport(
        ModelConfig.for_label("glm"),
        "placeholder-key",
        client_factory=Client,
    )
    with pytest.raises(ProviderCallError) as caught:
        transport.complete(messages=[], tools=[], timeout_seconds=1)
    serialized = json.dumps(caught.value.safe)
    assert SECRET not in serialized
    assert SECRET not in str(caught.value)
    assert caught.value.safe["status"] == "UNKNOWN_PROVIDER_ERROR"
    assert caught.value.safe["raw_response_persisted"] is False


def test_preflight_transport_request_is_fixed_to_24_tokens_and_json() -> None:
    request_kwargs: dict[str, object] = {}

    class Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            request_kwargs.update(kwargs)
            return _response(FORMAL_MODELS["kimi"])

    class Client:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=Completions())

    transport = SiliconFlowChatTransport(
        ProviderPreflightConfig.for_label("kimi"),
        "placeholder-key",
        client_factory=Client,
    )
    transport.complete(
        messages=[{"role": "user", "content": "preflight"}],
        tools=[],
        timeout_seconds=10,
        response_format={"type": "json_object"},
    )
    assert request_kwargs["model"] == FORMAL_MODELS["kimi"]
    assert request_kwargs["temperature"] == 0
    assert request_kwargs["max_tokens"] == 24
    assert request_kwargs["response_format"] == {"type": "json_object"}
    assert "tools" not in request_kwargs


@pytest.mark.parametrize(
    ("exception", "category", "http_status"),
    [
        (
            _status_error(429, code="rate_limit", message="Too many requests"),
            "RATE_LIMITED",
            429,
        ),
        (
            _status_error(500, message="Service unavailable"),
            "PROVIDER_OVERLOADED",
            500,
        ),
        (
            _status_error(502, message="Bad gateway"),
            "PROVIDER_OVERLOADED",
            502,
        ),
        (
            _status_error(503, message="Overloaded"),
            "PROVIDER_OVERLOADED",
            503,
        ),
        (
            _status_error(504, message="Gateway timeout"),
            "PROVIDER_OVERLOADED",
            504,
        ),
        (
            _status_error(408, message="Request timeout"),
            "READ_TIMEOUT",
            408,
        ),
        (
            openai.APIConnectionError(
                message="connection failed",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "NETWORK_CONNECTION_FAILED",
            None,
        ),
        (
            httpx.ConnectTimeout(
                "connect timeout",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "NETWORK_CONNECTION_FAILED",
            None,
        ),
        (
            httpx.ReadTimeout(
                "read timeout",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "READ_TIMEOUT",
            None,
        ),
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://provider.invalid")
            ),
            "READ_TIMEOUT",
            None,
        ),
    ],
)
def test_transient_error_classification(
    exception: BaseException,
    category: str,
    http_status: int | None,
) -> None:
    safe = classify_provider_exception(exception)
    assert safe["status"] == category
    assert safe["error_category"] == category
    assert safe["http_status"] == http_status
    assert safe["retryable"] is True
    assert safe["classification_version"] == "provider-error-classification/2.0"
    assert safe["raw_response_persisted"] is False


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (
            _status_error(
                402,
                code="30001",
                message="Sorry, your account balance is insufficient",
            ),
            "INSUFFICIENT_BALANCE",
        ),
        (
            _status_error(
                403,
                code="insufficient_balance",
                message="Account balance is insufficient; please recharge.",
            ),
            "INSUFFICIENT_BALANCE",
        ),
        (
            _status_error(403, message="账户余额不足，请充值"),
            "INSUFFICIENT_BALANCE",
        ),
        (
            _status_error(
                403,
                code="model_permission_denied",
                message="No permission to access this model",
            ),
            "MODEL_PERMISSION_DENIED",
        ),
        (
            _status_error(403, message="opaque forbidden"),
            "MODEL_PERMISSION_DENIED",
        ),
        (
            _status_error(403, message="invalid API key"),
            "AUTHENTICATION_FAILED",
        ),
        (
            _status_error(
                403,
                message=f"invalid API key authorization: Bearer {SECRET}",
            ),
            "AUTHENTICATION_FAILED",
        ),
        (
            _status_error(403, message="实名认证 required"),
            "AUTHENTICATION_FAILED",
        ),
        (
            _status_error(401, message="unauthorized"),
            "AUTHENTICATION_FAILED",
        ),
        (
            _status_error(404, message="model not found"),
            "MODEL_NOT_FOUND",
        ),
        (
            _status_error(400, message="bad request"),
            "UNKNOWN_PROVIDER_ERROR",
        ),
        (RuntimeError("unknown raw exception"), "UNKNOWN_PROVIDER_ERROR"),
    ],
)
def test_nontransient_error_classification(
    exception: BaseException, category: str
) -> None:
    safe = classify_provider_exception(exception)
    assert safe["status"] == category
    assert safe["retryable"] is False


def test_provider_error_projection_redacts_secrets_and_never_keeps_raw() -> None:
    safe = classify_provider_exception(
        _status_error(
            403,
            code=SECRET,
            message=f"authorization: Bearer {SECRET}",
        )
    )
    serialized = json.dumps(safe, ensure_ascii=False)
    assert SECRET not in serialized
    assert safe["provider_error_code"] is None
    assert safe["sanitized_provider_message"] == "[REDACTED_PROVIDER_MESSAGE]"
    assert safe["message_reflected"] is False
    assert safe["raw_response_persisted"] is False
    assert safe["response_headers_persisted"] is False
    assert safe["request_persisted"] is False

    unknown = classify_provider_exception(RuntimeError(SECRET))
    assert SECRET not in json.dumps(unknown)
    assert unknown["sanitized_provider_message"] == "Provider request failed."


def test_preflight_one_uses_24_tokens_one_turn_and_no_tools_or_native() -> None:
    capture = CaptureTransport(
        _response(
            FORMAL_MODELS["glm"],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=4,
                total_tokens=16,
            ),
        )
    )
    seen_config: list[ProviderPreflightConfig] = []

    def factory(config: ProviderPreflightConfig, api_key: str) -> CaptureTransport:
        seen_config.append(config)
        assert api_key == "placeholder"
        return capture

    result = preflight_one(
        "glm",
        api_key="placeholder",
        transport_factory=factory,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert result["status"] == "PASS"
    assert result["actual_model"] == FORMAL_MODELS["glm"]
    assert result["exact_identity"] is True
    assert result["structured_json_valid"] is True
    assert result["native_xtb_calls"] == 0
    assert result["formal_trajectory"] is False
    assert result["raw_response_persisted"] is False
    assert result["usage"]["total_tokens"] == 16
    assert len(capture.calls) == 1
    request = capture.calls[0]
    assert request["tools"] == []
    assert request["response_format"] == {"type": "json_object"}
    assert seen_config[0].max_tokens == 24
    assert seen_config[0].temperature == 0


def test_preflight_identity_and_json_protocol_fail_closed() -> None:
    mismatch = CaptureTransport(_response("fallback/model"))
    identity = preflight_one(
        "kimi",
        api_key="placeholder",
        transport_factory=lambda config, key: mismatch,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert identity["status"] == "FAIL"
    assert identity["error_category"] == "IDENTITY_MISMATCH"
    assert identity["fallback_used"] is True

    malformed = CaptureTransport(
        _response(FORMAL_MODELS["kimi"], content="not JSON")
    )
    protocol = preflight_one(
        "kimi",
        api_key="placeholder",
        transport_factory=lambda config, key: malformed,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert protocol["status"] == "FAIL"
    assert protocol["error_category"] == "PROTOCOL_VIOLATION"


def test_preflight_does_not_retry_even_a_transient_error() -> None:
    transient = CaptureTransport(
        ProviderCallError(
            classify_provider_exception(
                _status_error(503, message="provider overloaded")
            )
        )
    )
    result = preflight_one(
        "deepseek",
        api_key="placeholder",
        transport_factory=lambda config, key: transient,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert len(transient.calls) == 1
    assert result["status"] == "BLOCKED"
    assert result["error_category"] == "PROVIDER_OVERLOADED"
    assert result["provider_turns"] == 1


def test_preflight_error_receipt_is_safe_and_stops_campaign_gate() -> None:
    calls: list[str] = []

    def factory(config: ProviderPreflightConfig, api_key: str) -> CaptureTransport:
        calls.append(config.label)
        if config.label == "deepseek":
            error = ProviderCallError(
                classify_provider_exception(
                    _status_error(
                        403,
                        code="insufficient_balance",
                        message="Insufficient balance; please recharge.",
                    )
                )
            )
            return CaptureTransport(error)
        return CaptureTransport(_response(config.model))

    ticks = iter(
        [
            "2026-07-27T00:00:00Z",
            "2026-07-27T00:00:01Z",
            "2026-07-27T00:00:02Z",
            "2026-07-27T00:00:03Z",
            "2026-07-27T00:00:04Z",
        ]
    )
    report = run_preflight(
        credential_loader=lambda root: "placeholder",
        transport_factory=factory,
        now=lambda: next(ticks),
    )
    assert calls == ["kimi", "deepseek", "glm"]
    assert report["status"] == "BLOCKED"
    assert report["formal_campaign_allowed"] is False
    assert report["models"]["deepseek"]["error_category"] == (
        "INSUFFICIENT_BALANCE"
    )
    assert report["models"]["deepseek"]["http_status"] == 403
    assert report["raw_provider_responses_persisted"] is False
    assert SECRET not in json.dumps(report)
    assert report["provider_error_classification_version"] == (
        PROVIDER_ERROR_CLASSIFICATION_VERSION
    )


def test_preflight_rewhitelists_injected_error_metadata() -> None:
    malicious = ProviderCallError(
        {
            "status": "MODEL_PERMISSION_DENIED",
            "error_category": "MODEL_PERMISSION_DENIED",
            "http_status": 403,
            "provider_error_code": SECRET,
            "sanitized_provider_message": SECRET,
            "retryable": False,
            "raw_response_persisted": False,
        }
    )
    result = preflight_one(
        "kimi",
        api_key="placeholder",
        transport_factory=lambda config, key: CaptureTransport(malicious),
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert SECRET not in json.dumps(result)
    assert result["provider_error_code"] is None
    assert result["sanitized_provider_message"] == (
        "[REDACTED_PROVIDER_MESSAGE]"
    )


def test_preflight_report_path_cannot_escape_project_reports() -> None:
    allowed = _resolve_output_path(PREFLIGHT_ROOT / "reports" / "receipt.json")
    assert allowed == (PREFLIGHT_ROOT / "reports" / "receipt.json").resolve()
    with pytest.raises(ValueError):
        _resolve_output_path(PREFLIGHT_ROOT.parent / "outside.json")


def test_successful_three_model_preflight_allows_formal_campaign() -> None:
    calls: list[str] = []

    def factory(config: ProviderPreflightConfig, api_key: str) -> CaptureTransport:
        calls.append(config.model)
        return CaptureTransport(_response(config.model))

    report = run_preflight(
        credential_loader=lambda root: "placeholder",
        transport_factory=factory,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert calls == list(FORMAL_MODELS.values())
    assert report["status"] == "PASS"
    assert report["formal_campaign_allowed"] is True
    assert all(
        item["provider_turns"] == 1
        and item["native_xtb_calls"] == 0
        and item["formal_trajectory"] is False
        for item in report["models"].values()
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert '{"ok":true}' not in serialized
    assert "Connectivity and identity preflight" not in serialized


def test_missing_credential_blocks_all_preflights_without_provider_calls() -> None:
    def missing(root: object) -> str:
        raise MissingCredentialError("MISSING_GJLD_API_KEY")

    def forbidden_factory(
        config: ProviderPreflightConfig, api_key: str
    ) -> CaptureTransport:
        raise AssertionError("provider must not be called without credential")

    report = run_preflight(
        credential_loader=missing,
        transport_factory=forbidden_factory,
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert report["status"] == "BLOCKED"
    assert report["formal_campaign_allowed"] is False
    assert all(
        item["error_category"] == "AUTHENTICATION_FAILED"
        and item["provider_turns"] == 0
        and item["successful_provider_turns"] == 0
        for item in report["models"].values()
    )
