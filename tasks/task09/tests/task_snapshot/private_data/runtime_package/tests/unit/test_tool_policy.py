from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest

import tools.tool_registry as registry
from framework.schemas import ConfigurationError, ModelConfig
from framework.redaction import safe_provider_error
from framework.trusted_final import _first_sampling_turn_new_angles
from framework.siliconflow_adapter import (
    ProviderCallError,
    SiliconFlowChatTransport,
    classify_provider_exception,
)
from tools.science import canonical_json_bytes, sha256_file
from tools.tool_registry import ToolSession, load_tool_contracts


def _fake_native(*, reference_path: Path, angle_deg: int, attempt_root: Path, timeout_seconds: int):
    attempt_root.mkdir(parents=True, exist_ok=False)
    energy = -10.0 + 0.001 * angle_deg
    (attempt_root / "result.json").write_bytes(canonical_json_bytes({"fake": True}))
    (attempt_root / "attempt_manifest.json").write_bytes(canonical_json_bytes({"fake": True}))
    return {
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": energy,
        "final_angle_deg": float(angle_deg),
        "angle_error_deg": 0.0,
        "structure_validation": {
            "atom_count_ok": True,
            "atom_order_ok": True,
            "composition_ok": True,
            "formal_charge_ok": True,
            "connectivity_ok": True,
        },
        "profile_eligible": True,
        "reference_sha256": sha256_file(reference_path),
        "prepared_input": {"input_sha256": "A" * 64},
        "attempt_manifest_sha256": sha256_file(attempt_root / "attempt_manifest.json"),
        "xtb": {"binary_sha256": "B" * 64},
    }


def test_public_registry_is_exactly_seven_closed_tools() -> None:
    tools = load_tool_contracts()
    assert [tool["function"]["name"] for tool in tools] == [
        "get_case_spec", "run_scan_point", "inspect_scan_attempt", "list_scan_points",
        "fit_periodic_profile", "analyze_profile", "submit_final",
    ]
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in tools)


def test_angle_budget_retry_and_finalization_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)
    illegal = session.dispatch("run_scan_point", {"angle_deg": 7}, assistant_turn=1)
    assert illegal["ok"] is False
    assert illegal["budget"]["unique_scan_angles_used"] == 0
    for angle in range(0, 30, 5):
        assert session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=1
        )["ok"]
    fit = session.dispatch(
        "fit_periodic_profile", {"method": "periodic_cubic"}, assistant_turn=1
    )
    assert session.dispatch(
        "analyze_profile",
        {"profile_id": fit["profile"]["profile_id"]},
        assistant_turn=1,
    )["ok"]
    for turn, angles in (
        (2, (30, 35, 40)),
        (3, (45, 50, 55)),
        (4, (60, 65, 70)),
    ):
        for angle in angles:
            assert session.dispatch(
                "run_scan_point", {"angle_deg": angle}, assistant_turn=turn
            )["ok"]
        fit = session.dispatch(
            "fit_periodic_profile",
            {"method": "periodic_cubic"},
            assistant_turn=turn,
        )
        assert session.dispatch(
            "analyze_profile",
            {"profile_id": fit["profile"]["profile_id"]},
            assistant_turn=turn,
        )["ok"]
    assert session.dispatch(
        "run_scan_point", {"angle_deg": 75}, assistant_turn=5
    )["ok"]
    snapshot = session.budget_snapshot()
    assert snapshot["unique_scan_angles_used"] == 16
    assert snapshot["finalization_only"] is True
    blocked = session.dispatch("run_scan_point", {"angle_deg": 90}, assistant_turn=5)
    assert blocked["status"] == "FINALIZATION_ONLY"
    assert blocked["budget"]["unique_scan_angles_used"] == 16


def test_first_round_six_new_angle_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)
    case = session.dispatch("get_case_spec", {}, assistant_turn=1)
    assert case["ok"] is True
    for angle in [0, 5, 10, 15, 20, 25]:
        assert session.dispatch("run_scan_point", {"angle_deg": angle}, assistant_turn=2)["ok"]
    seventh = session.dispatch("run_scan_point", {"angle_deg": 30}, assistant_turn=2)
    assert seventh["status"] == "FIRST_ROUND_NEW_ANGLE_LIMIT"
    assert seventh["budget"]["unique_scan_angles_used"] == 6
    assert _first_sampling_turn_new_angles(session) == 6


def test_exact_model_configuration_rejects_override() -> None:
    config = ModelConfig.for_label("kimi")
    assert config.model == "Pro/moonshotai/Kimi-K2.6"
    assert config.max_retries == 3
    assert config.temperature == 0
    assert config.max_tokens == 384
    with pytest.raises(ConfigurationError):
        ModelConfig(label="kimi", model="alias-or-fallback")
    with pytest.raises(ConfigurationError):
        ModelConfig(label="kimi", model="Pro/moonshotai/Kimi-K2.6", sdk_max_retries=1)
    with pytest.raises(ConfigurationError):
        ModelConfig(
            label="kimi",
            model="Pro/moonshotai/Kimi-K2.6",
            max_retries=4,
        )


def test_provider_exception_does_not_reflect_secret() -> None:
    marker = "sk" + "-THIS_VALUE_MUST_NEVER_BE_REFLECTED"
    client_kwargs: dict[str, object] = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError(marker)

    class Client:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.chat = type("Chat", (), {"completions": Completions()})()

    adapter = SiliconFlowChatTransport(ModelConfig.for_label("glm"), "placeholder-key", client_factory=Client)
    with pytest.raises(ProviderCallError) as caught:
        adapter.complete(messages=[], tools=[], timeout_seconds=1)
    assert marker not in str(caught.value)
    assert marker not in json.dumps(caught.value.safe)
    assert caught.value.safe["message_reflected"] is False
    assert caught.value.retryable is False
    assert client_kwargs["max_retries"] == 0


def _status_error(status_code: int, marker: str) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(marker, response=response, body={"error": marker})


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        (
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://provider.invalid")
            ),
            "READ_TIMEOUT",
        ),
        (
            openai.APIConnectionError(
                message="secret connection detail",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "NETWORK_CONNECTION_FAILED",
        ),
        (
            httpx.ConnectTimeout(
                "secret connect timeout",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "NETWORK_CONNECTION_FAILED",
        ),
        (
            httpx.ReadTimeout(
                "secret read timeout",
                request=httpx.Request("POST", "https://provider.invalid"),
            ),
            "READ_TIMEOUT",
        ),
        (ConnectionResetError("connection reset"), "NETWORK_CONNECTION_FAILED"),
        (OSError(10054, "WSAECONNRESET"), "NETWORK_CONNECTION_FAILED"),
        (TimeoutError("socket timeout"), "READ_TIMEOUT"),
        (_status_error(429, "provider 429"), "RATE_LIMITED"),
        (_status_error(500, "provider 500"), "PROVIDER_OVERLOADED"),
        (_status_error(502, "provider 502"), "PROVIDER_OVERLOADED"),
        (_status_error(503, "provider 503"), "PROVIDER_OVERLOADED"),
        (_status_error(504, "provider 504"), "PROVIDER_OVERLOADED"),
    ],
)
def test_transient_provider_classification(
    exception: BaseException, expected_status: str
) -> None:
    safe = classify_provider_exception(exception)
    assert safe["retryable"] is True
    assert safe["status"] == expected_status
    assert safe["classification_version"] == "provider-error-classification/2.0"
    assert safe["raw_response_persisted"] is False


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        (_status_error(400, "provider 400"), "UNKNOWN_PROVIDER_ERROR"),
        (_status_error(401, "unauthorized"), "AUTHENTICATION_FAILED"),
        (_status_error(403, "opaque forbidden"), "MODEL_PERMISSION_DENIED"),
        (_status_error(404, "model not found"), "MODEL_NOT_FOUND"),
        (_status_error(409, "provider 409"), "UNKNOWN_PROVIDER_ERROR"),
        (_status_error(422, "provider 422"), "UNKNOWN_PROVIDER_ERROR"),
        (_status_error(501, "provider 501"), "UNKNOWN_PROVIDER_ERROR"),
        (_status_error(505, "provider 505"), "UNKNOWN_PROVIDER_ERROR"),
        (RuntimeError("unknown"), "UNKNOWN_PROVIDER_ERROR"),
    ],
)
def test_nontransient_provider_classification(
    exception: BaseException, expected_status: str
) -> None:
    safe = classify_provider_exception(exception)
    assert safe["retryable"] is False
    assert safe["status"] == expected_status
    assert safe["raw_response_persisted"] is False
