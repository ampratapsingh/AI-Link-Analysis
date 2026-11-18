import asyncio
import json
import logging

import httpx
import pytest

from app.config import Settings, load_settings
from app.providers.anthropic_provider import AnthropicAnalysisProvider
from app.providers.base import LLMConfigurationError, LLMOutputError, LLMProviderError
from app.services.analysis_service import get_analysis_provider

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


@pytest.fixture
def provider():
    return AnthropicAnalysisProvider(Settings(anthropic_api_key="fixture-key", llm_max_output_tokens=1250))


async def test_messages_uses_structured_output_and_logs_only_metrics(
    provider, mock_anthropic, anthropic_response, analysis, caplog
):
    calls = []

    def handler(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "fixture-key"
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "claude-haiku-5-5"
        assert body["max_tokens"] == 1250
        assert body["system"] == "Instructions"
        assert body["messages"] == [{"role": "user", "content": "Private fixture evidence"}]
        assert body["thinking"] == {"type": "disabled"}
        assert "tools" not in body
        output_format = body["output_config"]["format"]
        assert output_format["type"] == "json_schema"
        schema = output_format["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"brand", "analysis"}
        assert set(schema["$defs"]["Analysis"]["required"]) == set(analysis["analysis"])
        return anthropic_response(json.dumps(analysis))

    mock_anthropic(handler)
    with caplog.at_level(logging.INFO, logger="app.providers.anthropic_provider"):
        result = await provider.generate(system_prompt="Instructions", content="Private fixture evidence")

    assert result.brand.name == analysis["brand"]["name"]
    assert len(result.analysis.target_search_queries) == 5
    assert len(calls) == 1
    assert "input_tokens=200 output_tokens=95" in caplog.text
    assert "fixture-key" not in caplog.text and "Private fixture evidence" not in caplog.text


@pytest.mark.parametrize("key", [None, "", " "])
async def test_missing_credentials_do_not_fall_back_to_openai(key):
    with pytest.raises(LLMConfigurationError) as error:
        get_analysis_provider(Settings(anthropic_api_key=key, openai_api_key="fixture-key"))
    assert error.value.code == "missing_api_key"


async def test_environment_config_loads_and_redacts_anthropic_credentials(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    settings = load_settings()
    assert settings.anthropic_api_key is not None and "fixture-key" not in repr(settings)
    assert isinstance(get_analysis_provider(settings), AnthropicAnalysisProvider)


async def test_timeout_covers_entire_anthropic_request(mock_anthropic):
    async def stalled(request):
        await asyncio.Event().wait()

    mock_anthropic(stalled)
    provider = AnthropicAnalysisProvider(
        Settings(anthropic_api_key="fixture-key", llm_timeout_seconds=0.03)
    )
    with pytest.raises(LLMProviderError) as error:
        await provider.generate(system_prompt="Instructions", content="Evidence")
    assert error.value.code == "timeout"


@pytest.mark.parametrize(
    ("status", "upstream_type", "error_type", "code", "attempts"),
    [
        (429, "rate_limit_error", LLMProviderError, "rate_limited", 2),
        (529, "overloaded_error", LLMProviderError, "provider_error", 2),
        (401, "authentication_error", LLMConfigurationError, "invalid_credentials", 1),
        (403, "permission_error", LLMConfigurationError, "invalid_credentials", 1),
        (400, "invalid_request_error", LLMConfigurationError, "invalid_model_configuration", 1),
        (404, "not_found_error", LLMConfigurationError, "invalid_model_configuration", 1),
    ],
)
async def test_provider_errors_are_sanitized_and_retries_bounded(
    provider, mock_anthropic, status, upstream_type, error_type, code, attempts, caplog
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, headers={"retry-after-ms": "1"},
            json={"type": "error", "error": {"type": upstream_type, "message": "sensitive fixture"}},
        )

    mock_anthropic(handler)
    with pytest.raises(error_type) as error:
        await provider.generate(system_prompt="Instructions", content="Evidence")
    assert error.value.code == code
    assert "sensitive fixture" not in str(error.value)
    diagnostic = f"status={status} error_type={upstream_type}"
    assert caplog.text.count(diagnostic) == 1
    assert "sensitive fixture" not in caplog.text
    assert len(calls) == attempts


@pytest.mark.parametrize(
    ("status", "upstream_type", "message", "logged_type", "summary"),
    [
        (400, "invalid_request_error", "Schema is too complex for compilation",
         "invalid_request_error", "Structured output schema is too complex to compile."),
        (400, "invalid_request_error", "output_config.format.schema: invalid",
         "invalid_request_error", "Anthropic reported a structured output schema error."),
        (400, "invalid_request_error", "max_tokens: out of range",
         "invalid_request_error", "Anthropic reported a max_tokens error."),
        (400, "invalid_request_error", "output_config.effort is invalid",
         "invalid_request_error", "Anthropic reported a thinking or effort configuration error."),
        (404, "not_found_error", "model: claude-haiku-5-5",
         "not_found_error", "The requested model was not found for this API client."),
        (400, "fixture-key\nforged entry", "unrecognized message",
         "unknown", "Anthropic returned an unrecognized HTTP error."),
    ],
)
async def test_http_diagnostics_do_not_echo_response_or_request_data(
    provider, mock_anthropic, caplog, status, upstream_type, message, logged_type, summary
):
    sensitive = "fixture-key\nPrivate website content\nPrivate system prompt\nforged entry"
    mock_anthropic(lambda _: httpx.Response(status, json={
        "type": "error",
        "error": {"type": upstream_type, "message": f"{message}: {sensitive}"},
    }))

    with pytest.raises(LLMConfigurationError) as error:
        await provider.generate(system_prompt="Private system prompt", content="Private website content")

    assert error.value.code == "invalid_model_configuration"
    assert f"status={status} error_type={logged_type} message={summary}" in caplog.text
    assert summary not in str(error.value)
    for value in sensitive.splitlines():
        assert value not in caplog.text
        assert value not in str(error.value)


async def test_non_json_http_error_logs_status_without_logging_body(provider, mock_anthropic, caplog):
    mock_anthropic(lambda _: httpx.Response(404, text="private upstream response"))

    with pytest.raises(LLMConfigurationError) as error:
        await provider.generate(system_prompt="Instructions", content="Evidence")

    assert error.value.code == "invalid_model_configuration"
    assert "status=404 error_type=unknown" in caplog.text
    assert "private upstream response" not in caplog.text


@pytest.mark.parametrize(
    ("stop_reason", "text", "code"),
    [
        ("refusal", "Unstructured refusal text", "refused"),
        ("max_tokens", '{"brand":', "output_truncated"),
        ("pause_turn", None, "incomplete_output"),
        ("end_turn", None, "missing_output"),
        ("end_turn", "{not JSON}", "invalid_output"),
        ("end_turn", '{"brand":{}}', "invalid_output"),
    ],
)
async def test_unusable_messages_are_rejected(
    provider, mock_anthropic, anthropic_response, stop_reason, text, code
):
    mock_anthropic(lambda _: anthropic_response(text, stop_reason=stop_reason))
    with pytest.raises(LLMOutputError) as error:
        await provider.generate(system_prompt="Instructions", content="Evidence")
    assert error.value.code == code


async def test_local_pydantic_validation_enforces_query_count(
    provider, mock_anthropic, anthropic_response, analysis
):
    analysis["analysis"]["target_search_queries"].pop()
    mock_anthropic(lambda _: anthropic_response(json.dumps(analysis)))
    with pytest.raises(LLMOutputError) as error:
        await provider.generate(system_prompt="Instructions", content="Evidence")
    assert error.value.code == "invalid_output"
