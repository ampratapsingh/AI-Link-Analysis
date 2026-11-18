import asyncio
import logging
import re
from time import perf_counter
from typing import Any

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from app.config import Settings
from app.providers.base import LLMConfigurationError, LLMOutputError, LLMProviderError
from app.schemas import GeneratedAnalysis

logger = logging.getLogger(__name__)


class _OpenAIJsonSchema(GenerateJsonSchema):
    def url_schema(self, schema: core_schema.UrlSchema) -> JsonSchemaValue:
        # OpenAI's schema subset does not accept Pydantic's URI format.
        # The runtime fields still validate as HttpUrl during SDK parsing.
        return {"type": "string"}

    def list_schema(self, schema: core_schema.ListSchema) -> JsonSchemaValue:
        result = super().list_schema(schema)
        # Keep list limits in local validation for models that lack these keywords.
        result.pop("minItems", None)
        result.pop("maxItems", None)
        return result


class _OpenAIAnalysis(GeneratedAnalysis):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> JsonSchemaValue:
        return super().model_json_schema(**{"schema_generator": _OpenAIJsonSchema, **kwargs})


class OpenAIAnalysisProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.openai_model.strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]*", self._model):
            raise LLMConfigurationError(
                "invalid_model", "Set OPENAI_MODEL to a valid OpenAI model ID."
            )
        secret = settings.openai_api_key
        if not secret or not secret.get_secret_value().strip():
            raise LLMConfigurationError("missing_api_key", "Set OPENAI_API_KEY to run analysis.")
        self._api_key = secret

    async def generate(self, *, system_prompt: str, content: str) -> GeneratedAnalysis:
        started = perf_counter()
        input_tokens = output_tokens = None
        try:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                async with AsyncOpenAI(
                    api_key=self._api_key.get_secret_value().strip(),
                    timeout=self._settings.llm_timeout_seconds,
                    max_retries=1,
                ) as client:
                    raw = await client.responses.with_raw_response.parse(
                        model=self._model,
                        instructions=system_prompt,
                        input=[{"role": "user", "content": content}],
                        text_format=_OpenAIAnalysis,
                        max_output_tokens=self._settings.llm_max_output_tokens,
                        store=False,
                    )
                    # An incomplete response can contain truncated JSON. Check its
                    # status before invoking the SDK's Pydantic parser.
                    payload = raw.http_response.json()
                    if not isinstance(payload, dict):
                        raise LLMOutputError("invalid_output", "OpenAI returned an invalid response.")
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        if isinstance(usage.get("input_tokens"), int):
                            input_tokens = usage["input_tokens"]
                        if isinstance(usage.get("output_tokens"), int):
                            output_tokens = usage["output_tokens"]
                    _check_completion(payload)
                    response = raw.parse()
                    if response.output_parsed is None:
                        raise LLMOutputError(
                            "missing_output", "OpenAI returned no structured analysis."
                        )
                    return response.output_parsed
        except (TimeoutError, APITimeoutError):
            raise LLMProviderError("timeout", "The OpenAI analysis request timed out.") from None
        except RateLimitError:
            raise LLMProviderError(
                "rate_limited", "OpenAI rate limited the analysis request."
            ) from None
        except (AuthenticationError, PermissionDeniedError):
            raise LLMConfigurationError(
                "invalid_credentials", "Check OPENAI_API_KEY and the project's model permissions."
            ) from None
        except (BadRequestError, NotFoundError):
            raise LLMConfigurationError(
                "invalid_model_configuration",
                "OpenAI rejected the request. Check OPENAI_MODEL access, Structured Outputs "
                "support, and LLM_MAX_OUTPUT_TOKENS.",
            ) from None
        except APIError:
            raise LLMProviderError("provider_error", "The OpenAI analysis request failed.") from None
        except (ValidationError, ValueError, TypeError, AttributeError):
            raise LLMOutputError(
                "invalid_output", "OpenAI returned an invalid structured analysis."
            ) from None
        finally:
            logger.info(
                "OpenAI analysis duration_ms=%d input_tokens=%s output_tokens=%s",
                (perf_counter() - started) * 1000,
                input_tokens,
                output_tokens,
            )


def _check_completion(payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if status == "incomplete":
        details = payload.get("incomplete_details")
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            raise LLMOutputError(
                "output_truncated", "OpenAI reached the configured output token limit."
            )
        raise LLMOutputError("incomplete_output", "OpenAI returned an incomplete analysis.")
    if status != "completed":
        raise LLMOutputError("incomplete_output", "OpenAI did not complete the analysis.")

    output = payload.get("output", [])
    if not isinstance(output, list):
        raise LLMOutputError("invalid_output", "OpenAI returned invalid response items.")
    text_count = 0
    for item in output:
        if not isinstance(item, dict) or item.get("type") not in {"message", "reasoning"}:
            raise LLMOutputError("invalid_output", "OpenAI returned an unexpected response item.")
        if item["type"] != "message":
            continue
        if item.get("status") != "completed":
            raise LLMOutputError("incomplete_output", "OpenAI returned an incomplete message.")
        content = item.get("content")
        if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
            raise LLMOutputError("invalid_output", "OpenAI returned invalid message content.")
        if any(part.get("type") == "refusal" for part in content):
            raise LLMOutputError("refused", "OpenAI declined to generate the analysis.")
        text_count += sum(part.get("type") == "output_text" for part in content)
    if text_count > 1:
        raise LLMOutputError("invalid_output", "OpenAI returned multiple structured outputs.")
