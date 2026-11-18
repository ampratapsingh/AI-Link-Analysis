import asyncio
import json
import logging
import re
from time import perf_counter

from anthropic import (
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    omit,
    transform_schema,
)
from pydantic import ValidationError

from app.config import Settings
from app.providers.base import LLMConfigurationError, LLMOutputError, LLMProviderError
from app.schemas import GeneratedAnalysis

logger = logging.getLogger(__name__)


def _build_analysis_schema() -> dict[str, object]:
    schema = transform_schema(GeneratedAnalysis)
    definitions = schema["$defs"]
    competitor = definitions["Competitor"]
    comparison = definitions["CompetitorAnalysis"]

    # Required nullable values avoid optional grammar branches without losing any fields.
    for definition in definitions.values():
        if "properties" in definition:
            definition["required"] = list(definition["properties"])
    # The application sets the real research status after generation.
    comparison["properties"]["research_status"] = {"type": "null"}

    # Keep the URL grammar simple inside nested arrays. Pydantic still enforces HttpUrl.
    url_schemas = (
        competitor["properties"]["website"]["anyOf"][0],
        definitions["Evidence"]["properties"]["url"],
    )
    for url_schema in url_schemas:
        del url_schema["format"]
        url_schema["description"] = "An absolute HTTP or HTTPS URL."
    return schema


_ANALYSIS_SCHEMA = _build_analysis_schema()
_ANALYSIS_SECTIONS = frozenset(_ANALYSIS_SCHEMA["$defs"]["GeneratedSections"]["properties"])
_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n[ \t]*```", re.IGNORECASE | re.DOTALL)
_SAFE_VALIDATION_TYPES = {"json_invalid", "missing", "literal_error", "enum", "too_short", "too_long", "extra_forbidden"}
_JSON_FALLBACK_INSTRUCTIONS = (
    "\n\nReturn exactly one compact JSON object matching this schema. "
    "Do not include Markdown fences, comments, or surrounding prose. "
    "Use short phrases or single sentences for generated text. Keep open-ended recommendation "
    "and audience lists to at most three distinct entries each. Keep all required fields and "
    "exactly five search queries. Use empty arrays for unsupported lists and null for unavailable "
    "nullable fields. Preserve required verbatim evidence excerpts, escaping quotes as JSON.\n"
    + json.dumps(_ANALYSIS_SCHEMA, separators=(",", ":"))
)
_SCHEMA_COMPLEXITY_ERROR = "Structured output schema is too complex to compile."
_SCHEMA_FORMAT_ERROR = "Anthropic reported a structured output schema error."

_ERROR_MESSAGES = {
    "authentication_error": "Anthropic rejected authentication.",
    "billing_error": "Anthropic rejected the request for a billing issue.",
    "permission_error": "Anthropic denied access to the requested resource.",
    "not_found_error": "Anthropic could not find the requested resource.",
    "invalid_request_error": "Anthropic rejected the request parameters.",
    "conflict_error": "Anthropic reported a resource conflict.",
    "request_too_large": "Anthropic rejected the request size.",
    "rate_limit_error": "Anthropic rate limited the request.",
    "api_error": "Anthropic reported an internal error.",
    "timeout_error": "Anthropic reported an upstream timeout.",
    "overloaded_error": "Anthropic reported an overload.",
}


def _http_error_details(exc: APIStatusError) -> tuple[str, str]:
    # Upstream messages can echo arbitrary input. Classify their content without logging it.
    body = exc.body
    detail = body.get("error", body) if isinstance(body, dict) else None
    error_type = detail.get("type") if isinstance(detail, dict) else None
    if not isinstance(error_type, str) or error_type not in _ERROR_MESSAGES:
        error_type = "unknown"
    summary = _ERROR_MESSAGES.get(error_type, "Anthropic returned an unrecognized HTTP error.")
    message = detail.get("message") if isinstance(detail, dict) else None
    if isinstance(message, str):
        message = message[:1024].lower()
        if exc.status_code == 400:
            if "schema is too complex for compilation" in message:
                summary = _SCHEMA_COMPLEXITY_ERROR
            elif "output_config.format" in message or "schema" in message:
                summary = _SCHEMA_FORMAT_ERROR
            elif "max_tokens" in message:
                summary = "Anthropic reported a max_tokens error."
            elif "thinking" in message or "output_config.effort" in message:
                summary = "Anthropic reported a thinking or effort configuration error."
        elif exc.status_code == 404 and message.startswith("model:"):
            summary = "The requested model was not found for this API client."
    return error_type, summary


def _log_http_error(exc: APIStatusError, error_type: str, summary: str) -> None:
    logger.warning(
        "Anthropic HTTP error status=%d error_type=%s message=%s",
        exc.status_code,
        error_type,
        summary,
    )


def _log_validation_error(exc: ValidationError, output_mode: str) -> None:
    errors = exc.errors(include_input=False, include_context=False, include_url=False)
    problems = set()
    for error in errors:
        location = error["loc"]
        section = location[0] if location[:1] in {("brand",), ("analysis",)} else "root"
        if section == "analysis" and len(location) > 1 and location[1] in _ANALYSIS_SECTIONS:
            section = location[1]
        error_type = error["type"]
        if error_type not in _SAFE_VALIDATION_TYPES:
            error_type = "invalid_value"
        problems.add(f"{section}:{error_type}")
    logger.warning(
        "Claude analysis validation failed output_mode=%s error_count=%d problems=%s",
        output_mode,
        len(errors),
        ",".join(sorted(problems)[:6]),
    )


def _parse_fallback_json(raw_json: str) -> GeneratedAnalysis:
    try:
        return GeneratedAnalysis.model_validate_json(raw_json)
    except ValidationError as exc:
        invalid = set()
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            location = error["loc"]
            if location[:2] != ("analysis", "top_services") or len(location) < 3:
                raise
            index = location[2]
            if not isinstance(index, int):
                raise
            invalid.add(index)
        # Recover valid siblings, then run the normal schema and service evidence checks.
        payload = json.loads(raw_json)
        candidates = payload["analysis"]["top_services"]
        remaining = [item for index, item in enumerate(candidates) if index not in invalid]
        if not remaining:
            raise
        payload["analysis"]["top_services"] = remaining
        result = GeneratedAnalysis.model_validate(payload)
        logger.warning("Omitted invalid Claude service candidates count=%d", len(invalid))
        return result


class AnthropicAnalysisProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.anthropic_model.strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]*", self._model):
            raise LLMConfigurationError(
                "invalid_model", "Set ANTHROPIC_MODEL to a valid Claude model ID."
            )
        secret = settings.anthropic_api_key
        if not secret or not secret.get_secret_value().strip():
            raise LLMConfigurationError("missing_api_key", "Set ANTHROPIC_API_KEY to run analysis.")
        self._api_key = secret

    async def generate(self, *, system_prompt: str, content: str) -> GeneratedAnalysis:
        started = perf_counter()
        input_tokens = output_tokens = None
        output_mode = "structured"
        max_output_tokens = self._settings.llm_max_output_tokens
        try:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                async with AsyncAnthropic(
                    api_key=self._api_key.get_secret_value().strip(),
                    timeout=self._settings.llm_timeout_seconds,
                    max_retries=1,
                ) as client:
                    request = {
                        "model": self._model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": max_output_tokens,
                        "thinking": {"type": "disabled"} if self._model == "claude-haiku-5-5" else omit,
                    }
                    try:
                        reply = await client.messages.create(
                            **request,
                            system=system_prompt,
                            output_config={"format": {"type": "json_schema", "schema": _ANALYSIS_SCHEMA}},
                        )
                    except BadRequestError as exc:
                        error_type, summary = _http_error_details(exc)
                        if error_type != "invalid_request_error" or summary not in (
                            _SCHEMA_FORMAT_ERROR, _SCHEMA_COMPLEXITY_ERROR
                        ):
                            raise
                        _log_http_error(exc, error_type, summary)
                        logger.warning("Anthropic rejected the schema. Trying prompt-only JSON once.")
                        output_mode = "prompt_json"
                        reply = await client.with_options(max_retries=0).messages.create(
                            **request,
                            system=system_prompt + _JSON_FALLBACK_INSTRUCTIONS,
                        )
                    if reply.usage is not None:
                        if isinstance(reply.usage.input_tokens, int):
                            input_tokens = reply.usage.input_tokens
                        if isinstance(reply.usage.output_tokens, int):
                            output_tokens = reply.usage.output_tokens
                    # Check for refusal and truncation before trying to parse non-JSON text.
                    if reply.stop_reason == "refusal":
                        raise LLMOutputError("refused", "Claude declined to generate the analysis.")
                    if reply.stop_reason == "max_tokens":
                        raise LLMOutputError(
                            "output_truncated", "Claude reached the configured output token limit."
                        )
                    if reply.stop_reason != "end_turn":
                        raise LLMOutputError("incomplete_output", "Claude did not complete the analysis.")
                    text = [block.text for block in reply.content if block.type == "text"]
                    if not text or not any(part.strip() for part in text):
                        raise LLMOutputError("missing_output", "Claude returned no structured analysis.")
                    if output_mode == "structured" and len(text) != 1:
                        logger.warning("Claude structured response had multiple text blocks count=%d", len(text))
                        raise LLMOutputError("invalid_output", "Claude returned multiple text blocks.")
                    raw_json = "".join(text).strip()
                    if output_mode == "prompt_json":
                        fenced = _JSON_FENCE.fullmatch(raw_json)
                        if fenced:
                            raw_json = fenced.group(1)
                        return _parse_fallback_json(raw_json)
                    return GeneratedAnalysis.model_validate_json(raw_json)
        except (TimeoutError, APITimeoutError):
            raise LLMProviderError("timeout", "The Claude analysis request timed out.") from None
        except APIStatusError as exc:
            _log_http_error(exc, *_http_error_details(exc))
            if isinstance(exc, RateLimitError):
                raise LLMProviderError("rate_limited", "Anthropic rate limited the analysis request.") from None
            if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                raise LLMConfigurationError(
                    "invalid_credentials", "Check ANTHROPIC_API_KEY and the project's model permissions."
                ) from None
            if isinstance(exc, (BadRequestError, NotFoundError)):
                raise LLMConfigurationError(
                    "invalid_model_configuration",
                    "Anthropic rejected the request. Check ANTHROPIC_MODEL access, Structured Outputs "
                    "support, and LLM_MAX_OUTPUT_TOKENS.",
                ) from None
            raise LLMProviderError("provider_error", "The Anthropic analysis request failed.") from None
        except APIError:
            raise LLMProviderError("provider_error", "The Anthropic analysis request failed.") from None
        except ValidationError as exc:
            _log_validation_error(exc, output_mode)
            raise LLMOutputError("invalid_output", "Claude returned an invalid structured analysis.") from None
        except (ValueError, TypeError, AttributeError):
            logger.warning("Claude response could not be decoded output_mode=%s", output_mode)
            raise LLMOutputError("invalid_output", "Claude returned an invalid structured analysis.") from None
        finally:
            logger.info(
                "Claude analysis duration_ms=%d input_tokens=%s output_tokens=%s output_mode=%s max_output_tokens=%d",
                (perf_counter() - started) * 1000,
                input_tokens,
                output_tokens,
                output_mode,
                max_output_tokens,
            )
