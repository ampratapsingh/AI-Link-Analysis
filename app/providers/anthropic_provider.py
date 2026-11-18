import asyncio
import logging
import re
from time import perf_counter

from anthropic import (
    APIError,
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
_ANALYSIS_SCHEMA = transform_schema(GeneratedAnalysis)


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
        try:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                async with AsyncAnthropic(
                    api_key=self._api_key.get_secret_value().strip(),
                    timeout=self._settings.llm_timeout_seconds,
                    max_retries=1,
                ) as client:
                    reply = await client.messages.create(
                        model=self._model,
                        system=system_prompt,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=self._settings.llm_max_output_tokens,
                        output_config={"format": {"type": "json_schema", "schema": _ANALYSIS_SCHEMA}},
                        thinking={"type": "disabled"} if self._model == "claude-haiku-5-5" else omit,
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
                    if len(text) != 1:
                        raise LLMOutputError("invalid_output", "Claude returned multiple text blocks.")
                    return GeneratedAnalysis.model_validate_json(text[0])
        except (TimeoutError, APITimeoutError):
            raise LLMProviderError("timeout", "The Claude analysis request timed out.") from None
        except RateLimitError:
            raise LLMProviderError("rate_limited", "Anthropic rate limited the analysis request.") from None
        except (AuthenticationError, PermissionDeniedError):
            raise LLMConfigurationError(
                "invalid_credentials", "Check ANTHROPIC_API_KEY and the project's model permissions."
            ) from None
        except (BadRequestError, NotFoundError):
            raise LLMConfigurationError(
                "invalid_model_configuration",
                "Anthropic rejected the request. Check ANTHROPIC_MODEL access, Structured Outputs "
                "support, and LLM_MAX_OUTPUT_TOKENS.",
            ) from None
        except APIError:
            raise LLMProviderError("provider_error", "The Anthropic analysis request failed.") from None
        except (ValidationError, ValueError, TypeError, AttributeError):
            raise LLMOutputError("invalid_output", "Claude returned an invalid structured analysis.") from None
        finally:
            logger.info(
                "Claude analysis duration_ms=%d input_tokens=%s output_tokens=%s",
                (perf_counter() - started) * 1000,
                input_tokens,
                output_tokens,
            )
