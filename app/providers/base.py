from typing import Protocol

from app.schemas import GeneratedAnalysis


class LLMError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LLMConfigurationError(LLMError):
    """Analysis settings or provider access are invalid."""


class LLMProviderError(LLMError):
    """The provider could not complete the API request."""


class LLMOutputError(LLMError):
    """The model did not produce usable structured analysis."""


class AnalysisProvider(Protocol):
    async def generate(self, *, system_prompt: str, content: str) -> GeneratedAnalysis:
        """Generate the analysis payload from trusted instructions and separate page data."""
