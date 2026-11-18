import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator


class Settings(BaseModel):
    app_name: str = "AI-Powered Website Link Analysis API"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=4000, ge=1, le=65535)
    llm_provider: str = "anthropic"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    llm_max_output_tokens: int = Field(default=3000, ge=16)
    llm_max_content_chars: int = Field(default=12000, ge=256)
    search_provider: str = "tavily"
    search_api_key: SecretStr | None = None
    search_max_results: int = Field(default=5, ge=1, le=20)
    search_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    website_request_timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    max_website_response_size: int = Field(default=2_097_152, gt=0)
    max_website_redirects: int = Field(default=3, ge=0)
    website_user_agent: str = "AI-Link-Analysis/0.1"
    analysis_max_body_bytes: int = Field(default=32_768, gt=0)
    analysis_body_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    analysis_max_in_flight: int = Field(default=2, gt=0)
    analysis_requests_per_hour: int = Field(default=20, gt=0)

    @property
    def is_production(self) -> bool:
        return self.environment.strip().casefold() == "production"

    @field_validator("host", "environment")
    @classmethod
    def nonempty_setting(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("website_user_agent")
    @classmethod
    def valid_user_agent(cls, value: str) -> str:
        if not value.strip() or any(not 32 <= ord(char) <= 126 for char in value):
            raise ValueError("must be nonempty printable ASCII")
        return value.strip()


def load_settings() -> Settings:
    if os.environ.get("ENVIRONMENT", "").strip().casefold() != "production":
        load_dotenv()
    values = {
        name: os.environ[name.upper()]
        for name in Settings.model_fields
        if name.upper() in os.environ
    }
    for key in ("anthropic_api_key", "openai_api_key", "search_api_key"):
        if values.get(key) == "":
            values.pop(key)
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]).upper() for error in exc.errors())
        raise RuntimeError(f"Invalid settings: {fields}. Check the values in .env.example.") from None
