import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.providers.base import LLMConfigurationError, LLMError, LLMOutputError, LLMProviderError
from app.schemas import (
    HealthResponse,
    LinkAnalysisErrorResponse,
    LinkAnalysisRequest,
    LinkAnalysisResponse,
)
from app.services.analysis_service import AnalysisInternalError, analyze_website
from app.services.url_safety import UnsafeUrlError, WebsiteDnsError
from app.services.website_extractor import WebsiteFetchError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    summary="Check API health",
    description="Report whether the API process is responding.",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/link-analysis",
    summary="Analyze a website",
    description=(
        "Fetch a public HTTP or HTTPS website and return its brand profile and five "
        "analysis sections. The website field identifies the final page after redirects."
    ),
    responses={
        408: {"model": LinkAnalysisErrorResponse, "description": "Timed out reading the request body."},
        413: {"model": LinkAnalysisErrorResponse, "description": "Request body exceeds the limit."},
        422: {"model": LinkAnalysisErrorResponse, "description": "Invalid request or unsafe URL."},
        429: {"model": LinkAnalysisErrorResponse, "description": "Shared analysis rate limit reached."},
        502: {"model": LinkAnalysisErrorResponse, "description": "Unusable website or model response."},
        503: {"model": LinkAnalysisErrorResponse, "description": "Analysis capacity or provider unavailable."},
        504: {"model": LinkAnalysisErrorResponse, "description": "Website or analysis provider timed out."},
        500: {"model": LinkAnalysisErrorResponse, "description": "Unexpected internal error."},
    },
)
async def link_analysis(body: LinkAnalysisRequest, request: Request) -> LinkAnalysisResponse:
    return await analyze_website(body.url, request.app.state.settings)


def _error_response(status: int, code: str, detail: str) -> JSONResponse:
    body = LinkAnalysisErrorResponse(code=code, detail=detail)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


async def _request_validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    logger.info("Link analysis failed stage=input error_code=invalid_request")
    return _error_response(422, "invalid_request", "Send a JSON body with a valid HTTP or HTTPS URL.")


async def _analysis_error(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, UnsafeUrlError):
        return _error_response(422, "unsafe_url", str(exc))
    if isinstance(exc, WebsiteDnsError):
        return _error_response(502, "website_dns_error", str(exc))
    if isinstance(exc, WebsiteFetchError):
        status = 504 if exc.code == "timeout" else 502
        code = {"timeout": "website_timeout", "rate_limited": "website_rate_limited"}.get(
            exc.code, exc.code
        )
        return _error_response(status, code, str(exc))
    if isinstance(exc, LLMError):
        if isinstance(exc, LLMConfigurationError):
            return _error_response(
                503, f"llm_{exc.code}", "Analysis is unavailable. Contact the service operator."
            )
        if isinstance(exc, LLMProviderError):
            if exc.code == "timeout":
                return _error_response(504, "llm_timeout", "The analysis provider timed out.")
            return _error_response(
                503, f"llm_{exc.code}", "The analysis provider is temporarily unavailable."
            )
        if isinstance(exc, LLMOutputError):
            return _error_response(
                502, f"llm_{exc.code}", "The analysis provider could not produce a valid result."
            )
        return _error_response(502, exc.code, str(exc))
    return _error_response(500, "internal_error", "An unexpected error occurred during analysis.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    for error_type in (UnsafeUrlError, WebsiteDnsError, WebsiteFetchError, LLMError, AnalysisInternalError):
        app.add_exception_handler(error_type, _analysis_error)
