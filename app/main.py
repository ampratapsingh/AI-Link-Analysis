import uvicorn
from fastapi import FastAPI

from app.config import load_settings
from app.request_limits import AnalysisRequestLimits
from app.routes import register_error_handlers, router

settings = load_settings()
app = FastAPI(
    title=settings.app_name,
    description="Generate structured business and marketing insights from a website URL.",
)
app.state.settings = settings
if settings.is_production:
    app.add_middleware(AnalysisRequestLimits, settings=settings)
register_error_handlers(app)
app.include_router(router)


if __name__ == "__main__":
    log_config = {
        **uvicorn.config.LOGGING_CONFIG,
        "loggers": {
            **uvicorn.config.LOGGING_CONFIG["loggers"],
            "app": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "anthropic": {"level": "WARNING"},
        },
    }
    if settings.is_production:
        for name in ("openai", "httpx", "httpcore"):
            log_config["loggers"][name] = {"level": "WARNING"}
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_config=log_config,
        access_log=not settings.is_production,
    )
