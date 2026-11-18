import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request

from app import config, request_limits
from app.config import Settings
from app.request_limits import AnalysisRequestLimits

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


def guarded_client(**settings):
    app = FastAPI()
    app.state.started = asyncio.Event()
    app.state.release = asyncio.Event()
    app.state.hold = False

    @app.post("/link-analysis")
    async def analyze(request: Request):
        app.state.started.set()
        if app.state.hold:
            await app.state.release.wait()
        return {"bytes": len(await request.body())}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(
        AnalysisRequestLimits, settings=Settings(environment="production", **settings)
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    return app, client


@pytest.mark.parametrize("stream", [False, True])
async def test_oversized_declared_or_streamed_bodies_are_rejected(stream):
    async def chunks():
        yield b"12345"
        yield b"6789"

    app, client = guarded_client(analysis_max_body_bytes=8)
    async with client:
        response = await client.post("/link-analysis", content=chunks() if stream else b"123456789")
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"
        assert not app.state.started.is_set()
        response = await client.post("/link-analysis", content=b"12345678")
    assert response.json() == {"bytes": 8}


async def test_stalled_body_releases_capacity_and_skips_work():
    async def stalled():
        yield b"a"
        await asyncio.Event().wait()

    app, client = guarded_client(analysis_max_in_flight=1, analysis_body_timeout_seconds=0.02)
    async with client:
        response = await client.post("/link-analysis", content=stalled())
        assert response.status_code == 408
        assert response.json()["code"] == "request_body_timeout"
        assert not app.state.started.is_set()
        response = await client.post("/link-analysis", content=b"short")
    assert response.status_code == 200


async def test_busy_analysis_does_not_block_health_or_swagger():
    app, client = guarded_client(analysis_max_in_flight=1)
    app.state.hold = True
    async with client:
        active = asyncio.create_task(client.post("/link-analysis", content=b"one"))
        try:
            await asyncio.wait_for(app.state.started.wait(), timeout=1)
            response = await client.post("/link-analysis", content=b"two")
            assert response.status_code == 503
            assert response.json()["code"] == "analysis_busy"
            assert response.headers["Retry-After"] == "5"
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/docs")).status_code == 200
        finally:
            app.state.release.set()
            first = await active
        assert first.status_code == 200
        assert (await client.post("/link-analysis", content=b"three")).status_code == 200


async def test_shared_rate_limit_expires_and_openapi_stays_available(monkeypatch):
    now = 0
    monkeypatch.setattr(request_limits, "monotonic", lambda: now)
    app, client = guarded_client(analysis_requests_per_hour=1)
    async with client:
        assert (await client.post("/link-analysis", content=b"one")).status_code == 200
        now = 3590
        response = await client.post("/link-analysis", content=b"two")
        assert response.status_code == 429
        assert response.json()["code"] == "analysis_rate_limited"
        assert response.headers["Retry-After"] == "10"
        assert (await client.get("/openapi.json")).status_code == 200
        now = 3600
        assert (await client.post("/link-analysis", content=b"three")).status_code == 200


async def test_production_uses_process_environment_without_dotenv(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: pytest.fail("Read .env in production"))
    settings = config.load_settings()
    assert settings.is_production and settings.port == 10000
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None and settings.search_api_key is None


async def test_invalid_environment_values_are_reported_without_echo(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PORT", "fixture-sensitive-value")
    monkeypatch.setenv("ANALYSIS_REQUESTS_PER_HOUR", "0")
    with pytest.raises(RuntimeError, match="PORT, ANALYSIS_REQUESTS_PER_HOUR") as error:
        config.load_settings()
    assert "fixture-sensitive-value" not in str(error.value)
